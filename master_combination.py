import torch
import torch.nn as nn
import timm
import tome
import time
import gc
import matplotlib.pyplot as plt
import numpy as np
import bitsandbytes as bnb
import types
from datasets import load_dataset
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from tqdm import tqdm

# --- Configuration ---
DATA_FILES = "./data/validation-*.parquet" 
BATCH_SIZE = 64
NUM_WORKERS = 4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TOKENS_PER_IMAGE = 197 # 196 patches + 1 CLS token

# --- Master Sweep Grid (24 Combinations) ---
QUANTIZATIONS = ["FP32", "INT8", "INT4"]
PRUNE_RATIOS = [0.0, 0.10] # 10% physically drops 1 of the 6 heads
TOKEN_REDUCTIONS = ["None", "ToMe_r4", "ToMe_r8", "Mask_75"]

# ==========================================
# 1. Data Loading
# ==========================================
def get_dataloaders():
    print(f"Loading dataset from {DATA_FILES}...")
    dataset = load_dataset("parquet", data_files=DATA_FILES, split="train")
    
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    def preprocess(examples):
        images = [val_transform(img.convert('RGB')) for img in examples['image']]
        return {'image': images, 'label': examples['label']}

    dataset = dataset.with_transform(preprocess)
    
    calib_indices = list(range(500))
    eval_indices = list(range(500, len(dataset)))
    
    calib_loader = DataLoader(Subset(dataset, calib_indices), batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=True)
    eval_loader = DataLoader(Subset(dataset, eval_indices), batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=True)
    
    return calib_loader, eval_loader

# ==========================================
# 2. Compression Modules
# ==========================================
class MaskingTokenDropper:
    """Custom Token Masking Hook (preserves 2D tensor shape)"""
    def __init__(self, keep_ratio):
        self.keep_ratio = keep_ratio

    def hook_fn(self, module, input, output):
        if self.keep_ratio >= 1.0: return output 
        is_tuple = isinstance(output, tuple)
        x = output[0] if is_tuple else output

        B, N, C = x.shape
        patch_tokens = x[:, 1:, :] 
        importance = torch.norm(patch_tokens, p=2, dim=-1)
        num_keep = int((N - 1) * self.keep_ratio)
        
        if num_keep > 0:
            threshold_values, _ = torch.kthvalue(importance, (N - 1) - num_keep + 1, dim=-1, keepdim=True)
            mask = (importance >= threshold_values).float() 
        else:
            mask = torch.zeros_like(importance)
            
        mask = mask.unsqueeze(-1).expand(-1, -1, C)
        masked_patches = patch_tokens * mask
        cls_token = x[:, 0:1, :]
        masked_x = torch.cat([cls_token, masked_patches], dim=1)
        
        return (masked_x,) + output[1:] if is_tuple else masked_x

def calibrate_attention_forward(self, x, size=None, **kwargs):
    """Temporary forward pass to measure Attention Entropy."""
    B, N, C = x.shape
    # CRITICAL FIX: Use out_features instead of weight.shape to avoid INT4 packed tensor crash
    head_dim = self.qkv.out_features // (3 * self.num_heads)
    
    qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, head_dim).permute(2, 0, 3, 1, 4)
    q, k, v = qkv.unbind(0)
    q = q * self.scale
    attn = (q @ k.transpose(-2, -1))
    
    if kwargs.get('attn_mask') is not None: attn = attn + kwargs['attn_mask']
    attn = attn.softmax(dim=-1)

    entropy = -(attn * torch.log(attn + 1e-8)).sum(dim=-1) 
    mean_entropy = entropy.mean(dim=(0, 2)) 

    if not hasattr(self, 'accumulated_entropy'):
        self.accumulated_entropy = mean_entropy
        self.calib_steps = 1
    else:
        self.accumulated_entropy += mean_entropy
        self.calib_steps += 1

    attn = self.attn_drop(attn)
    x_out = (attn @ v).transpose(1, 2).reshape(B, N, self.num_heads * head_dim)
    x_out = self.proj(x_out)
    x_out = self.proj_drop(x_out)
    return x_out

def unified_attention_forward(self, x, size=None, **kwargs):
    """
    BULLETPROOF FORWARD PASS.
    Dynamically handles missing pruned heads AND strictly satisfies Facebook's ToMe Tuple requirements.
    It reads out_features safely so bitsandbytes 4-bit packing doesn't break the reshape math.
    """
    B, N, C = x.shape
    # CRITICAL FIX: Use out_features instead of weight.shape to avoid INT4 packed tensor crash
    head_dim = self.qkv.out_features // (3 * self.num_heads)
    
    qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, head_dim).permute(2, 0, 3, 1, 4)
    q, k, v = qkv.unbind(0)
    q = q * self.scale
    attn = (q @ k.transpose(-2, -1))
    
    # Facebook ToMe Compatibility Mask
    if size is not None:
        attn = attn + size.log()[:, None, None, :, 0]
        
    if kwargs.get('attn_mask') is not None: 
        attn = attn + kwargs['attn_mask']
        
    attn = attn.softmax(dim=-1)
    attn = self.attn_drop(attn)
    
    x_out = (attn @ v).transpose(1, 2).reshape(B, N, self.num_heads * head_dim)
    x_out = self.proj(x_out)
    x_out = self.proj_drop(x_out)
    
    # If ToMe is active, return the required Metric Tuple.
    if getattr(self, 'is_tome_active', False):
        metric = k.mean(dim=1)
        return x_out, metric
        
    return x_out

def physically_prune_heads(model, prune_ratio):
    for block in model.blocks:
        attn = block.attn
        num_heads = attn.num_heads
        # Read true out_features before slicing
        head_dim = attn.qkv.out_features // (3 * num_heads)
        num_keep = max(1, int(num_heads * (1 - prune_ratio)))
        
        if num_keep == num_heads: continue
            
        avg_entropy = attn.accumulated_entropy / attn.calib_steps
        _, sorted_indices = torch.sort(avg_entropy, descending=False)
        kept_heads = sorted_indices[:num_keep].tolist()
        kept_heads.sort() 
        
        D = num_heads * head_dim
        qkv_indices, proj_indices = [], []
        
        for h in kept_heads:
            start, end = h * head_dim, (h + 1) * head_dim
            qkv_indices.extend(range(start, end))             
            qkv_indices.extend(range(D + start, D + end))     
            qkv_indices.extend(range(2*D + start, 2*D + end)) 
            proj_indices.extend(range(start, end))

        device = attn.qkv.weight.device
        qkv_idx = torch.tensor(qkv_indices, device=device)
        proj_idx = torch.tensor(proj_indices, device=device)

        # Slice weights 
        attn.qkv.weight = nn.Parameter(torch.index_select(attn.qkv.weight, dim=0, index=qkv_idx))
        if attn.qkv.bias is not None:
            attn.qkv.bias = nn.Parameter(torch.index_select(attn.qkv.bias, dim=0, index=qkv_idx))
        attn.proj.weight = nn.Parameter(torch.index_select(attn.proj.weight, dim=1, index=proj_idx))
        
        # Explicitly update out_features so Downstream INT4/INT8 layers calibrate correctly
        attn.num_heads = num_keep
        attn.qkv.out_features = len(qkv_idx)
        attn.proj.in_features = len(proj_idx)
        del attn.accumulated_entropy, attn.calib_steps

def replace_linear_with_bnb(module, quant_type="int8"):
    for name, child in module.named_children():
        if isinstance(child, nn.Linear):
            has_bias = child.bias is not None
            if quant_type == "int8":
                quant_layer = bnb.nn.Linear8bitLt(child.in_features, child.out_features, bias=has_bias, has_fp16_weights=False, threshold=6.0)
            elif quant_type == "int4":
                quant_layer = bnb.nn.Linear4bit(child.in_features, child.out_features, bias=has_bias, quant_type="nf4")
            
            quant_layer.weight.data = child.weight.data.clone()
            if has_bias: quant_layer.bias.data = child.bias.data.clone()
            setattr(module, name, quant_layer.to(DEVICE))
        else:
            replace_linear_with_bnb(child, quant_type)

# ==========================================
# 3. Evaluation & Orchestration
# ==========================================
def evaluate_model(model, eval_loader, combo_name):
    model.eval()
    correct, total = 0, 0
    timings = []
    
    torch.cuda.reset_peak_memory_stats()
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

    with torch.no_grad():
        for i, batch in enumerate(tqdm(eval_loader, desc=f"Eval {combo_name}", leave=False)):
            inputs, targets = batch['image'].to(DEVICE), batch['label'].to(DEVICE)

            if i > 5: starter.record()
            outputs = model(inputs)
            if i > 5:
                ender.record()
                torch.cuda.synchronize()
                timings.append(starter.elapsed_time(ender))

            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    acc = 100. * correct / total
    lat = sum(timings) / len(timings) if timings else 0 
    throughput = (BATCH_SIZE / lat) * 1000 if lat > 0 else 0
    tok_sec = throughput * TOKENS_PER_IMAGE
    peak_vram = torch.cuda.max_memory_allocated() / (1024 * 1024)
    params = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    
    return acc, lat, throughput, tok_sec, peak_vram, params

def plot_master_results(results):
    fig, axes = plt.subplots(1, 3, figsize=(24, 8))
    fig.suptitle('Master Sweep: Interaction Effects (Standard ViT)', fontsize=20, fontweight='bold')
    
    names = list(results.keys())
    accuracies = [results[n]["Acc"] for n in names]
    vrams = [results[n]["VRAM"] for n in names]
    quant_colors = {"FP32": "#1f77b4", "INT8": "#ff7f0e", "INT4": "#2ca02c"}

    # Plot 1: Pareto Curve
    ax1 = axes[0]
    for name in names:
        q_type = name.split("|")[0]
        ax1.scatter(results[name]["Lat"], results[name]["Acc"], color=quant_colors[q_type], s=120, edgecolors='black', linewidth=0.5, label=q_type)
    
    handles, labels = ax1.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax1.legend(by_label.values(), by_label.keys(), title="Quantization", loc="lower left")
    
    ax1.set_title("Pareto Frontier: Accuracy vs Latency", fontsize=16)
    ax1.set_xlabel("Average Batch Latency (ms) ↓", fontsize=14)
    ax1.set_ylabel("Top-1 Accuracy (%) ↑", fontsize=14)
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.invert_xaxis()

    # Plot 2: Peak VRAM
    ax2 = axes[1]
    x_pos = np.arange(len(names))
    colors = [quant_colors[n.split("|")[0]] for n in names]
    
    ax2.bar(x_pos, vrams, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
    ax2.set_title("Peak VRAM Consumption", fontsize=16)
    ax2.set_ylabel("Memory Allocated (MB) ↓", fontsize=14)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(names, rotation=45, ha="right", fontsize=9)
    ax2.grid(axis='y', linestyle='--', alpha=0.6)

    # Plot 3: Accuracy Impact
    ax3 = axes[2]
    ax3.bar(x_pos, accuracies, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
    ax3.set_title("Top-1 Accuracy Retention", fontsize=16)
    ax3.set_ylabel("Accuracy (%) ↑", fontsize=14)
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(names, rotation=45, ha="right", fontsize=9)
    ax3.grid(axis='y', linestyle='--', alpha=0.6)
    ax3.set_ylim(0, 100)

    plt.tight_layout()
    plt.savefig("master_interaction_sweep.png", dpi=300, bbox_inches='tight')
    print(f"\n[Success] High-Res 3-Panel Graph saved as 'master_interaction_sweep.png'")

def main():
    calib_loader, eval_loader = get_dataloaders()
    results = {}

    print(f"\n{'='*115}")
    print(f"{'STARTING MASTER INTERACTION SWEEP (24 COMBINATIONS)':^115}")
    print(f"{'='*115}\n")

    for q in QUANTIZATIONS:
        for p in PRUNE_RATIOS:
            for tr in TOKEN_REDUCTIONS:
                combo_name = f"{q}|P{int(p*100)}|{tr}"
                print(f"--- Running: {combo_name} ---")
                
                # Zero out GPU memory 
                torch.cuda.empty_cache()
                gc.collect()

                # 1. Load Base Model
                model = timm.create_model('vit_small_patch16_224', pretrained=True).to(DEVICE)
                hook_handle = None
                
                # 2. Apply Pruning 
                if p > 0.0:
                    for blk in model.blocks: blk.attn.forward = types.MethodType(calibrate_attention_forward, blk.attn)
                    with torch.no_grad():
                        for batch in tqdm(calib_loader, desc="Calibrating", leave=False): 
                            model(batch['image'].to(DEVICE))
                    physically_prune_heads(model, p)

                # 3. Apply ToMe
                is_tome = "ToMe" in tr
                if is_tome:
                    r_val = int(tr.split("_r")[1])
                    tome.patch.timm(model)
                    model.r = r_val

                # 4. Attach Unified Forward Hook (Safely handles Pruning + ToMe)
                for blk in model.blocks:
                    blk.attn.is_tome_active = is_tome  # Explicit Flag
                    blk.attn.forward = types.MethodType(unified_attention_forward, blk.attn)

                # 5. Apply Token Masking
                if "Mask" in tr:
                    retention = int(tr.split("_")[1]) / 100.0
                    dropper = MaskingTokenDropper(keep_ratio=retention)
                    hook_handle = model.blocks[5].register_forward_hook(dropper.hook_fn)

                # 6. Apply Quantization
                if q != "FP32":
                    quant_type = "int8" if q == "INT8" else "int4"
                    replace_linear_with_bnb(model, quant_type)
                    with torch.no_grad(): # Warmup
                        for _ in range(3): model(next(iter(calib_loader))['image'].to(DEVICE))

                # 7. Evaluate
                acc, lat, thru, tok, vram, params = evaluate_model(model, eval_loader, combo_name)
                results[combo_name] = {"Acc": acc, "Lat": lat, "Thru": thru, "Tok/s": tok, "VRAM": vram, "Params": params}
                
                if hook_handle is not None: hook_handle.remove()
                del model

    print(f"\n{'='*115}")
    print(f"{'MASTER INTERACTION SWEEP RESULTS':^115}")
    print(f"{'='*115}")
    print(f"{'Configuration':<20} | {'Acc (%)':<8} | {'Lat (ms)':<8} | {'Img/sec':<8} | {'Tok/sec':<10} | {'VRAM (MB)':<10} | {'Params (M)'}")
    print("-" * 115)
    for name, m in results.items():
        print(f"{name:<20} | {m['Acc']:>8.2f} | {m['Lat']:>8.2f} | {m['Thru']:>8.0f} | {m['Tok/s']:>10.0f} | {m['VRAM']:>10.2f} | {m['Params']:>8.2f}")

    plot_master_results(results)

if __name__ == "__main__":
    main()