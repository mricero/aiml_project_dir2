import torch
import torch.nn as nn
import timm
import types
import time
import matplotlib.pyplot as plt
from datasets import load_dataset
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from tqdm import tqdm

# --- Configuration ---
DATA_FILES = "./data/validation-*.parquet" 
BATCH_SIZE = 64
NUM_WORKERS = 4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PRUNE_RATIOS = [0.1, 0.25, 0.50, 0.75] 

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

# --- Custom Forward: Phase 1 (Calibration) ---
def calibrate_attention_forward(self, x, **kwargs):
    B, N, C = x.shape
    head_dim = C // self.num_heads
    
    qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, head_dim).permute(2, 0, 3, 1, 4)
    q, k, v = qkv.unbind(0)
    
    q = q * self.scale
    attn = (q @ k.transpose(-2, -1))
    
    # Safely handle the attention mask passed by newer versions of timm
    if kwargs.get('attn_mask') is not None:
        attn = attn + kwargs['attn_mask']
        
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
    x = (attn @ v).transpose(1, 2).reshape(B, N, C)
    x = self.proj(x)
    x = self.proj_drop(x)
    return x

# --- Custom Forward: Phase 2 (Inference after Pruning) ---
def pruned_attention_forward(self, x, **kwargs):
    B, N, C = x.shape
    
    # Calculate true head_dim based on the sliced qkv matrix
    head_dim = self.qkv.weight.shape[0] // (3 * self.num_heads)
    
    qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, head_dim).permute(2, 0, 3, 1, 4)
    q, k, v = qkv.unbind(0)
    
    q = q * self.scale
    attn = (q @ k.transpose(-2, -1))

    # Safely handle the attention mask passed by newer versions of timm
    if kwargs.get('attn_mask') is not None:
        attn = attn + kwargs['attn_mask']

    attn = attn.softmax(dim=-1)
    attn = self.attn_drop(attn)
    
    # Re-assemble the smaller sequence of heads
    x = (attn @ v).transpose(1, 2).reshape(B, N, self.num_heads * head_dim)
    x = self.proj(x) 
    x = self.proj_drop(x)
    return x

def physically_prune_heads(model, prune_ratio):
    for block in model.blocks:
        attn = block.attn
        num_heads = attn.num_heads
        head_dim = attn.qkv.in_features // num_heads
        
        num_keep = max(1, int(num_heads * (1 - prune_ratio)))
        if num_keep == num_heads:
            continue
            
        avg_entropy = attn.accumulated_entropy / attn.calib_steps
        
        _, sorted_indices = torch.sort(avg_entropy, descending=False)
        kept_heads = sorted_indices[:num_keep].tolist()
        kept_heads.sort() 
        
        D = num_heads * head_dim
        qkv_indices = []
        proj_indices = []
        
        for h in kept_heads:
            start = h * head_dim
            end = (h + 1) * head_dim
            
            qkv_indices.extend(range(start, end))             
            qkv_indices.extend(range(D + start, D + end))     
            qkv_indices.extend(range(2*D + start, 2*D + end)) 
            
            proj_indices.extend(range(start, end))

        device = attn.qkv.weight.device
        qkv_idx_tensor = torch.tensor(qkv_indices, device=device)
        proj_idx_tensor = torch.tensor(proj_indices, device=device)

        attn.qkv.weight = nn.Parameter(torch.index_select(attn.qkv.weight, dim=0, index=qkv_idx_tensor))
        if attn.qkv.bias is not None:
            attn.qkv.bias = nn.Parameter(torch.index_select(attn.qkv.bias, dim=0, index=qkv_idx_tensor))

        attn.proj.weight = nn.Parameter(torch.index_select(attn.proj.weight, dim=1, index=proj_idx_tensor))
        
        attn.num_heads = num_keep
        
        del attn.accumulated_entropy
        del attn.calib_steps

def evaluate_model(model, eval_loader, ratio_name):
    model.eval()
    correct, total = 0, 0
    timings = []
    
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

    with torch.no_grad():
        for i, batch in enumerate(tqdm(eval_loader, desc=f"Evaluating {ratio_name}")):
            inputs = batch['image'].to(DEVICE)
            targets = batch['label'].to(DEVICE)

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
    
    params = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    return acc, lat, params

def plot_pruning_results(results):
    """Generates the 2-panel Pareto and Parameter graph automatically."""
    ratios = list(results.keys())
    accuracies = [results[r]["Accuracy"] for r in ratios]
    latencies = [results[r]["Latency (ms)"] for r in ratios]
    params = [results[r]["Params (M)"] for r in ratios]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle('Attention Head Pruning Trade-offs (Standard ViT)', fontsize=16, fontweight='bold')

    # --- Plot 1: Accuracy vs Latency ---
    colors = ['blue', 'orange', 'green', 'red']
    ax1.plot(latencies, accuracies, linestyle='--', color='gray', zorder=1)
    
    for i in range(len(ratios)):
        ax1.scatter(latencies[i], accuracies[i], color=colors[i], s=150, zorder=2, label=ratios[i])
        ax1.annotate(
            f"{ratios[i]}\n{accuracies[i]:.2f}%", 
            (latencies[i], accuracies[i]),
            xytext=(10, 5), textcoords='offset points', fontweight='bold'
        )

    ax1.set_title("Accuracy vs. Latency", fontsize=14)
    ax1.set_xlabel("Average Batch Latency (ms) ↓ [Lower is Better]", fontsize=12)
    ax1.set_ylabel("Top-1 Accuracy (%) ↑ [Higher is Better]", fontsize=12)
    ax1.grid(True, linestyle='--', alpha=0.6)

    # --- Plot 2: Parameter Count ---
    x_pos = range(len(ratios))
    bars = ax2.bar(x_pos, params, color=colors, alpha=0.8, width=0.6)

    for bar in bars:
        height = bar.get_height()
        ax2.annotate(
            f'{height:.2f} M',
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3), textcoords="offset points",
            ha='center', va='bottom', fontweight='bold'
        )

    ax2.set_title("Physical Parameter Count", fontsize=14)
    ax2.set_ylabel("Total Parameters (Millions) ↓ [Lower is Better]", fontsize=12)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(ratios, fontsize=11)
    ax2.grid(axis='y', linestyle='--', alpha=0.6)
    ax2.set_ylim(0, max(params) * 1.15) 

    plt.tight_layout()
    output_filename = "entropy_pruning_metrics.png"
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"\n[Success] Graph successfully saved as '{output_filename}'!")

def main():
    calib_loader, eval_loader = get_dataloaders()
    results = {}

    for ratio in PRUNE_RATIOS:
        print(f"\n{'='*60}")
        print(f"Testing Prune Ratio: {int(ratio*100)}% of Heads Removed")
        print(f"{'='*60}")
        
        model = timm.create_model('vit_small_patch16_224', pretrained=True).to(DEVICE)
        
        if ratio > 0.0:
            print("[1/3] Attaching Entropy Hooks...")
            for block in model.blocks:
                block.attn.forward = types.MethodType(calibrate_attention_forward, block.attn)
                
            print("[2/3] Calibrating Entropy (500 images)...")
            with torch.no_grad():
                for batch in tqdm(calib_loader, desc="Calibrating", leave=False):
                    model(batch['image'].to(DEVICE))
                    
            print("[3/3] Slicing Weight Matrices...")
            physically_prune_heads(model, ratio)
            
            # Inject the safe, pruned-aware forward pass instead of timm's default
            for block in model.blocks:
                block.attn.forward = types.MethodType(pruned_attention_forward, block.attn)
        
        acc, lat, params = evaluate_model(model, eval_loader, f"{int(ratio*100)}% Pruned")
        results[f"{int(ratio*100)}%"] = {"Accuracy": acc, "Latency (ms)": lat, "Params (M)": params}

    print(f"\n{'='*65}")
    print(f"{'Entropy Pruning Trade-off Matrix (Standard ViT)':^65}")
    print(f"{'='*65}")
    print(f"{'Pruned %':<12} | {'Accuracy (%)':<14} | {'Latency (ms)':<14} | {'Params (M)':<12}")
    print("-" * 65)
    for p, metrics in results.items():
        print(f"{p:<12} | {metrics['Accuracy']:>12.2f} % | {metrics['Latency (ms)']:>12.2f} ms | {metrics['Params (M)']:>10.2f} M")

    # --- Generate the Graph ---
    plot_pruning_results(results)

if __name__ == "__main__":
    main()