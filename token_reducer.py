import torch
import timm
import time
import matplotlib.pyplot as plt
import numpy as np
from datasets import load_dataset
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

# --- Configuration ---
DATA_FILES = "./data/validation-*.parquet" 
BATCH_SIZE = 64
NUM_WORKERS = 4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Retention ratios (1.0 = Baseline, dropping to 25% retention)
RETENTION_RATIOS = [1.0, 0.75, 0.50, 0.25]
DROP_LAYER_IDX = 5 # Apply the drop after the 6th block (0-indexed)

def get_dataloader():
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
    return DataLoader(dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=True)

class MaskingTokenDropper:
    """
    A PyTorch forward hook that identifies the least important tokens 
    via L2-norm and masks them to 0.0, preserving the 2D spatial tensor shape.
    """
    def __init__(self, keep_ratio):
        self.keep_ratio = keep_ratio

    def hook_fn(self, module, input, output):
        # Do nothing for 100% baseline retention
        if self.keep_ratio >= 1.0:
            return output 

        # output shape: (Batch, Seq_Len, Dim)
        B, N, C = output.shape
        
        # 1. Calculate Token Importance (Exclude CLS token at index 0)
        patch_tokens = output[:, 1:, :] 
        importance = torch.norm(patch_tokens, p=2, dim=-1) # Shape: (Batch, Seq_Len - 1)
        
        # 2. Determine threshold cutoff
        num_keep = int((N - 1) * self.keep_ratio)
        
        # 3. Create Binary Mask
        if num_keep > 0:
            threshold_values, _ = torch.kthvalue(importance, (N - 1) - num_keep + 1, dim=-1, keepdim=True)
            mask = (importance >= threshold_values).float() # 1.0 for keep, 0.0 for drop
        else:
            mask = torch.zeros_like(importance)

        # Expand mask to match embedding dimension
        mask = mask.unsqueeze(-1).expand(-1, -1, C)
        
        # 4. Apply Mask & Reattach CLS token
        masked_patches = patch_tokens * mask
        cls_token = output[:, 0:1, :]
        
        return torch.cat([cls_token, masked_patches], dim=1)

def evaluate_model(model, dataloader, ratio_name):
    model.eval()
    correct, total = 0, 0
    timings = []
    
    use_cuda_timing = DEVICE.type == 'cuda'
    if use_cuda_timing:
        starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

    with torch.no_grad():
        for i, batch in enumerate(tqdm(dataloader, desc=f"Evaluating {ratio_name}")):
            inputs = batch['image'].to(DEVICE)
            targets = batch['label'].to(DEVICE)

            if i > 5:
                if use_cuda_timing: starter.record()
                else: start_time = time.time()
                    
            outputs = model(inputs)
            
            if i > 5:
                if use_cuda_timing:
                    ender.record()
                    torch.cuda.synchronize()
                    timings.append(starter.elapsed_time(ender))
                else:
                    timings.append((time.time() - start_time) * 1000)

            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    acc = 100. * correct / total
    lat = sum(timings) / len(timings) if timings else 0 
    throughput = (BATCH_SIZE / lat) * 1000 if lat > 0 else 0
    
    return acc, lat, throughput

def plot_custom_dropping_results(results):
    """Generates a 2-panel Pareto (Accuracy vs Latency) and Throughput Graph."""
    ratios = list(results.keys())
    labels = [f"{int(r*100)}% Retained" if r < 1.0 else "Baseline" for r in ratios]
    accuracies = [results[r]["Accuracy"] for r in ratios]
    latencies = [results[r]["Latency (ms)"] for r in ratios]
    throughputs = [results[r]["Throughput"] for r in ratios]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle('Custom Threshold Masking Trade-offs (Standard ViT)', fontsize=16, fontweight='bold')
    colors = ['blue', 'orange', 'green', 'red']

    # --- Plot 1: Accuracy vs Latency Pareto Curve ---
    ax1.plot(latencies, accuracies, linestyle='--', color='gray', zorder=1)
    for i in range(len(ratios)):
        ax1.scatter(latencies[i], accuracies[i], color=colors[i], s=150, zorder=2, label=labels[i])
        ax1.annotate(
            f"{labels[i]}\n{accuracies[i]:.2f}%", 
            (latencies[i], accuracies[i]),
            xytext=(10, 5), textcoords='offset points', fontweight='bold'
        )

    ax1.set_title("Accuracy vs. Latency", fontsize=14)
    ax1.set_xlabel("Average Batch Latency (ms) ↓ [Lower is Better]", fontsize=12)
    ax1.set_ylabel("Top-1 Accuracy (%) ↑ [Higher is Better]", fontsize=12)
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.invert_xaxis() # Standard Pareto orientation (Best models top-right)

    # --- Plot 2: Throughput Bar Chart ---
    x_pos = np.arange(len(ratios))
    bars = ax2.bar(x_pos, throughputs, color=colors, alpha=0.8, width=0.6)
    
    for bar in bars:
        height = bar.get_height()
        ax2.annotate(
            f'{height:.0f} img/s',
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3), textcoords="offset points",
            ha='center', va='bottom', fontweight='bold'
        )

    ax2.set_title("Inference Throughput", fontsize=14)
    ax2.set_ylabel("Images Processed per Second ↑ [Higher is Better]", fontsize=12)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(labels, fontsize=11)
    ax2.grid(axis='y', linestyle='--', alpha=0.6)
    ax2.set_ylim(0, max(throughputs) * 1.15) 

    plt.tight_layout()
    output_filename = "custom_masking_metrics.png"
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"\n[Success] Graph successfully saved as '{output_filename}'!")

def main():
    dataloader = get_dataloader()
    results = {}

    print(f"\n{'='*60}")
    print(f"{'Starting Custom Threshold Masking Sweep':^60}")
    print(f"{'='*60}")

    # Load the model once
    model = timm.create_model('vit_small_patch16_224', pretrained=True).to(DEVICE)
    target_layer = model.blocks[DROP_LAYER_IDX]

    for ratio in RETENTION_RATIOS:
        ratio_label = f"{int(ratio*100)}% Retained" if ratio < 1.0 else "Baseline"
        print(f"\nTesting Configuration: {ratio_label}")
        
        # 1. Initialize the dropper for this specific ratio
        dropper = MaskingTokenDropper(keep_ratio=ratio)
        
        # 2. Attach the hook
        handle = target_layer.register_forward_hook(dropper.hook_fn)
        
        # 3. Evaluate
        acc, lat, thru = evaluate_model(model, dataloader, ratio_label)
        results[ratio] = {"Accuracy": acc, "Latency (ms)": lat, "Throughput": thru}
        
        # 4. CRITICAL: Remove the hook so they don't stack during the next loop!
        handle.remove()

    # --- Print Final Trade-off Matrix ---
    print(f"\n{'='*65}")
    print(f"{'Custom Masking Trade-off Matrix (Standard ViT)':^65}")
    print(f"{'='*65}")
    print(f"{'Retained %':<12} | {'Accuracy (%)':<14} | {'Latency (ms)':<14} | {'Throughput':<12}")
    print("-" * 65)
    for r, metrics in results.items():
        label = f"{int(r*100)}%" if r < 1.0 else "Baseline"
        print(f"{label:<12} | {metrics['Accuracy']:>12.2f} % | {metrics['Latency (ms)']:>12.2f} ms | {metrics['Throughput']:>6.0f} img/s")

    # Generate Graph
    plot_custom_dropping_results(results)

if __name__ == "__main__":
    main()