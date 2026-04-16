import torch
import timm
import time
import gc
import matplotlib.pyplot as plt
from datasets import load_dataset
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import sys

# --- Configuration ---
DATA_FILES = "./data/validation-*.parquet" 
BATCH_SIZE = 64
NUM_WORKERS = 4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# 1. Data Loading
# ==========================================
def get_dataloader():
    print(f"Loading parquet dataset from {DATA_FILES}...")
    dataset = load_dataset("parquet", data_files=DATA_FILES, split="train")
    
    print(f"Total images loaded: {len(dataset)}")
    if len(dataset) != 50000:
        print("WARNING: Dataset length is not 50,000! Check your parquet files.")

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
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=True)
    return dataloader

# ==========================================
# 2. Evaluation Engine
# ==========================================
def evaluate_model(model, dataloader, model_name):
    print(f"\n{'='*60}")
    print(f"Evaluating: {model_name}")
    print(f"{'='*60}")
    
    model.eval()
    correct = 0
    total = 0
    timings = []
    
    use_cuda_timing = DEVICE.type == 'cuda'
    if use_cuda_timing:
        starter = torch.cuda.Event(enable_timing=True)
        ender = torch.cuda.Event(enable_timing=True)
    
    # Calculate parameter count (in Millions)
    params_m = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    print(f"Parameters: {params_m:.2f} M")

    with torch.no_grad():
        for i, batch in enumerate(tqdm(dataloader, desc="Inference")):
            inputs = batch['image'].to(DEVICE)
            targets = batch['label'].to(DEVICE)

            # --- Timing Start ---
            if i > 5: # Warmup for the first 5 batches
                if use_cuda_timing:
                    starter.record()
                else:
                    start_time = time.time()
                    
            outputs = model(inputs)
            
            # --- Timing End ---
            if i > 5:
                if use_cuda_timing:
                    ender.record()
                    torch.cuda.synchronize()
                    timings.append(starter.elapsed_time(ender))
                else:
                    end_time = time.time()
                    timings.append((end_time - start_time) * 1000) # Convert to ms

            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    # Calculate final metrics
    top1_acc = 100. * correct / total
    avg_latency_ms = sum(timings) / len(timings) if timings else 0 
    throughput = (BATCH_SIZE / avg_latency_ms) * 1000 if avg_latency_ms > 0 else 0
    
    print(f"\n--- Final Results for {model_name} ---")
    print(f"Top-1 Accuracy:    {top1_acc:.2f}%")
    print(f"Avg Batch Latency: {avg_latency_ms:.2f} ms")
    print(f"Throughput:        {throughput:.0f} img/sec")
    print(f"Parameters:        {params_m:.2f} M")
    
    return top1_acc, avg_latency_ms, throughput, params_m

# ==========================================
# 3. Dynamic Plotting
# ==========================================
def plot_baseline_pareto(results_dict):
    """
    Plots the Accuracy vs. Latency baseline dynamically using the real data.
    """
    plt.figure(figsize=(10, 6))
    
    styles = {
        "Standard ViT (Small)": {"color": "#d62728", "marker": "o"}, # Red
        "EfficientFormer (L1)": {"color": "#1f77b4", "marker": "s"}, # Blue
        "EfficientViT (M4)":    {"color": "#2ca02c", "marker": "^"}  # Green
    }

    for model_name, metrics in results_dict.items():
        plt.scatter(
            metrics["latency"], 
            metrics["accuracy"], 
            label=model_name,
            color=styles[model_name]["color"],
            marker=styles[model_name]["marker"],
            s=180, 
            edgecolors='black',
            zorder=5
        )
        
        # Add labels dynamically next to the points
        plt.annotate(
            f"{model_name}\n({metrics['accuracy']:.1f}%)",
            (metrics["latency"], metrics["accuracy"]),
            xytext=(10, -10), 
            textcoords='offset points',
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8)
        )

    plt.title("Baseline Accuracy vs. Latency (Uncompressed)", fontsize=16, fontweight='bold')
    plt.xlabel("Average Batch Latency (ms) ↓ [Lower is Better]", fontsize=14)
    plt.ylabel("Top-1 Accuracy (%) ↑ [Higher is Better]", fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc="lower right", fontsize=11)
    
    # Invert X-axis so the best models (fastest) appear on the right
    plt.gca().invert_xaxis() 

    output_filename = "baseline_pareto_curve.png"
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"\n[Success] Graph successfully saved as '{output_filename}'")

# ==========================================
# 4. Main Orchestrator
# ==========================================
def main():
    if not torch.cuda.is_available():
        print("\n[WARNING] CUDA is not available. Running on CPU. Latency will be slow.\n")
        
    dataloader = get_dataloader()
    results = {}

    # Updated with fully qualified timm registry names to prevent Unknown Model crashes
    model_registry = {
        "Standard ViT (Small)": 'vit_small_patch16_224',
        "EfficientFormer (L1)": 'efficientformer_l1.snap_dist_in1k',
        "EfficientViT (M4)":    'efficientvit_m4.r224_in1k'
    }

    for name, model_id in model_registry.items():
        try:
            # Load model dynamically to save RAM
            model = timm.create_model(model_id, pretrained=True).to(DEVICE)
        except RuntimeError as e:
            print(f"\n[ERROR] Could not load {model_id}. Ensure timm is fully updated: `pip install --upgrade timm`")
            print(f"Exact Error: {e}")
            sys.exit(1)
            
        # Evaluate
        acc, lat, thru, params = evaluate_model(model, dataloader, name)
        
        # Save metrics for plotting
        results[name] = {"accuracy": acc, "latency": lat, "throughput": thru, "params": params}
        
        # Deep clean memory before loading the next model to prevent VRAM overflow
        del model
        torch.cuda.empty_cache()
        gc.collect()

    # Pass the gathered data directly to the plotter
    plot_baseline_pareto(results)

if __name__ == "__main__":
    main()