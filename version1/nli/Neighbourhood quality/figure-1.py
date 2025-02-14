import matplotlib.pyplot as plt
import numpy as np
import glob
import os

def extract_epochs_required(val_accuracies, neighborhood_quality):
    """Determine the epoch where early stopping occurs for each method."""
    traditional_epochs = next((i for i in range(len(val_accuracies) - 1) if val_accuracies[i] > val_accuracies[i + 1]), len(val_accuracies))
    nq_epochs = next((i for i in range(len(neighborhood_quality) - 1) if neighborhood_quality[i] > neighborhood_quality[i + 1]), len(neighborhood_quality))
    return traditional_epochs, nq_epochs

def compute_training_time(embedding_dim, traditional_epochs, nq_epochs):
    """Compute training time based on forward passes."""
    traditional_time = 2 * traditional_epochs
    nq_time = nq_epochs + 20 * embedding_dim
    return traditional_time, nq_time

def generate_table(base_dir, metrics_file, embedding_dim, model_used):
    """Generate and print the comparison table for different models."""
    epochs = list(range(1, 13)) + [15, 18, 21, 24, 27, 30]
    
    file_pattern = os.path.join(base_dir, f"results/{model_used}/knn_accuracies_model*.txt")
    files = sorted(glob.glob(file_pattern), key=lambda x: int(x.split("model")[-1].split(".")[0]))
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(files)))
    plt.figure(figsize=(10, 6))
    
    for i, file in enumerate(files):
        epoch = int(file.split("model")[-1].split(".")[0])
        
        with open(file, 'r') as f:
            lines = f.readlines()
        
        k_values = []
        accuracies = []
        record_data = False
        
        for line in lines:
            if line.startswith("Percentage:"):
                percentage = float(line.strip().split(":")[1].strip().replace("%", ""))
                record_data = (percentage == 100.0)
            
            if record_data and "k=" in line:
                try:
                    parts = line.strip().split()
                    k_value = int(parts[0].split('=')[1])
                    accuracy = float(parts[2])
                    k_values.append(k_value)
                    accuracies.append(accuracy)
                except (IndexError, ValueError):
                    print(f"Skipping malformed line in {file}: {line.strip()}")
        
        if not k_values:
            print(f"Warning: No valid data found in {file} for Percentage=100%, skipping this file.")
            continue
        
        plt.plot(k_values, accuracies, label=f"Epoch {epoch}", color=colors[i], marker='o')
    
    plt.xlabel("K (Number of Neighbors)", fontsize=12)
    plt.ylabel("Neighborhood Quality", fontsize=12)
    plt.title(f"{model_used} Neigh. Quality vs K for Percentage=100%", fontsize=14)
    plt.xticks(k_values if len(k_values) > 0 else [])
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(title="Epoch", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    output_path = os.path.join(base_dir, f"results/{model_used}/figure1-{model_used}.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Plot saved as {output_path}")
    plt.show()

# Example usage
base_dir = "results - nli/"
metrics_file = "/home/gpuuser1/gpuuser1_a/sandeep/sandeep/nli/FFNN/logs/metrics.txt"
embedding_dim = 30
model_used = "FFNN"
generate_table(base_dir, metrics_file, embedding_dim, model_used)
