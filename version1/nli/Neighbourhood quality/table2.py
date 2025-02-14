import numpy as np
import glob
import os

def extract_epochs_required(val_accuracies, neighborhood_quality):
    """Determine the epoch where early stopping occurs for each method."""
    traditional_epochs = next((i + 1 for i in range(len(val_accuracies) - 1) if val_accuracies[i] > val_accuracies[i + 1]), len(val_accuracies))
    nq_epochs = next((i + 1 for i in range(len(neighborhood_quality) - 1) if neighborhood_quality[i] > neighborhood_quality[i + 1]), len(neighborhood_quality))
    return traditional_epochs, nq_epochs

def compute_training_time(embedding_dim, traditional_epochs, nq_epochs):
    """Compute training time based on forward passes."""
    traditional_time = 2 * traditional_epochs
    nq_time = nq_epochs + 20 * embedding_dim
    return traditional_time, nq_time

def extract_neighborhood_quality(base_dir, model_used, epochs):
    """Extract k=20 accuracy values from knn_accuracies_model_{i}.txt files."""
    neighborhood_quality = []
    for epoch in epochs:
        file_path = os.path.join(base_dir, model_used, f"knn_accuracies_model{epoch}.txt")
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            continue
        
        with open(file_path, 'r') as f:
            lines = f.readlines()
        
        for line in lines:
            if line.startswith("k=20"):
                accuracy = float(line.strip().split()[2])
                neighborhood_quality.append(accuracy)
                break
    print("Extracted Neighborhood Quality:", neighborhood_quality)
    return neighborhood_quality

def extract_metrics(metrics_file, epochs):
    """Extract validation and test accuracies from metrics.txt."""
    val_accuracies = []
    test_accuracies = []
    
    with open(metrics_file, 'r') as f:
        lines = f.readlines()
    
    for epoch in epochs:
        val_acc = None
        test_acc = None
        
        for line in reversed(lines):
            if f"Epoch: [{epoch}/" in line and "Validation Acc:" in line:
                val_acc = float(line.strip().split()[-1]) / 100
                break
        
        for line in lines:
            if f"test accuracy at epoch {epoch} is" in line:
                test_acc = float(line.strip().split()[-1]) / 100
                break
        
        if val_acc is not None:
            val_accuracies.append(val_acc)
        else:
            print(f"Warning: Could not find validation accuracy for epoch {epoch} in {metrics_file}")
        
        if test_acc is not None:
            test_accuracies.append(test_acc)
        else:
            print(f"Warning: Could not find test accuracy for epoch {epoch} in {metrics_file}")
    
    print("Extracted Validation Accuracies:", val_accuracies)
    print("Extracted Test Accuracies:", test_accuracies)
    return val_accuracies, test_accuracies

def generate_table(base_dir, metrics_file, embedding_dim, model_used):
    """Generate and print the comparison table for different models."""
    epochs = list(range(1, 13)) + [15, 18, 21, 24, 27, 30]
    
    neighborhood_quality = extract_neighborhood_quality(base_dir, model_used, epochs)
    val_accuracies, test_accuracies = extract_metrics(metrics_file, epochs)
    
    if not val_accuracies or not neighborhood_quality:
        print("Error: No valid accuracy data extracted.")
        return
    
    traditional_epochs, nq_epochs = extract_epochs_required(val_accuracies, neighborhood_quality)
    traditional_time, nq_time = compute_training_time(embedding_dim, traditional_epochs, nq_epochs)
    
    traditional_test_acc = test_accuracies[traditional_epochs - 1] if traditional_epochs - 1 < len(test_accuracies) else "N/A"
    nq_test_acc = test_accuracies[nq_epochs - 1] if nq_epochs - 1 < len(test_accuracies) else "N/A"
    
    print("\nComparison Table:")
    print("| Model | Method | Epochs Required | Training Time | Test Accuracy |")
    print("|-------|--------|----------------|--------------|--------------|")
    print(f"| {model_used} | Traditional | {traditional_epochs} | {traditional_time} | {traditional_test_acc} |")
    print(f"| {model_used} | Neighborhood Quality | {nq_epochs} | {nq_time} | {nq_test_acc} |")


# Example usage
base_dir = "results - nli/"
metrics_file = "/home/gpuuser1/gpuuser1_a/sandeep/sandeep/nli/FFNN/logs/metrics.txt"
embedding_dim = 30
model_used = "FFNN"
generate_table(base_dir, metrics_file, embedding_dim, model_used)
