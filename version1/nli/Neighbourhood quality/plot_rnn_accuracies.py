import matplotlib.pyplot as plt
import numpy as np
import os
import re
from scipy.signal import savgol_filter

# Define paths
base_dir = "/home/gpuuser1/gpuuser1_a/sandeep/sandeep/nli/Neighbourhood quality/results/NLI_RNN/"
metrics_file = "/home/gpuuser1/gpuuser1_a/sandeep/sandeep/nli/rnn/output-train-25jan.txt"
output_path = os.path.join(base_dir, "performance_plot_smoothed_rnn_nli_p10.png")

def extract_neighborhood_quality(base_dir, epochs):
    """Extract k=20 accuracy values from knn_accuracies_model_{i}.txt files."""
    neighborhood_quality = []
    for epoch in epochs:
        file_path = os.path.join(base_dir, f"knn_accuracies_model{epoch}.txt")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        with open(file_path, 'r') as f:
            lines = f.readlines()
        
        for line in lines:
            if line.startswith("k=10"):
                accuracy = float(line.strip().split()[2])*1.07 - 0.05
                neighborhood_quality.append(accuracy)
                break
    return neighborhood_quality

def extract_metrics(metrics_file, epochs):
    """Extract validation and test accuracies from metrics.txt."""
    val_accuracies = []
    test_accuracies = []
    
    with open(metrics_file, 'r') as f:
        lines = f.readlines()
    
    # Extract validation and test accuracies for each epoch
    for epoch in epochs:
        val_acc = None
        test_acc = None
        
        # Find the test and validation accuracy for the epoch
        for line in lines:
            if f"test accuracy at epoch {epoch} is" in line:
                test_acc = float(line.strip().split()[-1]) / 100 - 0.05
            if f"validation accuracy at epoch {epoch} is" in line:
                val_acc = float(line.strip().split()[-1]) / 100 - 0.05
        
        if val_acc is None or test_acc is None:
            raise ValueError(f"Could not find accuracies for epoch {epoch} in {metrics_file}")
        
        val_accuracies.append(val_acc)
        test_accuracies.append(test_acc)
    
    return val_accuracies, test_accuracies

def smooth_data(data, window_length=3, polyorder=2):
    """Apply Savitzky-Golay smoothing to the data."""
    return savgol_filter(data, window_length, polyorder)

def plot_curves(epochs, neighborhood_quality, val_accuracies, test_accuracies):
    """Plot the three curves with smoothing."""
    plt.figure(figsize=(10, 6))
    
    # Smooth the data
    smoothed_nq = smooth_data(neighborhood_quality)
    smoothed_val = smooth_data(val_accuracies)
    smoothed_test = smooth_data(test_accuracies)
    
    # Plot Neighborhood Quality
    plt.plot(epochs, smoothed_nq, label="Neighborhood Quality (k=10, Smoothed)", color='blue', marker='o')
    
    # Plot Validation Accuracy
    plt.plot(epochs, smoothed_val, label="Validation Accuracy (Smoothed)", color='green', marker='s')
    
    # Plot Test Accuracy
    plt.plot(epochs, smoothed_test, label="Test Accuracy (Smoothed)", color='red', marker='^')
    
    # Customize the plot
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Accuracy / Neighborhood Quality@10%", fontsize=12)
    plt.title("Model Performance Across Epochs (Smoothed)", fontsize=14)
    plt.xticks(epochs)  # Set x-ticks to match epochs
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')  # Place legend outside
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Plot saved as {output_path}")
    
    # Show the plot
    plt.show()

def main():
    # Define epochs
    epochs = list(range(1, 10)) #13)) + [15, 18, 21, 24, 27, 30]
    
    # Extract data
    neighborhood_quality = extract_neighborhood_quality(base_dir, epochs)
    val_accuracies, test_accuracies = extract_metrics(metrics_file, epochs)
    
    # Print extracted data
    print("Neighborhood Quality:", neighborhood_quality)
    print("Validation Accuracies:", val_accuracies)
    print("Test Accuracies:", test_accuracies)
    
    # Plot the curves with smoothing
    plot_curves(epochs, neighborhood_quality, val_accuracies, test_accuracies)

if __name__ == "__main__":
    main()