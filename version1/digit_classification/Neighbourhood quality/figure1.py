import matplotlib.pyplot as plt
import numpy as np
import glob
import os

# Define the base directory and file pattern
base_dir = "results/NLI_RNN"
file_pattern = os.path.join(base_dir, "knn_accuracies_model*.txt")  # Removed underscore after "model"
# Save the plot
output_path = os.path.join(base_dir, "figure1-rnn_nli.png")

# Get all matching files
files = sorted(glob.glob(file_pattern), key=lambda x: int(x.split("model")[-1].split(".")[0]))
print("files", files)  # Debugging: Print the list of files

# Define colors for the curves
colors = plt.cm.viridis(np.linspace(0, 1, len(files)))

# Create a plot
plt.figure(figsize=(10, 6))

# Initialize k_values (common for all files)
k_values = None

# Read and plot data from each file
for i, file in enumerate(files):
    # Extract epoch number from the filename
    epoch = int(file.split("model")[-1].split(".")[0])  # Updated to match filename format
    
    # Read data from the file
    with open(file, 'r') as f:
        lines = f.readlines()
    
    # Parse data
    k_values = []
    accuracies = []
    for line in lines:
        parts = line.strip().split()
        k_values.append(int(parts[0].split('=')[1]))  # Extract K value
        accuracies.append(float(parts[2]))  # Extract accuracy (neighborhood quality)
    
    # Convert to numpy arrays
    k_values = np.array(k_values)
    accuracies = np.array(accuracies)
    
    # Plot the curve
    plt.plot(k_values, accuracies, label=f"Epoch {epoch}", color=colors[i], marker='o')

# Customize the plot
plt.xlabel("K (Number of Neighbors)", fontsize=12)
plt.ylabel("Neighborhood Quality", fontsize=12)
plt.title("Neighborhood Quality vs K for Different Epochs", fontsize=14)
plt.xticks(k_values)  # Set x-ticks to match K values
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(title="Epoch", bbox_to_anchor=(1.05, 1), loc='upper left')  # Place legend outside
plt.tight_layout()
plt.savefig(output_path, dpi=300, bbox_inches="tight")
print(f"Plot saved as {output_path}")

# Show the plot (optional)
plt.show()