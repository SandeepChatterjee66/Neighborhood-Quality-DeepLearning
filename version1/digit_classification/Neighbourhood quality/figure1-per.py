import matplotlib.pyplot as plt
import numpy as np
import glob
import os

model_used = "FFNN"
# Define the base directory and file pattern
base_dir = "results/" + model_used
output_path = os.path.join(base_dir, f"figure1-{model_used}.png")

file_pattern = os.path.join(base_dir, "knn_accuracies_model*.txt")  # Removed underscore after "model"
# Get all matching files
files = sorted(glob.glob(file_pattern), key=lambda x: int(x.split("model")[-1].split(".")[0]))
print("files", files)  # Debugging: Print the list of files

# Define colors for the curves
colors = plt.cm.viridis(np.linspace(0, 1, len(files)))

# Create a plot
plt.figure(figsize=(10, 6))

# Read and plot data from each file
for i, file in enumerate(files):
    # Extract epoch number from the filename
    epoch = int(file.split("model")[-1].split(".")[0])  # Updated to match filename format
    
    # Read data from the file
    with open(file, 'r') as f:
        lines = f.readlines()
    
    # Parse data, but only for Percentage: 100.0%
    k_values = []
    accuracies = []
    record_data = False  # Flag to start recording once we hit Percentage: 100.0%

    for line in lines:
        if line.startswith("Percentage:"):
            percentage = float(line.strip().split(":")[1].strip().replace("%", ""))
            record_data = (percentage == 100.0)  # Only set to True for 100%

        if record_data and "k=" in line:
            try:
                parts = line.strip().split()  # Split by whitespace
                k_value = int(parts[0].split('=')[1])  # Extract K value
                accuracy = float(parts[2])  # Extract accuracy (neighborhood quality)
                k_values.append(k_value)
                accuracies.append(accuracy)
            except (IndexError, ValueError):
                print(f"Skipping malformed line in {file}: {line.strip()}")
    
    if not k_values:
        print(f"Warning: No valid data found in {file} for Percentage=100%, skipping this file.")
        continue

    # Convert to numpy arrays
    k_values = np.array(k_values)
    accuracies = np.array(accuracies)

    # Plot the curve
    plt.plot(k_values, accuracies, label=f"Epoch {epoch}", color=colors[i], marker='o')

# Customize the plot
plt.xlabel("K (Number of Neighbors)", fontsize=12)
plt.ylabel("Neighborhood Quality", fontsize=12)
plt.title(f"{model_used} Neigh. Quality vs K for Percentage=100%", fontsize=14)
plt.xticks(k_values if len(k_values) > 0 else [])  # Set x-ticks to match K values
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(title="Epoch", bbox_to_anchor=(1.05, 1), loc='upper left')  # Place legend outside
plt.tight_layout()
plt.savefig(output_path, dpi=300, bbox_inches="tight")
print(f"Plot saved as {output_path}")

# Show the plot (optional)
plt.show()
