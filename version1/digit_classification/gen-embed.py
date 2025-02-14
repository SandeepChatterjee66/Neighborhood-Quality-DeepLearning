import torch
import torch.nn as nn
import os
import time
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets

# Define constants
BATCH_SIZE = 128
INPUT_SIZE = 28 * 28  # Flattened MNIST images
HIDDEN_SIZES = [512, 256, 128, 64]
OUTPUT_SIZE = 10  # Digits 0-9
NUM_EPOCHS = 10

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load MNIST dataset (reuse training data)
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
train_dataset = datasets.MNIST(root="./data", train=True, transform=transform, download=True)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Define function to save embeddings
def save_embeddings_to_text_file(embeddings, output_file_path):
    with open(output_file_path, 'a') as file:
        for embedding in embeddings:
            file.write(" ".join(map(str, embedding.tolist())) + "\n")

# Define FFNN model with embeddings extraction
class FFNNWithEmbeddings(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super(FFNNWithEmbeddings, self).__init__()
        layers = []
        prev_size = input_size
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))
            prev_size = hidden_size
        layers.append(nn.Linear(prev_size, output_size))
        self.model = nn.Sequential(*layers)
    
    def forward(self, x, embed_path=""):
        x = x.view(-1, INPUT_SIZE)  # Flatten images
        embeddings = None
        for i, layer in enumerate(self.model):
            x = layer(x)
            if i == len(self.model) - 4:  # Extract embeddings from the second last layer
                embeddings = x
        if embed_path:
            save_embeddings_to_text_file(embeddings.cpu().detach().numpy(), embed_path)
        return x, embeddings

# Function to generate embeddings
def gen_embeddings(loader, model, embed_path):
    model.eval()
    all_labels = []
    all_predictions = []
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs, embeddings = model(images, embed_path)
            _, predicted = torch.max(outputs, 1)
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())
    return all_labels, all_predictions

# Load model and generate embeddings
model = FFNNWithEmbeddings(INPUT_SIZE, HIDDEN_SIZES, OUTPUT_SIZE).to(device)

for epoch in range(1, NUM_EPOCHS + 1):
    start = time.time()
    print(f"\nGenerating embeddings for epoch {epoch}")
    
    model_path = f"mnist_ffnn_epoch_{epoch}.pth"
    model.load_state_dict(torch.load(model_path, map_location=device))
    
    embeddings_output_path = f"embeddings/train/ffnn_model_{epoch}_embeddings.txt"
    os.makedirs(os.path.dirname(embeddings_output_path), exist_ok=True)
    
    labels, predictions = gen_embeddings(train_loader, model, embeddings_output_path)
    
    # Save labels and predictions
    with open(f"embeddings/train/ffnn_model_{epoch}_labels.txt", 'w') as file:
        file.write("\n".join(map(str, labels)) + "\n")
    
    with open(f"embeddings/train/ffnn_model_{epoch}_predictions.txt", 'w') as file:
        file.write("\n".join(map(str, predictions)) + "\n")
    
    print(f"Total time taken: {time.time() - start:.2f} seconds")

print("Embedding generation complete!")
