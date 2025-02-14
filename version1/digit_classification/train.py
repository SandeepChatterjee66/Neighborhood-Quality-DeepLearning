import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split

# Define constants
BATCH_SIZE = 128
INPUT_SIZE = 28 * 28  # Flattened MNIST images
HIDDEN_SIZES = [512, 256, 128, 64]
OUTPUT_SIZE = 10  # Digits 0-9
EPOCHS = 10
LEARNING_RATE = 0.001

# Load the MNIST dataset
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

train_dataset = torchvision.datasets.MNIST(root="./data", train=True, transform=transform, download=True)
test_dataset = torchvision.datasets.MNIST(root="./data", train=False, transform=transform, download=True)

# Split training dataset into train and validation
train_size = int(0.8 * len(train_dataset))
val_size = len(train_dataset) - train_size
train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Define Feedforward Neural Network model
class FFNN(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super(FFNN, self).__init__()
        layers = []
        prev_size = input_size
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))
            prev_size = hidden_size
        layers.append(nn.Linear(prev_size, output_size))
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        x = x.view(-1, INPUT_SIZE)  # Flatten images
        return self.model(x)

# Instantiate model, loss function, and optimizer
model = FFNN(INPUT_SIZE, HIDDEN_SIZES, OUTPUT_SIZE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

def evaluate_model(loader, dataset_type="Validation"):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in loader:
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    return accuracy

def train_model():
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        # Evaluate after each epoch
        val_accuracy = evaluate_model(val_loader, "Validation")
        test_accuracy = evaluate_model(test_loader, "Test")
        
        print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {running_loss/len(train_loader):.4f}, "
              f"Validation Accuracy: {val_accuracy:.2f}%, Test Accuracy: {test_accuracy:.2f}%")
        
        # Save model after each epoch
        torch.save(model.state_dict(), f"mnist_ffnn_epoch_{epoch+1}.pth")

# Train the model
train_model()
