import numpy as np
import pandas as pd
import json
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# Load configuration
with open('config.json', 'r') as f:
    config = json.load(f)
print("Loaded config.json")

# Load the dataset
train_file_path = config["train_set_path"]
dataframe = pd.read_csv(train_file_path)

sentence1_list = dataframe["question1"].tolist()
sentence2_list = dataframe["question2"].tolist()
target_label_list = dataframe["is_duplicate"].tolist()

print(f"Dataset loaded. Total samples: {len(sentence1_list)}")

# Ensure labels are integers (if they are stored as strings)
target_label_list = [int(label) for label in target_label_list]

# Split dataset (same as embedding generation)
sentence1_train, sent1_test, sentence2_train, sent2_test, target_train, test_label = train_test_split(
    sentence1_list, sentence2_list, target_label_list, test_size=0.1, random_state=42
)

sent1_data, sent1_val, sent2_data, sent2_val, data_label, val_label = train_test_split(
    sentence1_train, sentence2_train, target_train, test_size=0.1 / 0.9, random_state=42
)

print(f"Training set: {len(sent1_data)} samples")
print(f"Validation set: {len(sent1_val)} samples")
print(f"Test set: {len(sent1_test)} samples")

# Encode labels as integers
label_encoder = LabelEncoder()
train_labels = label_encoder.fit_transform(data_label)

# Save labels to a NumPy file for KNN
labels_output_path = "labels.npy"
np.save(labels_output_path, train_labels)

print(f"Ground truth labels saved to {labels_output_path} with shape {train_labels.shape}")
