# embedgen ffnn ndd

import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import time
import os, json
import numpy as np
import time  # not required neccassarily
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from collections import Counter
import pickle as pkl
import random
import pdb
import io
import nltk
import os
from nltk.tokenize import sent_tokenize, word_tokenize
import pandas as pd
import pdb
import matplotlib
import matplotlib.pyplot as plt
import json
from sklearn.metrics import classification_report
from tqdm import tqdm
import numpy as np
from sklearn.decomposition import PCA


test_groundtruth_file_path = './FFNN_NDD/qqp_groundtruth.txt'
best_model_path = "./FFNN_NDD/BestModel/ffnn_model.pth"
pred_test_file_path = './FFNN_NDD/qqp_test_preds.txt'
labels_test_file_path = './FFNN_NDD/qqp_test_labels.txt'

#batch size 128
BATCH_SIZE = 256
NUM_EPOCHS = 15
EMBED_SIZE = 30

with open('config.json', 'r') as f:
    config = json.load(f)
print("config.json loaded")
    
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('device:',device)

FastText = []
with open(config['fasttext_embeddings_path'], "r", encoding="utf-8") as ft:
    for i, line in enumerate(ft):
        if i == 0:
            continue
        FastText.append(line)
print("fastext lines",len(FastText))
        





PICKLE_FILE = f"pca_embeddings{EMBED_SIZE}.pkl"

def build_embedding(data):
    word2id = {"<pad>": 0, "<unk>": 1}
    id2word = {0: "<pad>", 1: "<unk>"}
    
    embeddings = [np.zeros(300), np.random.normal(0, 0.01, 300)]
    pca = PCA(n_components=EMBED_SIZE)
    
    # Check if the pickle file already exists
    if os.path.exists(PICKLE_FILE):
        print(f"Loading PCA transformed embeddings from {PICKLE_FILE}")
        with open(PICKLE_FILE, 'rb') as f:
            word2id, id2word, embeddings = pkl.load(f)
        return word2id, id2word, embeddings
    
    # If pickle doesn't exist, process data and save the PCA-transformed embeddings
    print("Building embeddings...")
    for i, line in tqdm(enumerate(data), total=len(data), desc="Building embeddings"):
        parsed = line.split()
        word = parsed[0]
        array = np.array([float(x) for x in parsed[1:]])

        word2id[word] = i + 2
        id2word[i + 2] = word
        embeddings.append(array)
    
    print("Applying PCA transformation...")
    pca.fit(np.array(embeddings))
    
    # Apply PCA transformation to all embeddings
    transformed_embeddings = []
    for i in tqdm(range(len(embeddings)), desc="PCA Transformation", position=1):
        transformed_embeddings.append(pca.transform(embeddings[i].reshape(1, -1)).flatten())
    
    # Save only the PCA-transformed embeddings and the word2id, id2word to a pickle file
    with open(PICKLE_FILE, 'wb') as f:
        print(f"Saving PCA-transformed embeddings to {PICKLE_FILE}")
        pkl.dump((word2id, id2word, transformed_embeddings), f)
    
    return word2id, id2word, transformed_embeddings






#token2id, id2token, word_vectors = build_embedding(FastText)


PAD_IDX = 0
UNK_IDX = 1

def convert_labels_to_integers(data_label):
    for i in range(len(data_label)):
        if data_label[i] == 0:
            data_label[i] = 0
        elif data_label[i] == 1:
            data_label[i] = 1
    return data_label


    
# Word tokenize each entry in a list of sentences
import collections
def tokenize(sentence_list):
    d = collections.defaultdict(int)
    for i in range(len(sentence_list)):
        d[type(sentence_list[i])]+=1
    print(d)
    filtered_sentences = [sent for sent in sentence_list if isinstance(sent, (float, int))]
    print(filtered_sentences)
    return [word_tokenize(str(sentence_list[i])) for i in range(len(sentence_list))]

# "one-hot encode": convert each token to id in vocabulary vector (token2id)
def token2index_dataset(tokens_data):
    indices_data = []
    for tokens in tokens_data:
        index_list = [token2id[token] if token in token2id else UNK_IDX for token in tokens]
        indices_data.append(index_list)
    return indices_data 

print("starting to build embeddings")
token2id, id2token, word_vectors = build_embedding(FastText)
PAD_IDX = 0
UNK_IDX = 1
print("built the embeddings", len(token2id), len(id2token), len(word_vectors))

_weights = np.array(word_vectors)
_WEIGHTS = _weights

print("weights array", _WEIGHTS.shape)


def data_pipeline(sent1s, sent2s, labels, verify=True):
    labels = convert_labels_to_integers(labels)
          
    print("\nTokenizing sentence 1 list...")    
    sent1s_tokenized = tokenize(sent1s)
    print("done!")
    print("\nTokenizing sentence 2 list... ")  
    sent2s_tokenized = tokenize(sent2s)
    print("done!")
    
    print("\nOne-hot encoding words for sentence 1 list...")  
    sent1s_indices = token2index_dataset(sent1s_tokenized)
    print("done!")
    print("\nOne-hot encoding words for sentence 2 list...")  
    sent2s_indices = token2index_dataset(sent2s_tokenized)
    print("done!")
    
    return (sent1s_indices, sent2s_indices, labels)


import numpy as np
import torch
from torch.utils.data import Dataset

class TwoSentencesDataset(Dataset):
    """
    Class that represents a train/validation/test dataset that's readable for PyTorch
    """
    
    def __init__(self, sent1_data_list, sent2_data_list, target_list):
        """
        @param sent1_data_list: list of sentence1's (index matches sentence2's and target_list below)
        @param sent2_data_list: list of sentence2's
        @param target_list: list of correct labels

        """
        self.sent1_data_list = sent1_data_list
        self.sent2_data_list = sent2_data_list
        self.target_list = target_list
        assert (len(self.sent1_data_list) == len(self.target_list) and len(self.sent2_data_list) == len(self.target_list))

    def __len__(self):
        return len(self.sent1_data_list)
        
    def __getitem__(self, key):
        ###
        ### Returns [[sentence, 1, tokens], [sentence, 2, tokens]]
        ###
        """
        Triggered when you call dataset[i]
        """
        sent1_tokens_idx = self.sent1_data_list[key][:MAX_SENTENCE_LENGTH]
        sent2_tokens_idx = self.sent2_data_list[key][:MAX_SENTENCE_LENGTH]
        combined_tokens_idx = [sent1_tokens_idx, sent2_tokens_idx]
        label = self.target_list[key]
        return [combined_tokens_idx, len(sent1_tokens_idx), len(sent2_tokens_idx), label]

def twosentences_collate_func(batch):
    """
    Customized function for DataLoader that dynamically pads the batch so that all 
    data have the same length
    """
    sent1_data_list = []
    sent2_data_list = []
    sent1_length_list = []
    sent2_length_list = []
    label_list = []
    combined_data_list = []
    #print("collate batch: ", batch[0][0])
    #batch[0][0] = batch[0][0][:MAX_SENTENCE_LENGTH]
    for datum in batch:
        label_list.append(datum[3])
        sent1_length_list.append(datum[1])
        sent2_length_list.append(datum[2])
    # padding
    for datum in batch:
        padded_vec_1 = np.pad(np.array(datum[0][0]), pad_width=((0,MAX_SENTENCE_LENGTH-datum[1])), 
                                mode="constant", constant_values=0)
        padded_vec_2 = np.pad(np.array(datum[0][1]), pad_width=((0,MAX_SENTENCE_LENGTH-datum[2])), 
                                mode="constant", constant_values=0)
        combined_data_list.append([padded_vec_1, padded_vec_2])
    return [torch.from_numpy(np.array(combined_data_list)), 
            torch.LongTensor(sent1_length_list), torch.LongTensor(sent2_length_list), torch.LongTensor(label_list)]    

import pandas as pd
from sklearn.model_selection import train_test_split


# Load the data
train_file_path = config["train_set_path"]
dataframe = pd.read_csv(train_file_path)
sentence1_list = dataframe['question1'].tolist()
sentence2_list = dataframe['question2'].tolist()
target_label_list = dataframe['is_duplicate'].tolist()

print("initial dataset size", len(sentence1_list))

# First split: train+val and test (10% of the original data for test set)
sentence1_train, sent1_test, sentence2_train, sent2_test, target_train, test_label = train_test_split(
    sentence1_list, sentence2_list, target_label_list, test_size=0.1, random_state=42)

# Second split: train and validation (10% of the original data for validation set, which is 1/9th of the remaining 90%)
sent1_data, sent1_val, sent2_data, sent2_val, data_label, val_label = train_test_split(
    sentence1_train, sentence2_train, target_train, test_size=0.1/0.9, random_state=42)

# Output the sizes of each set
print(f"Training set size: {len(sent1_data)}")
print(f"Validation set size: {len(sent1_val)}")
print(f"Test set size: {len(sent1_test)}")
print(type(val_label[40]))
print(sent2_val[40])
print(sent1_val[40])

# Write the list to the file
with open(test_groundtruth_file_path, 'w+') as file:
    for item in test_label:
        file.write(f"{item}\n")
print(f"List written to {test_groundtruth_file_path}")

sent1_train_indices, sent2_train_indices, train_label = data_pipeline(sent1_data, sent2_data, data_label)
train_dataset = TwoSentencesDataset(sent1_train_indices, sent2_train_indices, train_label)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                           batch_size=BATCH_SIZE, 
                                           collate_fn=twosentences_collate_func,
                                           #shuffle=True
                                          )
print("Finished creating train_loader.")

MAX_SENTENCE_LENGTH = max(max([len(sent) for sent in sent1_train_indices]), max([len(sent) for sent in sent2_train_indices]))
MAX_SENTENCE_LENGTH

sent1_val_indices, sent2_val_indices, val_label = data_pipeline(sent1_val, sent2_val, val_label)
val_dataset = TwoSentencesDataset(sent1_val_indices, sent2_val_indices, val_label)
val_loader = torch.utils.data.DataLoader(dataset=val_dataset, 
                                           batch_size=BATCH_SIZE, 
                                           collate_fn=twosentences_collate_func,
                                           #shuffle=True
                                          )



sent1_test_indices, sent2_test_indices, test_label = data_pipeline(sent1_test, sent2_test, test_label)
test_dataset = TwoSentencesDataset(sent1_test_indices, sent2_test_indices, test_label)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, 
                                           batch_size=BATCH_SIZE, 
                                           collate_fn=twosentences_collate_func,
                                           #shuffle=True
                                          )

def save_embeddings_to_text_file(embeddings, output_file_path):
    with open(output_file_path, 'a') as file:
        for embedding in embeddings:
            for i,emb in enumerate(embedding):
                if i != len(embedding) - 1:
                    file.write(f'{emb} ')
                else:
                    file.write(f'{emb}\n')

class TwoSentenceModel(nn.Module):
    def __init__(self, hidden_size, num_classes, emb_size, max_sent_length=MAX_SENTENCE_LENGTH, weights=_WEIGHTS, embed_path=""):
        super(TwoSentenceModel, self).__init__()

        self.hidden_size = hidden_size
        self.embed_path = embed_path
        
        # Use pretrained weights if available
        weight = torch.FloatTensor(weights) if weights is not None else None
        self.embedding = nn.Embedding.from_pretrained(weight) if weight is not None else nn.Embedding(emb_size, emb_size)
        
        # Correct input_size calculation
        input_size = 2 * emb_size * max_sent_length
        
        # Defining the 5 hidden layers
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, hidden_size)
        self.linear4 = nn.Linear(hidden_size, hidden_size)
        self.linear5 = nn.Linear(hidden_size, hidden_size)
        
        # Output layer
        self.linear6 = nn.Linear(hidden_size, num_classes)

        # Dropout layer
        self.dropout = nn.Dropout(0.2)

    def forward(self, x, sent1_lengths, sent2_lengths):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        x = x.to(torch.long).to(device)
        batch_size = x.size()[0]
        
        # Splitting input into sentences
        sent1s = x[:, 0, :]
        sent2s = x[:, 1, :]
        sents = torch.cat([sent1s, sent2s], dim=1).to(device)
        
        # Embedding lookup
        embed = self.embedding(sents)
    
        # Print the shape of embed before reshaping
        #print(f"Embed shape before flattening: {embed.shape}")
        
        embed = embed.view(embed.size(0), -1)  # Flatten the embedding

        # Print the shape after flattening
        #print(f"Embed shape after flattening: {embed.shape}")

        # First hidden layer with ReLU activation
        linear1 = self.linear1(embed)
        linear1 = torch.relu(linear1)  # Using ReLU instead of Tanh
        linear1 = self.dropout(linear1)

        # Second hidden layer with ReLU activation
        linear2 = self.linear2(linear1)
        linear2 = torch.relu(linear2)  # Using ReLU
        linear2 = self.dropout(linear2)

        # Third hidden layer with ReLU activation
        linear3 = self.linear3(linear2)
        linear3 = torch.relu(linear3)  # Using ReLU
        linear3 = self.dropout(linear3)

        # Fourth hidden layer with ReLU activation
        linear4 = self.linear4(linear3)
        linear4 = torch.relu(linear4)  # Using ReLU
        linear4 = self.dropout(linear4)

        # Fifth hidden layer with ReLU activation
        linear5 = self.linear5(linear4)
        linear5 = torch.relu(linear5)  # Using ReLU
        linear5 = self.dropout(linear5)

        if len(self.embed_path) > 0:
            save_embeddings_to_text_file(linear5, self.embed_path)


        # Output layer (logits)
        logits = self.linear6(linear5)

        return logits



# class TwoSentenceModel(nn.Module):
    
#     def __init__(self, hidden_size, num_classes, emb_size = 20):
#         super(TwoSentenceModel, self).__init__()

#         self.hidden_size = hidden_size
#         weight = torch.FloatTensor(_WEIGHTS)
#         self.embedding = nn.Embedding.from_pretrained(weight)
#         self.linear1 = nn.Linear(2*emb_size*MAX_SENTENCE_LENGTH, 100)
#         self.linear2 = nn.Linear(100, num_classes)
#         self.dropout = nn.Dropout(0.2)
#     def forward(self, x, sent1_lengths, sent2_lengths):
#         batch_size = x.size()[0]

#         sent1s = x[:, 0, :]
#         sent2s = x[:, 1, :]
#         # print(sent1s.size())
#         sents = torch.cat([sent1s, sent2s], dim=1).to(device)
#         # print(sents.size())
        
#         # get embedding
#         embed = self.embedding(sents)
#         # print(embed.size())
#         embed = embed.view(embed.size(0), -1)
        
#         linear1 = self.linear1(embed)
#         linear1 = torch.tanh(linear1.contiguous().view(-1, linear1.size(-1))).view(linear1.shape)   

#         save_embeddings_to_text_file(linear1, embeddings_output_path)
#         #need to change the file name to create embeddings after every epoch
        
#         #linear1 = self.dropout(linear1)
#         logits = self.linear2(linear1)
#         # print(logits.size())
#         return logits





# class TwoSentenceModel(nn.Module):
    
#     def __init__(self, hidden_size, num_classes, emb_size = EMBED_SIZE):
#         super(TwoSentenceModel, self).__init__()

#         self.hidden_size = hidden_size
#         weight = torch.FloatTensor(_WEIGHTS)
#         self.embedding = nn.Embedding.from_pretrained(weight)
        
#         # First hidden layer with ReLU activation
#         self.linear1 = nn.Linear(2*emb_size*MAX_SENTENCE_LENGTH, 100)
        
#         # Second hidden layer with ReLU activation
#         self.linear2 = nn.Linear(100, 100)
        
#         # Third hidden layer with ReLU activation
#         self.linear3 = nn.Linear(100, 100)
        
#         # Fourth hidden layer with ReLU activation
#         self.linear4 = nn.Linear(100, 100)
        
#         # Output layer
#         self.linear5 = nn.Linear(100, num_classes)
        
#         # Dropout for regularization
#         self.dropout = nn.Dropout(0.2)

#     def forward(self, x, sent1_lengths, sent2_lengths):
#         x = x.to(torch.long) 
#         batch_size = x.size()[0]
#         sent1s = x[:, 0, :]
#         sent2s = x[:, 1, :]
#         sents = torch.cat([sent1s, sent2s], dim=1).to(device)
        
#         # Get embedding
#         embed = self.embedding(sents)
#         embed = embed.view(embed.size(0), -1)
        
#         # First hidden layer with ReLU activation
#         linear1 = self.linear1(embed)
#         linear1 = F.relu(linear1)  # Using ReLU activation
#         linear1 = self.dropout(linear1)
        
#         # Second hidden layer with ReLU activation
#         linear2 = self.linear2(linear1)
#         linear2 = F.relu(linear2)  # Using ReLU activation
#         linear2 = self.dropout(linear2)
        
#         # Third hidden layer with ReLU activation
#         linear3 = self.linear3(linear2)
#         linear3 = F.relu(linear3)  # Using ReLU activation
#         linear3 = self.dropout(linear3)
        
#         # Fourth hidden layer with ReLU activation
#         linear4 = self.linear4(linear3)
#         linear4 = F.relu(linear4)  # Using ReLU activation
#         linear4 = self.dropout(linear4)
        
#         save_embeddings_to_text_file(linear4, embeddings_output_path)
#         #need to change the file name to create embeddings after every epoch
        
#         # Output layer
#         logits = self.linear5(linear4)
        
#         # Softmax activation for final output
#         output = F.softmax(logits, dim=-1)
        
#         return output

def gen_embed(loader, model):
    """
    Helper function that tests the model's performance on a dataset
    """
    model.eval()
    for (data, sent1_lengths, sent2_lengths, labels) in loader:
        data_batch, sent1_length_batch, sent2_length_batch, label_batch = data.to(device), sent1_lengths.to(device), sent2_lengths.to(device), labels.to(device)
        model(data_batch, sent1_length_batch, sent2_length_batch)

for epoch in range(1,1+NUM_EPOCHS):
    start = time.time()
    print("\nTrain Embeddings",epoch)
    #change the model path to create embeddings after every epoch
    embeddings_output_path = f'Embeddings/Train/ffnn_model_{epoch}_ndd.txt'
    emb_directory = os.path.dirname(embeddings_output_path)
    os.makedirs(emb_directory, exist_ok=True)
    
    model = TwoSentenceModel(emb_size = EMBED_SIZE, hidden_size=100, num_classes=2, embed_path=embeddings_output_path).to(device)
    model_path = f"./FFNN_NDD/ffnn_model_{epoch}.pth"
    model.load_state_dict(torch.load(model_path))
    gen_embed(train_loader, model)
    
    # print("\nTest Embeddings")
    # embeddings_output_path = f"Embeddings/Test/ffnn_model_{epoch}_ndd.txt"
    # gen_embed(test_loader, model)
    
    end = time.time()
    total_time = end - start
    print("total time taken is ",total_time) 


model_test = TwoSentenceModel(emb_size = EMBED_SIZE, hidden_size=300, num_layers=1, num_classes=2).to(device)
model_test.load_state_dict(torch.load(best_model_path))


import torch.nn.functional as F

# Function for testing the model
def test_model(loader, model):
    """
    Helper function that tests the model's performance on a dataset
    """
    correct = 0
    total = 0
    all_labels = []
    all_predictions = []
    model.eval()
    with torch.no_grad():  # Disable gradient calculation for testing
        for (data, sent1_lengths, sent2_lengths, labels) in loader:
            data_batch, sent1_length_batch, sent2_length_batch, label_batch = data.to(device), sent1_lengths.to(device), sent2_lengths.to(device), labels.to(device)
            outputs = F.softmax(model(data_batch, sent1_length_batch, sent2_length_batch), dim=1)
            predicted = outputs.max(1, keepdim=True)[1]
            
            all_labels.extend(label_batch.cpu().numpy())  # Save labels
            all_predictions.extend(predicted.cpu().numpy())  # Save predictions
            
            total += label_batch.size(0)
            correct += predicted.eq(label_batch.view_as(predicted)).sum().item()
    
    accuracy = 100 * correct / total
    return accuracy, all_labels, all_predictions

# Test the model
accuracy, labels, predictions = test_model(test_loader, model_test)

print(f"Accuracy: {accuracy:.2f}%")
print(f"Number of labels: {len(labels)}")
print(f"Labels: {labels}")
print(f"Predictions: {predictions}")

# Write labels to a file
with open(labels_test_file_path, 'w') as file:
    for label in labels:
        file.write(f"{label}\n")

print(f"Labels written to {labels_test_file_path}")

# Write predictions to a file
with open(pred_test_file_path, 'w') as file:
    for prediction in predictions:
        file.write(f"{prediction}\n")

print(f"Predictions written to {pred_test_file_path}")