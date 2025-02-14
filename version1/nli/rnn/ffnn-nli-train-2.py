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
from nltk.tokenize import sent_tokenize, word_tokenize
import pandas as pd
import pdb
import matplotlib
import matplotlib.pyplot as plt
import json
from sklearn.metrics import classification_report



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
        


from tqdm import tqdm
import numpy as np
from sklearn.decomposition import PCA

import os
import pickle as pkl
import numpy as np
from sklearn.decomposition import PCA
from tqdm import tqdm


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

BATCH_SIZE = 64
PAD_IDX = 0
UNK_IDX = 1

def convert_labels_to_integers(data_label):
    for i in range(len(data_label)):
        if data_label[i] == "contradiction":
            data_label[i] = 0
        elif data_label[i] == "entailment":
            data_label[i] = 1
        elif data_label[i] == "neutral":
            data_label[i] = 2
    return data_label

def verify_order(sent1_data, sent2_data, data_label):
    i = random.randint(1, len(sent1_data))
    print(sent1_data[i])
    print(sent2_data[i])
    print(data_label[i])
    
# Word tokenize each entry in a list of sentences
def tokenize(sentence_list):
    return [word_tokenize(sentence_list[i]) for i in range(len(sentence_list))]

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
    
    print("\nVerifying that the data and label match after shuffling")
    print(sent1s[2])
    print(sent2s[2])
    if verify:
        verify_order(sent1s, sent2s, labels)
        verify_order(sent1s, sent2s, labels)
          
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
    
    
    
sent1_train = []
sent2_train = []
labels_train = []
cnt = 0
flag = 0
lengths = set()
with open(config['train_set_path'], 'r', encoding='utf-8') as train:
    for line in train:
        if flag == 0:
            flag = 1 
            continue
        parts = line.strip().split('\t')
        lengths.add(len(parts))
        label = parts[0]
        if len(parts) == 10:
            s1 = parts[-5]
            s2 = parts[-4]
        elif len(parts) == 12:
            s1 = parts[-7]
            s2 = parts[-6]
            
        elif len(parts) == 13:
            s1 = parts[-8]
            s2 = parts[-7]
            
        elif len(parts) == 14:
            s1 = parts[-9]
            s2 = parts[-8]
        
        if label == 'contradiction':
            labels_train.append(0)
            sent1_train.append(s1)
            sent2_train.append(s2)
        elif label == 'neutral':
            labels_train.append(1)
            sent1_train.append(s1)
            sent2_train.append(s2)
        elif label == 'entailment':
            labels_train.append(2)
            sent1_train.append(s1)
            sent2_train.append(s2)
            
        
    print("len of sen1 train sent2 train labels train",len(sent1_train), len(sent2_train), len(labels_train))
    print(lengths,cnt)

sent1_data = sent1_train
sent2_data = sent2_train
data_label = labels_train
print("Size of training data: {}".format(len(sent1_data)))

sent1_train_indices, sent2_train_indices, train_label = data_pipeline(sent1_data, sent2_data, data_label)
train_dataset = TwoSentencesDataset(sent1_train_indices, sent2_train_indices, train_label)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                           batch_size=BATCH_SIZE, 
                                           collate_fn=twosentences_collate_func,
                                           #shuffle=True
                                          )
print("Finished creating train_loader.")

MAX_SENTENCE_LENGTH = max(max([len(sent) for sent in sent1_train_indices]), max([len(sent) for sent in sent2_train_indices]))


sent1_val = []
sent2_val = []
val_label = []
lengths = set()
with open(config['validation_set_path'], 'r', encoding='utf-8') as val:
    for line in val:
        parts = line.strip().split('\t')
        label = parts[0]
        if len(parts) == 13:
            s1 = parts[-8]
            s2 = parts[-7]
        elif len(parts) == 14:
            s1 = parts[-9]
            s2 = parts[-8]
        lengths.add(len(parts))
        if label == 'contradiction':
            val_label.append(0)
            sent1_val.append(s1)
            sent2_val.append(s2)
        if label == 'neutral':
            val_label.append(1)
            sent1_val.append(s1)
            sent2_val.append(s2)
        if label == 'entailment':
            val_label.append(2)
            sent1_val.append(s1)
            sent2_val.append(s2)
            
    print(lengths)
print("Size of val data: {}".format(len(sent1_val)))

sent1_val_indices, sent2_val_indices, val_label = data_pipeline(sent1_val, sent2_val, val_label)
val_dataset = TwoSentencesDataset(sent1_val_indices, sent2_val_indices, val_label)
val_loader = torch.utils.data.DataLoader(dataset=val_dataset, 
                                           batch_size=BATCH_SIZE, 
                                           collate_fn=twosentences_collate_func,
                                           #shuffle=True
                                          )



sent1_test = []
sent2_test = []
test_label = []
lengths = set()
with open(config['test_set_path'], 'r', encoding='utf-8') as test:
    for line in test:
        parts = line.strip().split('\t')
        label = parts[0]
        if len(parts) == 13:
            s1 = parts[-8]
            s2 = parts[-7]
        elif len(parts) == 14:
            s1 = parts[-9]
            s2 = parts[-8]
        lengths.add(len(parts))
        if label == 'contradiction':
            test_label.append(0)
            sent1_test.append(s1)
            sent2_test.append(s2)
        if label == 'neutral':
            test_label.append(1)
            sent1_test.append(s1)
            sent2_test.append(s2)
        if label == 'entailment':
            test_label.append(2)
            sent1_test.append(s1)
            sent2_test.append(s2)
            
    print(lengths)

sent1_test_indices, sent2_test_indices, test_label = data_pipeline(sent1_test, sent2_test, test_label)
test_dataset = TwoSentencesDataset(sent1_test_indices, sent2_test_indices, test_label)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, 
                                           batch_size=BATCH_SIZE, 
                                           collate_fn=twosentences_collate_func,
                                           #shuffle=True
                                          )

class TwoSentenceModel(nn.Module):
    
    def __init__(self, hidden_size, num_classes, emb_size = EMBED_SIZE):
        super(TwoSentenceModel, self).__init__()

        self.hidden_size = hidden_size
        weight = torch.FloatTensor(_WEIGHTS)
        self.embedding = nn.Embedding.from_pretrained(weight)
        self.linear1 = nn.Linear(2*emb_size*MAX_SENTENCE_LENGTH, 100)
        self.linear2 = nn.Linear(100, 100)  # New hidden layer with 100 neurons
        self.linear3 = nn.Linear(100, num_classes)  # Output layer
        self.dropout = nn.Dropout(0.2)

    def forward(self, x, sent1_lengths, sent2_lengths):
        x = x.to(torch.long) 
        batch_size = x.size()[0]
        sent1s = x[:, 0, :]
        sent2s = x[:, 1, :]
        sents = torch.cat([sent1s, sent2s], dim=1).to(device)
        
        # Get embedding
        embed = self.embedding(sents)
        embed = embed.view(embed.size(0), -1)
        
        # First hidden layer
        linear1 = self.linear1(embed)
        linear1 = torch.tanh(linear1.contiguous().view(-1, linear1.size(-1))).view(linear1.shape)
        linear1 = self.dropout(linear1)
        
        # Second hidden layer (added)
        linear2 = self.linear2(linear1)
        linear2 = torch.tanh(linear2)
        linear2 = self.dropout(linear2)
        
        # Output layer
        logits = self.linear3(linear2)
        return logits

BATCH_SIZE = config['batch_size']

import torch
from sklearn.metrics import precision_recall_fscore_support

def compute_metrics(y_true, y_pred, average='macro'):
    """
    Compute precision, recall, F1-score, and support directly on GPU using PyTorch.
    This is equivalent to sklearn's classification_report.
    
    Parameters:
    - y_true: Tensor (with ground truth labels)
    - y_pred: Tensor (with predicted labels)
    - average: String (type of averaging. 'macro', 'micro', 'weighted')
    
    Returns:
    - A dictionary with precision, recall, F1-score, and support
    """
    # Ensure the inputs are tensors
    if not isinstance(y_true, torch.Tensor):
        y_true = torch.tensor(y_true)
    if not isinstance(y_pred, torch.Tensor):
        y_pred = torch.tensor(y_pred)
    
    # Compute confusion matrix elements
    tp = torch.sum((y_pred == 1) & (y_true == 1)).item()  # True Positives
    tn = torch.sum((y_pred == 0) & (y_true == 0)).item()  # True Negatives
    fp = torch.sum((y_pred == 1) & (y_true == 0)).item()  # False Positives
    fn = torch.sum((y_pred == 0) & (y_true == 1)).item()  # False Negatives
    
    # Calculate precision, recall, f1-score for binary classification (if needed)
    precision = tp / (tp + fp) if tp + fp > 0 else 0
    recall = tp / (tp + fn) if tp + fn > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0
    
    # You can also compute support for each class (the number of occurrences)
    support = {
        'class_0': torch.sum(y_true == 0).item(),
        'class_1': torch.sum(y_true == 1).item()
    }
    
    # If you want to use the average method (like 'macro', 'micro', 'weighted'), you can:
    if average == 'macro':
        return {
            'precision': precision,
            'recall': recall,
            'f1-score': f1_score,
            'support': support
        }
    
    return precision, recall, f1_score, support





def train_model(model, lr = 0.0001, num_epochs = 30, criterion = nn.CrossEntropyLoss()):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr) 
    max_val_acc = 0
    losses = []
    xs = 0
    val_accs = []
    patience = 3
    counter = 0
    best_val_acc = 0
    with open(config['log_file_path'], 'w') as log_file:
        log_file.write(f"Experiment Log - {time.strftime('%Y-%m-%d %H:%M:%S')}\n")  # Optional header with timestamp
        for epoch in range(num_epochs):
            for i, (data, sent1_lengths, sent2_lengths, labels) in enumerate(train_loader):
                model.train()
                data_batch, sent1_length_batch, sent2_length_batch, label_batch = data.to(device), sent1_lengths.to(device), sent2_lengths.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(data_batch, sent1_length_batch, sent2_length_batch)
                loss = criterion(outputs, label_batch)
                losses.append(loss)
                loss.backward()
                optimizer.step()
                # validate every 100 iterations
                if i > 0 and i % 1000 == 0:
                    # validate
                    val_acc,_,__ = test_model(val_loader, model)
                    val_accs.append(val_acc)
                    xs += 1000
                    if val_acc > max_val_acc:
                        max_val_acc = val_acc
                    log_file.write(f"Epoch: [{epoch+1}/{num_epochs}], Step: [{i+1}/{len(train_loader)}], Validation Acc: {val_acc}\n")
                    log_file.write(f"Epoch: [{epoch+1}/{num_epochs}], Step: [{i+1}/{len(train_loader)}], Validation Loss: {loss}\n")
            

            file_name = f"{config['models_path']}/ffnn_model_{epoch + 1}.pth"
            torch.save(model.state_dict(), file_name)
            val_acc,_,__ = test_model(val_loader, model)
            test_acc,y_test,y_pred = test_model(test_loader, model)
            
            # y_test = np.array(y_test).cpu().numpy() #if y_test.is_cuda else y_test.numpy()
            # y_pred = np.array(y_pred).cpu().numpy() #if y_pred.is_cuda else y_pred.numpy()
            #log_file.write(f"{classification_report(y_test, y_pred)}\n")
            if isinstance(y_test, list):
                y_test = torch.tensor(y_test)
            if isinstance(y_pred, list):
                y_pred = torch.tensor(y_pred)
            log_file.write(f"{classification_report(y_test.cpu(), y_pred.cpu())}\n")
            #log_file.write(f"{compute_metrics(y_test, y_pred)}\n")
            
            log_file.write(f"test accuracy at epoch {epoch+1} is {test_acc}\n")
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                counter = 0
                # Save the model if validation accuracy improves
                torch.save(model.state_dict(), f"{config['best_model_path']}/ffnn_model.pth")
            else:
                counter += 1
                if counter >= patience:
                    log_file.write(f"Validation accuracy didn't improve for {patience} epochs. Early stopping at {epoch+1}\n")
                    break
                    
        log_file.write(f"Max Validation Accuracy: {max_val_acc}\n")
        return max_val_acc, losses, xs, val_accs
    

def test_model(loader, model):
    """
    Helper function that tests the model's performance on a dataset
    """
    correct = 0
    total = 0
    model.eval()
    all_labels = []
    all_predictions = []
    for (data, sent1_lengths, sent2_lengths, labels) in loader:
        data_batch, sent1_length_batch, sent2_length_batch, label_batch = data.to(device), sent1_lengths.to(device), sent2_lengths.to(device), labels.to(device)
        outputs = F.softmax(model(data_batch, sent1_length_batch, sent2_length_batch), dim=1)
        predicted = outputs.max(1, keepdim=True)[1]
        labels = labels.to(device)
        total += labels.size(0)
        correct += predicted.eq(labels.view_as(predicted)).sum().item()
        all_labels.extend(labels)
        all_predictions.extend(predicted.squeeze())
    return (100 * correct / total), all_labels, all_predictions


import os

# Define the path where you want to save the model
model_dir = "Models"

# Check if the directory exists, if not create it
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

# # Define the file name and path
# file_name = os.path.join(model_dir, "model.pth")

# # Save the model's state_dict
# torch.save(model.state_dict(), file_name)



model = TwoSentenceModel(emb_size = EMBED_SIZE, hidden_size=100, num_classes=3).to(device)
print("model",model)
max_val_acc, losses, xs, val_accs = train_model(model, num_epochs=config['num_epochs'])
print("max_val_acc, losses, xs, val_accs", max_val_acc, losses, xs, val_accs)