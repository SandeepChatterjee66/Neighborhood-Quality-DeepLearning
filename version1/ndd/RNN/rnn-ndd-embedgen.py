import numpy as np
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
import json, os
import time
from train import build_embedding, convert_labels_to_integers
from train import tokenize, token2index_dataset, data_pipeline
from train import TwoSentencesDataset, twosentences_collate_func
from train import test_model, train_model

#batch size 128
BATCH_SIZE = 512
EMBED_SIZE = 300
qqp_groundtruth_path = 'QQP_groundtruth_test.txt'
EMBED_PATH = './RNN_NDD/rnn_model_7_test_embeddings.txt'

with open('config.json', 'r') as f:
    config = json.load(f)
print("config.json loaded")


    
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('device:',device)

best_model_path = 'RNN_NDD/BestModel/rnn_model.pth'
directory = os.path.dirname(best_model_path)
# Check if the directory exists; if not, create it
if not os.path.exists(directory):
    os.makedirs(directory)
    print(f"Directory '{directory}' created.")
else:
    print(f"Directory '{directory}' already exists.")


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


PICKLE_FILE = f"embeddings{EMBED_SIZE}.pkl"









#token2id, id2token, word_vectors = build_embedding(FastText)
PAD_IDX = 0
UNK_IDX = 1



import collections


print("starting to build embeddings")
token2id, id2token, word_vectors = build_embedding(FastText)
PAD_IDX = 0
UNK_IDX = 1
print("built the embeddings", len(token2id), len(id2token), len(word_vectors))


_weights = np.array(word_vectors)
_WEIGHTS = _weights
_WEIGHTS.shape



import numpy as np
import torch
from torch.utils.data import Dataset

    

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
with open(qqp_groundtruth_path, 'w') as file:
    for item in test_label:
        file.write(f"{item}\n")
print(f"List written to {qqp_groundtruth_path}")



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
    
    def __init__(self, hidden_size, num_layers, num_classes, emb_size = 300):
        # RNN Accepts the following hyperparams:
        # emb_size: Embedding Size
        # hidden_size: Hidden Size of layer in RNN
        # num_layers: number of layers in RNN
        # num_classes: number of output classes
        # vocab_size: vocabulary size
        super(TwoSentenceModel, self).__init__()

        self.num_layers, self.hidden_size = num_layers, hidden_size
        weight = torch.FloatTensor(_WEIGHTS)
        self.embedding = nn.Embedding.from_pretrained(weight)
        self.rnn = nn.GRU(emb_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.linear1 = nn.Linear(2*hidden_size, 100)
        self.linear2 = nn.Linear(100, num_classes)
        self.dropout = nn.Dropout(0.2)

    def init_hidden(self, batch_size):
        return torch.randn(2, batch_size, self.hidden_size).to(device)


    def forward(self, x, sent1_lengths, sent2_lengths):
        # reset hidden state
        batch_size = x.size()[0]
        
        ordered_slengths = sent1_lengths + sent2_lengths
        
    
        reverse_sorted_lengths, reverse_sorted_indices = torch.sort(ordered_slengths, descending=True)
        reverse_sorted_lengths = reverse_sorted_lengths.to(x.device)
        reverse_sorted_lengths = reverse_sorted_lengths.cpu().numpy()
        ordered_sents = x
        reverse_sorted_data = ordered_sents[reverse_sorted_indices].to(device)
        # get embedding
        embed = self.embedding(reverse_sorted_data)
        
        

        # pack padded sequence
        embed = torch.nn.utils.rnn.pack_padded_sequence(embed, reverse_sorted_lengths, batch_first=True)
            
        self.hidden = self.init_hidden(batch_size)
        # fprop though RNN
        rnn_out, self.hidden = self.rnn(embed, self.hidden)
        
        ### MATCHING BACK
        
        change_back_indices = reverse_sorted_indices.argsort()
        self.hidden = self.hidden[:, change_back_indices]
              
        ### GRU stuff
        hidden_sents = torch.cat([self.hidden[0, :, :], self.hidden[1, :, :]], dim=1)
        save_embeddings_to_text_file(hidden_sents, EMBED_PATH)
        #need to change the file name to create embeddings after every epoch
        linear1 = self.linear1(hidden_sents)
        linear1 = F.relu(linear1.contiguous().view(-1, linear1.size(-1))).view(linear1.shape)   
        linear1 = self.dropout(linear1)
        logits = self.linear2(linear1)
        return logits


def gen_embed(loader, model):
    """
    Helper function that tests the model's performance on a dataset
    """
    model.eval()
    for (data, sent1_lengths, sent2_lengths, labels) in loader:
        data_batch, sent1_length_batch, sent2_length_batch, label_batch = data.to(device), sent1_lengths.to(device), sent2_lengths.to(device), labels.to(device)
        model(data_batch, sent1_length_batch, sent2_length_batch)
        

start = time.time()
model = TwoSentenceModel(emb_size = 300, hidden_size=300, num_layers=1, num_classes=2).to(device)
model.load_state_dict(torch.load("./RNN_NDD/rnn_model_7.pth"))
#change the model path to create embeddings after every epoch
gen_embed(test_loader, model)
end = time.time()
total_time = end - start
print("total time taken is ",total_time) 


model_test = TwoSentenceModel(emb_size = 300, hidden_size=300, num_layers=1, num_classes=2).to(device)
model.load_state_dict(torch.load("./RNN_NDD/BestModel/rnn_model.pth"))

_, labels = test_model(test_loader, model_test)

print(len(labels))

nested_list = labels
flat_list = [item for sublist in nested_list for item in sublist]
print(flat_list)


file_path = './RNN_NDD/qqp_test_preds_corrected.txt'

# Write the list to the file
with open(file_path, 'w') as file:
    for item in flat_list:
        file.write(f"{item}\n")

print(f"List written to {file_path}")




