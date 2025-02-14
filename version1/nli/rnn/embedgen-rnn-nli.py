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
import time
import json



with open('config.json', 'r') as f:
    config = json.load(f)
print("config.json loaded")

BATCH_SIZE = 1024 #config['embedding_batch_size']
pred_file_path = 'RNN_NLI/snli_test_preds.txt'
label_file_path = 'RNN_NLI/snli_test_labels.txt'
train_file_path = config["train_set_path"]
fasttext_embeddings_path = config["fasttext_embeddings_path"]
best_model_path = "./RNN_NLI/BestModel/rnn_model.pth"

embeddings_output_path = f'Embeddings/Train/rnn_model_1_nli.txt'
epochs = 10
EMBED_SIZE = 300




device = torch.device("cuda:0")
print(device)

FastText = []
cnt = 0
with open(fasttext_embeddings_path, "r") as ft:
    for i, line in enumerate(ft):
        if i == 0:
            continue
        FastText.append(line)
        cnt+=1
print(cnt)

import os
import pickle as pkl
import numpy as np
from sklearn.decomposition import PCA
from tqdm import tqdm


PICKLE_FILE = f"embeddings{EMBED_SIZE}.pkl"

def build_embedding(data):
    word2id = {"<pad>": 0, "<unk>": 1}
    id2word = {0: "<pad>", 1: "<unk>"}
    
    embeddings = [np.zeros(300), np.random.normal(0, 0.01, 300)]
    #pca = PCA(n_components=EMBED_SIZE)
    
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
    
    #print("Applying PCA transformation...")
    #pca.fit(np.array(embeddings))
    
    # Apply PCA transformation to all embeddings
    transformed_embeddings = []
    # for i in tqdm(range(len(embeddings)), desc="PCA Transformation", position=1):
    #     transformed_embeddings.append(pca.transform(embeddings[i].reshape(1, -1)).flatten())
    
    transformed_embeddings = embeddings
    
    # Save only the PCA-transformed embeddings and the word2id, id2word to a pickle file
    with open(PICKLE_FILE, 'wb') as f:
        print(f"Saving PCA-transformed embeddings to {PICKLE_FILE}")
        pkl.dump((word2id, id2word, transformed_embeddings), f)
    
    return word2id, id2word, transformed_embeddings


token2id, id2token, word_vectors = build_embedding(FastText)
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
    i = 2
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


_weights = np.array(word_vectors)
_WEIGHTS = _weights
_WEIGHTS.shape

def data_pipeline(sent1s, sent2s, labels, verify=True):
    labels = convert_labels_to_integers(labels)
    # seed = random.randint(1, 100)
    # print("Random seed for shuffling: {}".format(seed))
    # random.Random(seed).shuffle(sent1s)
    # random.Random(seed).shuffle(sent2s)
    # random.Random(seed).shuffle(labels)
    
    #print("\nVerifying that the data and label match after shuffling")
    print(sent1s[2])
    print(sent2s[2])
    # if verify:
    #     verify_order(sent1s, sent2s, labels)
    #     verify_order(sent1s, sent2s, labels)
          
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
    padded_vec = []
    for datum in batch:
        label_list.append(datum[3])
        sent1_length_list.append(datum[1])
        sent2_length_list.append(datum[2])
    # padding
    for datum in batch:
        padded_vec.append(np.pad(np.concatenate((np.array(datum[0][0]),np.array(datum[0][1]))), pad_width=((0,2*MAX_SENTENCE_LENGTH-datum[1]-datum[2])), 
                                mode="constant", constant_values=0))
        
    return [torch.from_numpy(np.array(padded_vec)), 
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
            # print(s1)
            # print(s2)
            # print(parts)
            # break
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
        # if len(parts)==13:
        #     print(s1)
        #     print(s2)
        #     break
            
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
        # if len(parts)==13:
        #     print(s1)
        #     print(s2)
        #     break
            
    print(lengths)

sent1_test_indices, sent2_test_indices, test_label = data_pipeline(sent1_test, sent2_test, test_label)
test_dataset = TwoSentencesDataset(sent1_test_indices, sent2_test_indices, test_label)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, 
                                           batch_size=BATCH_SIZE, 
                                           collate_fn=twosentences_collate_func,
                                           #shuffle=True
                                          )

'''
#----------------------------------------------
'''

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
        # print((reverse_sorted_lengths).device)
        reverse_sorted_lengths = reverse_sorted_lengths.to(x.device)
        reverse_sorted_lengths = reverse_sorted_lengths.cpu().numpy()
        ordered_sents = x
        reverse_sorted_data = ordered_sents[reverse_sorted_indices].to(device)
        # get embedding
        embed = self.embedding(reverse_sorted_data)
        
        # pack padded sequence
        embed = torch.nn.utils.rnn.pack_padded_sequence(embed, reverse_sorted_lengths, batch_first=True)
        
        self.hidden = self.init_hidden(batch_size)
        rnn_out, self.hidden = self.rnn(embed, self.hidden)
        
        ### MATCHING BACK
        change_back_indices = reverse_sorted_indices.argsort()
        # print(change_back_indices)
        self.hidden = self.hidden[:, change_back_indices]
        
        ### GRU stuff
        hidden_sents = torch.cat([self.hidden[0, :, :], self.hidden[1, :, :]], dim=1)
        save_embeddings_to_text_file(hidden_sents, embeddings_output_path)
        #need to change the file name to create embeddings after every epoch
        
        linear1 = self.linear1(hidden_sents)
#       addition of encoded sentences
        linear1 = F.relu(linear1.contiguous().view(-1, linear1.size(-1))).view(linear1.shape)   
        #linear1 = self.dropout(linear1)
        logits = self.linear2(linear1)
        return logits
    
def gen_embed(loader, model):
    model.eval()
    for (data, sent1_lengths, sent2_lengths, labels) in loader:
        data_batch, sent1_length_batch, sent2_length_batch, label_batch = data.to(device), sent1_lengths.to(device), sent2_lengths.to(device), labels.to(device)
        model(data_batch, sent1_length_batch, sent2_length_batch)
        

for epoch in range(1,1+epochs):
    start = time.time()
    model = TwoSentenceModel(emb_size = 300, hidden_size=300, num_layers=1, num_classes=3).to(device)
    model_path = f"./RNN_NLI/rnn_model_{epoch}.pth"
    model.load_state_dict(torch.load(model_path))
    
    print("\nTrain Embeddings",epoch)
    #change the model path to create embeddings after every epoch
    embeddings_output_path = f'Embeddings/Train/rnn_model_{epoch}_nli.txt'
    os.makedirs(os.path.dirname(embeddings_output_path), exist_ok=True)
    gen_embed(train_loader, model)
    # print("\nTest Embeddings")
    # embeddings_output_path = f"Embeddings/Test/rnn_model_{epoch}_nli.txt"
    # os.makedirs(os.path.dirname(embeddings_output_path), exist_ok=True)
    # gen_embed(test_loader, model)
    
    end = time.time()
    total_time = end - start
    print(f"{epoch} total time taken is {total_time}") 



# Function for testing the model
def test_model(loader, model):
    """
    Helper function that tests the model's performance on a dataset
    """
    correct = 0
    total = 0
    model.eval()
    for (data, sent1_lengths, sent2_lengths, labels) in loader:
        data_batch, sent1_length_batch, sent2_length_batch, label_batch = data.to(device), sent1_lengths.to(device), sent2_lengths.to(device), labels.to(device)
        outputs = F.softmax(model(data_batch, sent1_length_batch, sent2_length_batch), dim=1)
        predicted = outputs.max(1, keepdim=True)[1]
        labels = labels.to(device)
        total += labels.size(0)
        correct += predicted.eq(labels.view_as(predicted)).sum().item()
    return (100 * correct / total), predicted, labels


model_test = TwoSentenceModel(emb_size = 300, hidden_size=300, num_layers=1, num_classes=3).to(device)
model_test.load_state_dict(torch.load(best_model_path))

acc, predicted, labels = test_model(test_loader, model_test)
print(f"accuracy is {acc}")

print(len(labels))
print(labels[:10])
nested_list = labels
flat_list = [item for sublist in nested_list for item in sublist]
print(flat_list)
with open(label_file_path, 'w+') as file:
    for item in flat_list:
        if item == 0:
            file.write("contradiction \n")
        elif item == 1:
            file.write("neutral\n")
        elif item == 2:
            file.write("entailment\n")
print(f"List written to {label_file_path}")

print(len(predicted))
print(predicted[:10])
nested_list = predicted
flat_list = [item for sublist in nested_list for item in sublist]
print(flat_list)
with open(pred_file_path, 'w+') as file:
    for item in flat_list:
        if item == 0:
            file.write("contradiction \n")
        elif item == 1:
            file.write("neutral\n")
        elif item == 2:
            file.write("entailment\n")
print(f"List written to {pred_file_path}")

