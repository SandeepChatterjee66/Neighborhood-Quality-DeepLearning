import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from collections import Counter
import pickle as pkl
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
import pandas as pd
import json, os
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
import numpy as np
from sklearn.decomposition import PCA
import os
import pickle as pkl
import numpy as np
from sklearn.decomposition import PCA
from tqdm import tqdm

#batch size 128
BATCH_SIZE = 512
device = torch.device("cuda:0")
print(device)

EMBED_SIZE = 300

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
_WEIGHTS.shape

def data_pipeline(sent1s, sent2s, labels, verify=True):
    labels = convert_labels_to_integers(labels)
    print(len(sent1s)) 
    print((sent1s[-1]))
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
    #print("collate batch: ", batch[0][0])
    #batch[0][0] = batch[0][0][:MAX_SENTENCE_LENGTH]
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
        linear1 = self.linear1(hidden_sents)
        linear1 = F.relu(linear1.contiguous().view(-1, linear1.size(-1))).view(linear1.shape)   
        linear1 = self.dropout(linear1)
        logits = self.linear2(linear1)
        return logits



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
    return (100 * correct / total)

def train_model(model, lr = 0.001, num_epochs = 30, criterion = nn.CrossEntropyLoss()):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr) 
    max_val_acc = 0
    losses = []
    xs = 0
    val_accs = []
    patience = 3
    counter = 0
    best_val_acc = 0
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
            if i > 0 and i % 100 == 0:
                # validate
                val_acc = test_model(val_loader, model)
                val_accs.append(val_acc)
                xs += 1000
                if val_acc > max_val_acc:
                    max_val_acc = val_acc
                print('Epoch: [{}/{}], Step: [{}/{}], Validation Acc: {}'.format( 
                           epoch+1, num_epochs, i+1, len(train_loader), val_acc))
                print('Epoch: [{}/{}], Step: [{}/{}], Training Loss: {}'.format( 
                           epoch+1, num_epochs, i+1, len(train_loader), loss))

        if epoch < 11 or (epoch+1)%3==0:
            file_name = f"RNN_NDD/rnn_model_{epoch + 1}.pth"
            torch.save(model.state_dict(), file_name)
            val_acc = test_model(val_loader, model)
            test_acc = test_model(test_loader, model)
            print(f"val accuracy at epoch {epoch+1} is {val_acc}")
            print(f"test accuracy at epoch {epoch+1} is {test_acc}")
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            counter = 0
            # Save the model if validation accuracy improves
            torch.save(model.state_dict(), best_model_path)
        else:
            counter += 1
            if counter > patience:
                print(f"Validation accuracy didn't improve for {patience} epochs. Early stopping at {epoch+1}")
                #break
                
    print("Max Validation Accuracy: {}".format(max_val_acc))
    return max_val_acc, losses, xs, val_accs

model = TwoSentenceModel(emb_size = 300, hidden_size=300, num_layers=1, num_classes=2).to(device)
max_val_acc, losses, xs, val_accs = train_model(model, num_epochs=30)