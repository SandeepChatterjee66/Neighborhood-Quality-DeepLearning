from model import TwoSentenceModel
import json
from sklearn.metrics import classification_report
import torch
import torch.nn as nn
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from tqdm import tqdm
import numpy as np
from sklearn.decomposition import PCA
import random
import os
import pickle as pkl
import numpy as np
from sklearn.decomposition import PCA
from tqdm import tqdm
import time
import torch.nn.functional as F


EMBED_SIZE = 30
BATCH_SIZE = 1024
MAX_SENT_LEN = 256






with open('config.json', 'r') as f:
    config = json.load(f)
print("config.json loaded")
    
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('device:',device)

log_file_path = config['log_file_path']
log_dir = os.path.dirname(log_file_path)
os.makedirs(log_dir, exist_ok=True)

best_model_path = config['best_model_path']
best_model_dir = os.path.dirname(best_model_path)
os.makedirs(best_model_dir, exist_ok=True)


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


token2id, id2token, word_vectors = build_embedding(FastText)
PAD_IDX = 0
UNK_IDX = 1
len(word_vectors)


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
    
    # print("\nVerifying that the data and label match after shuffling")
    # if verify:
    #     verify_order(sent1s, sent2s, labels)
    #     verify_order(sent1s, sent2s, labels)
          
    print("\nTokenizing sentence 1 list...") 

    print(len(sent1s)) 
    print(type(sent1s[0]))
    
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

from torch.utils.data import Dataset

class TwoSentencesDataset(Dataset):
    """
    Class that represents a train/validation/test dataset that's readable for PyTorch
    """
    
    def __init__(self, sent1_data_list, sent2_data_list, target_list, max_sent_length):
        """
        @param sent1_data_list: list of sentence1's (index matches sentence2's and target_list below)
        @param sent2_data_list: list of sentence2's
        @param target_list: list of correct labels

        """
        self.sent1_data_list = sent1_data_list
        self.sent2_data_list = sent2_data_list
        self.max_sent_len = max_sent_length
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
        sent1_tokens_idx = self.sent1_data_list[key][:self.max_sent_len]
        sent2_tokens_idx = self.sent2_data_list[key][:self.max_sent_len]
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
        padded_vec_1 = np.pad(np.array(datum[0][0]), pad_width=((0,MAX_SENT_LEN -datum[1])), 
                                mode="constant", constant_values=0)
        padded_vec_2 = np.pad(np.array(datum[0][1]), pad_width=((0,MAX_SENT_LEN -datum[2])), 
                                mode="constant", constant_values=0)
        combined_data_list.append([padded_vec_1, padded_vec_2])
    return [torch.from_numpy(np.array(combined_data_list)), 
            torch.LongTensor(sent1_length_list), torch.LongTensor(sent2_length_list), torch.LongTensor(label_list)]
    

def load_data(dataset_path):
    sent1_train = []
    sent2_train = []
    labels_train = []
    cnt = 0
    flag = 0
    lengths = set()
    with open(dataset_path, 'r', encoding='utf-8') as train:
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

    sent1_data_indices, sent2_data_indices, data_label = data_pipeline(sent1_data, sent2_data, data_label)
    max_sent_len = max(max([len(sent) for sent in sent1_data_indices]), max([len(sent) for sent in sent2_data_indices]))
    
    data_dataset = TwoSentencesDataset(sent1_data_indices, sent2_data_indices, data_label, max_sent_length=max_sent_len)
    if dataset_path == config['train_set_path']: MAX_SENT_LEN = max_sent_len
    data_loader = torch.utils.data.DataLoader(dataset=data_dataset, 
                                            batch_size=BATCH_SIZE, 
                                            collate_fn=twosentences_collate_func,
                                            shuffle=False
                                            )
    print(f"Finished creating data_loader {dataset_path}")

    return data_loader, max_sent_len

def train_model(model, train_loader, val_loader, test_loader, lr = 0.001, num_epochs = 30, criterion = nn.CrossEntropyLoss()):
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
            start_time = time.time()
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
                    val_acc,_,__ = test_model(val_loader, model)
                    val_accs.append(val_acc)
                    xs += 1000
                    if val_acc > max_val_acc:
                        max_val_acc = val_acc
                    log_file.write(f"Epoch: [{epoch+1}/{num_epochs}], Step: [{i+1}/{len(train_loader)}], Validation Acc: {val_acc}\n")
                    log_file.write(f"Epoch: [{epoch+1}/{num_epochs}], Step: [{i+1}/{len(train_loader)}], Validation Loss: {loss}\n")
            
            end_time = time.time()
            epoch_time = end_time - start_time
            print(f"epoch {epoch+1} time : {epoch_time}")

            if epoch < 11 or (epoch+1)%3==0:
                file_name = f"{config['models_path']}/ffnn_model_{epoch + 1}.pth"
                torch.save(model.state_dict(), file_name)
            val_acc,_,__ = test_model(val_loader, model)
            test_acc,y_test,y_pred = test_model(test_loader, model)
            log_file.write(f"{classification_report(y_test, y_pred)}\n")
            log_file.write(f"test accuracy at epoch {epoch+1} is {test_acc}\n")
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                counter = 0

                best_model_path = f"{config['best_model_path']}/ffnn_model.pth"  # Should include filename, e.g., "Models/BestModel/ffnn_model.pth"
                best_model_dir = os.path.dirname(best_model_path)  # Extracts "Models/BestModel" from the path
                os.makedirs(best_model_dir, exist_ok=True)
                torch.save(model.state_dict(), best_model_path)  # Save directly to best_model_path
            else:
                counter += 1
                if counter >= patience:
                    log_file.write(f"Validation accuracy didn't improve for {patience} epochs. Early stopping at {epoch+1}\n")
                    #break
                    
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
    with torch.no_grad():  # Disable gradient computation
        for (data, sent1_lengths, sent2_lengths, labels) in loader:
            # Move data to the device
            data_batch = data.to(device)
            sent1_length_batch = sent1_lengths.to(device)
            sent2_length_batch = sent2_lengths.to(device)
            label_batch = labels.to(device)
            
            # Forward pass
            outputs = model(data_batch, sent1_length_batch, sent2_length_batch)
            predicted = outputs.max(1, keepdim=True)[1]  # Get predicted class indices
            
            # Update counts
            total += label_batch.size(0)
            correct += predicted.eq(label_batch.view_as(predicted)).sum().item()
            
            # Collect labels and predictions (convert to CPU and numpy for storage)
            all_labels.extend(label_batch.cpu().numpy())
            all_predictions.extend(predicted.squeeze().cpu().numpy())
    accuracy = 100.0 * correct / total
    return accuracy #, all_labels, all_predictions


def gen_test_preds(test_loader):
    embed_path = "Embeddings/ffnn_nli_test_embeddings_bestmodel.txt"
    os.makedirs(os.path.dirname(embed_path), exist_ok=True)
    
    # Initialize and load the model
    model_test = TwoSentenceModel(
        emb_size=EMBED_SIZE,
        hidden_size=100,
        num_classes=3,
        max_sent_length=MAX_SENT_LEN,
        weights=_WEIGHTS
        #embed_path = embed_path
    ).to(device)
    
    model_test.load_state_dict(torch.load("Models/BestModel/ffnn_model.pth"))
    
    # Evaluate the model
    acc, true_labels, predictions = test_model(test_loader, model_test)
    print(f"Accuracy: {acc:.2f}%")
    
    # Save predictions and labels to correct files
    # Save predictions to 'snli_test_preds.txt'
    preds_path = 'Models/snli_train_preds.txt'
    os.makedirs(os.path.dirname(preds_path), exist_ok=True)
    with open(preds_path, 'w') as f:
        for pred in predictions:
            f.write(f"{pred}\n")
    print(f"Predictions saved to {preds_path}")
    
    # Save true labels to 'snli_test_labels.txt'
    labels_path = 'Models/snli_train_labels.txt'
    with open(labels_path, 'w') as f:
        for label in true_labels:
            f.write(f"{label}\n")
    print(f"True labels saved to {labels_path}")


import csv
import time
import torch


def train_acc():
    # Filepath for saving the CSV
    csv_file = 'accuracies.csv'

    # Define the header once for the CSV file
    header = ['Epoch', 'Train Accuracy', 'Validation Accuracy', 'Test Accuracy']

    # Open the CSV file in append mode (to avoid overwriting existing content)
    with open(csv_file, mode='a', newline='') as f:
        writer = csv.writer(f)

        # Write the header if the file is empty
        if f.tell() == 0:
            writer.writerow(header)

        train_set_path = config['train_set_path']
        validation_set_path = config['val_set_path']
        test_set_path = config['test_set_path']

        # Load data
        train_loader, l1 = load_data(train_set_path)
        val_loader, l2 = load_data(validation_set_path)
        test_loader, l3 = load_data(test_set_path)

        # Define the epochs
        epochs = list(range(1, 13)) + [15, 18, 21, 27, 30]

        # Loop through epochs
        for epoch in epochs:
            start = time.time()

            # Initialize model
            model = TwoSentenceModel(
                emb_size=EMBED_SIZE,
                hidden_size=100,
                num_classes=3,
                max_sent_length=MAX_SENT_LEN,
                weights=_WEIGHTS
            ).to(device)
            model.load_state_dict(torch.load(f"Models/ffnn_model_{epoch}.pth"))

            # Get the accuracies
            train_accuracy = test_model(model=model, loader=train_loader)
            val_accuracy = test_model(model=model, loader=val_loader)
            test_accuracy = test_model(model=model, loader=test_loader)

            # Record results
            writer.writerow([epoch, train_accuracy, val_accuracy, test_accuracy])

            end = time.time()
            total_time = end - start
            print(f"Inference single pass - time taken for epoch {epoch}: {total_time}")



    
    

def train():
    #to-do
    # check if some file exists at dataset_paths

    train_set_path = config['train_set_path']
    validation_set_path = config['val_set_path']
    test_set_path = config['test_set_path']




    train_loader, l1 = load_data(train_set_path)
    val_loader, l2 = load_data(validation_set_path)
    test_loader, l3 = load_data(test_set_path)
    
    model = TwoSentenceModel(
        emb_size=EMBED_SIZE,
        hidden_size=100,
        num_classes=3,
        max_sent_length=MAX_SENT_LEN,  # Ensure this is correct
        weights=_WEIGHTS
    ).to(device)


    max_val_acc, losses, xs, val_accs = train_model(model, train_loader=train_loader, val_loader=val_loader, test_loader=test_loader, num_epochs=30)
    print("max_val_acc, losses, xs, val_accs")
    print(max_val_acc, losses, xs, val_accs, sep="\n\n------\n\n")
    
def gen_embed(loader, model):
    model.eval()
    for (data, sent1_lengths, sent2_lengths, labels) in loader:
        data_batch, sent1_length_batch, sent2_length_batch, label_batch = data.to(device), sent1_lengths.to(device), sent2_lengths.to(device), labels.to(device)
        model(data_batch, sent1_length_batch, sent2_length_batch)
      
    
def gen_embeddings(train_loader, num_epochs):
    for epoch in range(1,1+num_epochs):
        if epoch >= 12 and epoch%3!=0:
            continue
        start = time.time()
        embed_path = f"Embeddings/ffnn_nli_train_embeddings_model_{epoch}.txt"
        os.makedirs(os.path.dirname(embed_path), exist_ok=True)
        model = TwoSentenceModel(
            emb_size=EMBED_SIZE,
            hidden_size=100,
            num_classes=3,
            max_sent_length=MAX_SENT_LEN,  # Ensure this is correct
            weights=_WEIGHTS,
            embed_path=embed_path
        ).to(device)
        model.load_state_dict(torch.load(f"Models/ffnn_model_{epoch}.pth"))
        gen_embed(train_loader, model)
        
        
        end = time.time()
        total_time = end - start
        print(f"Embedding Gen - time taken {epoch}:",total_time) 

# def gen_test_preds(test_loader):
#     embed_path = f"Embeddings/ffnn_nli_test_embeddings_bestmodel.txt"
#     os.makedirs(os.path.dirname(embed_path), exist_ok=True)
#     model_test = TwoSentenceModel(
#             emb_size=EMBED_SIZE,
#             hidden_size=100,
#             num_classes=3,
#             max_sent_length=MAX_SENT_LEN,  # Ensure this is correct
#             weights=_WEIGHTS
#         ).to(device)
#     model_test.load_state_dict(torch.load("Models/BestModel/ffnn_model.pth"))

#     acc, labels, preds = test_model(test_loader, model_test)
#     print(f"Accuracy is {acc}")

#     print(len(preds))
#     nested_list = preds
#     flat_list = [item for sublist in nested_list for item in sublist]
#     print(flat_list)
#     testpred_file_path = 'Models/snli_test_labels.txt'
#     os.makedirs(os.path.dirname(testpred_file_path), exist_ok=True)
#     with open(testpred_file_path, 'w+') as file:
#         for item in flat_list:
#             file.write(f"{item}\n")
#     print(f"List written to {testpred_file_path}")
    
#     print(len(labels))
#     nested_list = labels
#     flat_list = [item for sublist in nested_list for item in sublist]
#     print(flat_list)
#     testpred_file_path = 'Models/snli_test_preds.txt'
#     os.makedirs(os.path.dirname(testpred_file_path), exist_ok=True)
#     with open(testpred_file_path, 'w+') as file:
#         for item in flat_list:
#             file.write(f"{item}\n")
#     print(f"List written to {testpred_file_path}")