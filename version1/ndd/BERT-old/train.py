import torch
from tqdm import tqdm
import time
import pickle

from sklearn.metrics import accuracy_score
import os
import gc
import argparse
import pandas as pd
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoConfig, AutoModelForTokenClassification, AutoModel, BertPreTrainedModel
from torch import cuda
from torch.utils.data import ConcatDataset
from data_loader import load_qqp
from test import inference
from torch.utils.data import random_split
from sklearn.metrics import accuracy_score, classification_report



MAX_LEN = 512
BATCH_SIZE = 400
LEARNING_RATE = 1e-5
input_path = './'
num_labels = 2
models_path = 'Models'
os.makedirs(os.path.dirname(models_path), exist_ok=True)

NUM_EPOCHS = 30



class LossFunction(nn.Module):
    def forward(self, probability):
        loss = torch.log(probability)
        loss = -1 * loss
        # print(loss)
        loss = loss.mean()
        # print(loss)
        return loss



class MainModel(BertPreTrainedModel):
    def __init__(self, config, loss_fn = None):
        super(MainModel,self).__init__(config)
        self.num_labels = 2
        self.loss_fn = loss_fn
        config.output_hidden_states = True
        self.bert = AutoModel.from_pretrained("bert-base-uncased",config = config)
        self.classifier = nn.Linear(768, self.num_labels)

    def forward(self, input_ids, attention_mask, token_type_ids, labels,device):
              
        output = self.bert(input_ids, attention_mask = attention_mask, token_type_ids = token_type_ids)
        output = output.last_hidden_state
        output = output[:,0,:] # Use the first token (usually [CLS]) for classification
        classifier_out = self.classifier(output)
        main_prob = F.softmax(classifier_out, dim = 1)
        # main_gold_prob = torch.gather(main_prob, 1, labels)
        
        # Gather the probabilities corresponding to the true class labels
        main_gold_prob = torch.gather(main_prob, 1, labels.unsqueeze(1))  # Fix here
        
        loss_main = self.loss_fn.forward(main_gold_prob)
        return loss_main,main_prob
        
def train(model, dataloader, optimizer, device):
    tr_loss, tr_accuracy = 0, 0
    # tr_preds, tr_labels = [], []
    nb_tr_steps = 0
    #put model in training mode
    model.train()
    
    for idx, batch in enumerate(dataloader):
        #indexes = batch['index']
        input_ids = batch['input_ids'].to(device, dtype=torch.long)  # Correct key
        mask = batch['attention_mask'].to(device, dtype=torch.long)  # Correct key
        targets = batch['labels'].to(device, dtype=torch.long)  # Correct key
        token_type_ids = batch['token_type_ids'].to(device, dtype = torch.long)

        loss_main, main_prob = model(input_ids=input_ids, attention_mask=mask, token_type_ids = token_type_ids, labels=targets,device = device)

        #print(f'\tLoss Main : {loss_main}')
        tr_loss += loss_main.item()
        nb_tr_steps += 1
        #compute training accuracy
        predicted_labels = torch.argmax(main_prob, dim=1)
        # print(predicted_labels.shape)
        targets = targets.view(-1)
        # print(targets.shape)
        tmp_tr_accuracy = accuracy_score(targets.cpu().numpy(), predicted_labels.cpu().numpy())
        tr_accuracy += tmp_tr_accuracy
        if idx % 100 == 0:
            print(f'\tTrain loss at {idx} steps: {tr_loss}')
            if idx != 0:
                print(f'\tTrain Accuracy : {tr_accuracy/nb_tr_steps}')
                with open('live-train.txt', 'a+') as fh:
                    fh.write(f'\tTrain Loss at {idx} steps : {tr_loss}\n')
                    if idx != 0:
                        fh.write(f'\tTrain Accuracy : {tr_accuracy/nb_tr_steps}\n')
                        
                        # y_test = targets
                        # y_pred = predicted_labels
                        
                        # if isinstance(y_test, list):
                        #     y_test = torch.tensor(y_test)
                        # if isinstance(y_pred, list):
                        #     y_pred = torch.tensor(y_pred)
                        # fh.write(f"{classification_report(y_test.cpu(), y_pred.cpu())}\n")
                        
        optimizer.zero_grad()
        loss_main.backward()
        optimizer.step()

    print(f'\tTrain loss for the epoch: {tr_loss}')
    print(f'\tTraining accuracy for epoch: {tr_accuracy/nb_tr_steps}')
    with open('live-train.txt', 'a+') as fh:
        fh.write(f'\tTraining Accuracy : {tr_accuracy/nb_tr_steps}\n')


def valid(model, dataloader, device):
    eval_loss = 0
    eval_accuracy = 0
    model.eval()
    nb_eval_steps = 0
    for idx, batch in enumerate(dataloader):
        #indexes = batch['index']
        input_ids = batch['input_ids'].to(device, dtype=torch.long)  # Correct key
        mask = batch['attention_mask'].to(device, dtype=torch.long)  # Correct key
        targets = batch['labels'].to(device, dtype=torch.long)  # Correct key
        token_type_ids = batch['token_type_ids'].to(device, dtype = torch.long)

        loss_main, main_prob = model(input_ids=input_ids, attention_mask=mask, token_type_ids = token_type_ids, labels=targets,device = device)
        eval_loss += loss_main.item()
        nb_eval_steps += 1
        #compute training accuracy
        predicted_labels = torch.argmax(main_prob, dim=1)
        # print(predicted_labels.shape)
        targets = targets.view(-1)
        # print(targets.shape)
        tmp_eval_accuracy = accuracy_score(targets.cpu().numpy(), predicted_labels.cpu().numpy())
        eval_accuracy += tmp_eval_accuracy
        
        if idx % 100 == 0:
            print(f'\tval loss at {idx} steps: {eval_loss}')
            if idx != 0:
                print(f'\tval Accuracy : {eval_accuracy/nb_eval_steps}')
                with open('live-val.txt', 'a+') as fh:
                    fh.write(f'\n epoch {idx+1}')
                    fh.write(f'\tval Loss at {idx} steps : {eval_loss}\n')
                    if idx != 0:
                        fh.write(f'\tval Accuracy : {eval_accuracy/nb_eval_steps}\n')
                        # y_test = targets
                        # y_pred = predicted_labels
                        
                        # if isinstance(y_test, list):
                        #     y_test = torch.tensor(y_test)
                        # if isinstance(y_pred, list):
                        #     y_pred = torch.tensor(y_pred)
                        # fh.write(f"{classification_report(y_test.cpu(), y_pred.cpu())}\n")
            
    
    return eval_loss, eval_accuracy/nb_eval_steps 

def save_dataset(data, data_path):
    with open(data_path, 'wb') as outp:
        pickle.dump(data, outp, pickle.HIGHEST_PROTOCOL)
    return

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

def collate_fn(batch):
    input_ids = [item['ids'] for item in batch]
    attention_mask = [item['mask'] for item in batch]
    token_type_ids = [item['token_type_ids'] for item in batch]
    labels = [item['target'] for item in batch]

    # Ensure padding and truncation to the max length of 512 tokens
    input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
    attention_mask = torch.nn.utils.rnn.pad_sequence(attention_mask, batch_first=True, padding_value=0)
    token_type_ids = torch.nn.utils.rnn.pad_sequence(token_type_ids, batch_first=True, padding_value=0)
    labels = torch.stack(labels, dim=0)

    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'token_type_ids': token_type_ids,
        'labels': labels
    }






    
def main():
    gc.collect()
    
    torch.cuda.empty_cache()
    print("Training model :")
    start = time.time()
    
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset_name', type=str, required=True)
    parser.add_argument('--output_model_directory', type=str, required=True)
    parser.add_argument('--output_tokenizer_directory', type=str, required=True)
    
    args = parser.parse_args()

    output_model_path = os.path.join(input_path, args.output_model_directory)
    output_tokenizer_path = os.path.join(input_path, args.output_tokenizer_directory)
 
    best_output_model_path = output_model_path + '/BestModel' 
       
    if not os.path.exists(output_model_path):
        os.makedirs(output_model_path)
    if not os.path.exists(best_output_model_path):
        os.makedirs(best_output_model_path)

    train_file_path = os.path.join(input_path, args.dataset_name, 'train.csv')
    test_file_path= os.path.join(input_path, args.dataset_name, 'dev.csv')

    
    
    model = MainModel.from_pretrained("bert-base-uncased", num_labels = 2, loss_fn = LossFunction())
    device = 'cuda' if cuda.is_available() else 'cpu'
    model.to(device)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=LEARNING_RATE)
    
    data = load_qqp(file_path=train_file_path, tokenizer=tokenizer)
    print("data length", data.len)
    val_size = int(0.05 * len(data))
    #train_size = len(data) - val_size
    temp_size = int(0.3 * len(data)) #int(0.95 * len(data))
    train_size = len(data) - val_size - temp_size

    train_data, eval_data, temp_data = random_split(data, [train_size, val_size, temp_size]) #, d2_train_size, d2_test_size, d2_val_size,])
    test_data = load_qqp(file_path=test_file_path, tokenizer=tokenizer)
    test_size = len(test_data)
    print("train, val, test datasize", train_size, val_size, test_size)

    # Use this collate_fn in your DataLoader
    train_dataloader = DataLoader(train_data, shuffle=False, batch_size=BATCH_SIZE, collate_fn=collate_fn)
    eval_dataloader = DataLoader(eval_data, shuffle=False, batch_size=BATCH_SIZE, collate_fn=collate_fn)
    test_dataloader = DataLoader(test_data, shuffle=False, batch_size=BATCH_SIZE, collate_fn=collate_fn)

    
    num_epochs = NUM_EPOCHS
    max_acc = 0.0
    patience = 0
    best_model = model
    best_tokenizer = tokenizer
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}:')
        with open('live-train.txt', 'a+') as fh:
            fh.write(f'Epoch : {epoch+1}\n')
        train(model, train_dataloader, optimizer, device)
        validation_loss, eval_acc = valid(model, eval_dataloader, device)
        test_acc,_,_ = inference(model, test_dataloader, tokenizer, device)
        print(f'\tValidation loss: {validation_loss}')
        print(f'\tValidation accuracy for epoch {epoch+1}: {eval_acc}')
        print(f'\ttest accuracy: {test_acc}')
        
        with open('live-val.txt', 'a+') as fh:
            fh.write(f'\n epoch {epoch+1}')
            fh.write(f'\tValidation Loss : {validation_loss}\n')
            fh.write(f'\tValidation accuracy for epoch {epoch+1}: {eval_acc}\n')
            fh.write(f'\ttest accuracy: {test_acc}\n')
            
            file_name = f"{models_path}/bert_ndd_model_{epoch + 1}.pth"
            torch.save(model.state_dict(), file_name)

        
        if eval_acc > max_acc:
            max_acc = eval_acc
            patience = 0
            best_model = model
            best_tokenizer = tokenizer
            best_model.save_pretrained(best_output_model_path)
            best_tokenizer.save_pretrained(output_tokenizer_path)
        else:
            patience += 1
            if patience > 3:
                print("Early stopping at epoch : ",epoch)
                best_model.save_pretrained(best_output_model_path)
                best_tokenizer.save_pretrained(output_tokenizer_path)
                patience = 0
                #break
            
        model_path = output_model_path + f"/model_{epoch + 1}"
        if not os.path.exists(model_path):
            os.makedirs(model_path)
            
        if epoch < 11 or (epoch+1)%3 == 0:
            model.save_pretrained(model_path)
            tokenizer.save_pretrained(output_tokenizer_path)
        

    

    best_model.save_pretrained(best_output_model_path)
    best_tokenizer.save_pretrained(output_tokenizer_path)

    end = time.time()
    total_time = end - start
    with open('live-val.txt', 'a+') as fh:
        fh.write(f'Total training time : {total_time}\n')

    print(f"Total training time : {total_time}")
    
if __name__ == '__main__':
    main()