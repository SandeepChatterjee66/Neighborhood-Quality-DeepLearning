import argparse
import gc
import os
import pickle
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
from torch import cuda
from torch.utils.data import DataLoader

# from tqdm import tqdm
from transformers import (  # AutoConfig,; AutoModelForTokenClassification,
    AutoModel,
    AutoTokenizer,
    BertPreTrainedModel,
)

from data_loader import load_snli

# from test import inference


# from torch.utils.data import ConcatDataset, Dataset

# from torchvision import datasets
# from torchvision.transforms import ToTensor


MAX_LEN = 512
BATCH_SIZE = 64
LEARNING_RATE = 1e-5
input_path = "./"
output_tokenizer_path = 'Tokenizer'
os.makedirs(output_tokenizer_path, exist_ok=True)
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
    def __init__(self, config, loss_fn=None):
        super(MainModel, self).__init__(config)
        self.num_labels = 3
        self.loss_fn = loss_fn
        config.output_hidden_states = True
        self.bert = AutoModel.from_pretrained("bert-base-uncased", config=config)
        self.hidden = nn.Linear(768, 2 * (self.num_labels))
        self.classifier = nn.Linear(2 * (self.num_labels), self.num_labels)

    def forward(self, input_ids, attention_mask, token_type_ids, labels, device):

        output = self.bert(
            input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids
        )
        output = output.last_hidden_state
        output = output[:, 0, :]
        hidden_output = self.hidden(output)
        classifier_out = self.classifier(hidden_output)
        main_prob = F.softmax(classifier_out, dim=1)
        main_gold_prob = torch.gather(main_prob, 1, labels)
        loss_main = self.loss_fn.forward(main_gold_prob)
        return loss_main, main_prob


def train(model, dataloader, optimizer, device):
    tr_loss, tr_accuracy = 0, 0
    # bias_loss = 0
    # tr_preds, tr_labels = [], []
    nb_tr_steps = 0
    # put model in training mode
    model.train()

    for idx, batch in enumerate(dataloader):
        input_ids = batch["ids"].to(device, dtype=torch.long)
        mask = batch["mask"].to(device, dtype=torch.long)
        targets = batch["target"].to(device, dtype=torch.long)
        token_type_ids = batch["token_type_ids"].to(device, dtype=torch.long)

        loss_main, main_prob = model(
            input_ids=input_ids,
            attention_mask=mask,
            token_type_ids=token_type_ids,
            labels=targets,
            device=device,
        )
        tr_loss += loss_main.item()
        nb_tr_steps += 1
        # compute training accuracy
        predicted_labels = torch.argmax(main_prob, dim=1)
        targets = targets.view(-1)
        tmp_tr_accuracy = accuracy_score(
            targets.cpu().numpy(), predicted_labels.cpu().numpy()
        )
        tr_accuracy += tmp_tr_accuracy
        if idx % 100 == 0:
            print(f"\tModel loss at {idx} steps: {tr_loss}")
            if idx != 0:
                print(f"\tModel Accuracy : {tr_accuracy / nb_tr_steps}")
            with open("live.txt", "a") as fh:
                fh.write(f"\tModel Loss at {idx} steps : {tr_loss}\n")
                if idx != 0:
                    fh.write(f"\tModel Accuracy : {tr_accuracy / nb_tr_steps}")
        optimizer.zero_grad()
        loss_main.backward()
        optimizer.step()

    print(f"\tModel loss for the epoch: {tr_loss}")
    print(f"\tTraining accuracy for epoch: {tr_accuracy / nb_tr_steps}")
    with open("live.txt", "a") as fh:
        fh.write(f"\tTraining Accuracy : {tr_accuracy / nb_tr_steps}\n")


def valid(model, dataloader, device):
    eval_loss = 0
    # bias_loss = 0
    eval_accuracy = 0
    model.eval()
    nb_eval_steps = 0
    for batch in dataloader:
        # indexes = batch["index"]
        input_ids = batch["ids"].to(device, dtype=torch.long)
        mask = batch["mask"].to(device, dtype=torch.long)
        targets = batch["target"].to(device, dtype=torch.long)
        token_type_ids = batch["token_type_ids"].to(device, dtype=torch.long)

        loss_main, main_prob = model(
            input_ids=input_ids,
            attention_mask=mask,
            token_type_ids=token_type_ids,
            labels=targets,
            device=device,
        )
        eval_loss += loss_main.item()
        nb_eval_steps += 1
        # compute training accuracy
        predicted_labels = torch.argmax(main_prob, dim=1)
        # print(predicted_labels.shape)
        targets = targets.view(-1)
        # print(targets.shape)
        tmp_eval_accuracy = accuracy_score(
            targets.cpu().numpy(), predicted_labels.cpu().numpy()
        )
        eval_accuracy += tmp_eval_accuracy

    return eval_loss, eval_accuracy / nb_eval_steps


def read_dataset(data_path):
    with open(data_path, "rb") as inp:
        data = pickle.load(inp)
    return data


def save_dataset(data, data_path):
    with open(data_path, "wb") as outp:
        pickle.dump(data, outp, pickle.HIGHEST_PROTOCOL)
    return


def main():
    gc.collect()

    torch.cuda.empty_cache()
    print("Training model :")
    start = time.time()

    parser = argparse.ArgumentParser()

    parser.add_argument("--train_file_path", type=str, required=True)
    parser.add_argument("--dev_file_path", type=str, required=True)
    parser.add_argument("--output_model_directory", type=str, required=True)

    args = parser.parse_args()

    output_model_path = os.path.join(input_path, args.output_model_directory)

    best_output_model_path = output_model_path + "/BestModel"
    if not os.path.exists(output_model_path):
        os.makedirs(output_model_path)
    if not os.path.exists(best_output_model_path):
        os.makedirs(best_output_model_path)
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    data = load_snli(file_path=args.train_file_path, tokenizer=tokenizer)
    print(f"SNLI train length : {data.len}")

    model = MainModel.from_pretrained(
        "bert-base-uncased", num_labels=3, loss_fn=LossFunction()
    )
    device = "cuda" if cuda.is_available() else "cpu"
    model.to(device)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=LEARNING_RATE)

    eval_data = load_snli(file_path=args.dev_file_path, tokenizer=tokenizer)
    print(f"Validation dataset length : {len(eval_data)}")

    # save_dataset(eval_data, './dev.pkl')

    train_dataloader = DataLoader(data, shuffle=True, batch_size=BATCH_SIZE)
    eval_dataloader = DataLoader(eval_data, shuffle=True, batch_size=BATCH_SIZE)

    num_epochs = NUM_EPOCHS
    max_acc = 0.0
    patience = 0
    best_model = model
    best_tokenizer = tokenizer
    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}:")
        with open("live.txt", "a") as fh:
            fh.write(f"Epoch : {epoch + 1}\n")
        train(model, train_dataloader, optimizer, device)
        validation_loss, eval_acc = valid(model, eval_dataloader, device)
        print(f"\tValidation loss: {validation_loss}")
        print(f"\tValidation accuracy for epoch: {eval_acc}")
        with open("live.txt", "a+") as fh:
            fh.write(f"\tValidation Loss : {validation_loss}\n")
            fh.write(f"\tValidation accuracy for epoch: {eval_acc}\n")

        if eval_acc > max_acc:
            max_acc = eval_acc
            patience = 0
            best_model = model
            best_tokenizer = tokenizer
            best_model.save_pretrained(best_output_model_path)
            best_tokenizer.save_pretrained(best_output_model_path)
        else:
            patience += 1
            if patience > 2:
                print("Early stopping at epoch : ", epoch)
                best_model.save_pretrained(best_output_model_path)
                best_tokenizer.save_pretrained(best_output_model_path)
                patience = 0
                #break
        if epoch < 11 or (epoch+1) % 3 == 0:
            model_path = output_model_path + f"_{epoch + 1}"
            if not os.path.exists(model_path):
                os.makedirs(model_path)
            model.save_pretrained(model_path)
            tokenizer.save_pretrained(output_tokenizer_path)

    # model.save_pretrained(output_model_path)
    # tokenizer.save_pretrained(output_tokenizer_path)

    # best_model.save_pretrained(best_output_model_path)
    # best_tokenizer.save_pretrained(best_output_model_path)

    end = time.time()
    total_time = end - start
    with open("live.txt", "a") as fh:
        fh.write(f"Total training time : {total_time}\n")

    print(f"Total training time : {total_time}")


if __name__ == "__main__":
    main()
