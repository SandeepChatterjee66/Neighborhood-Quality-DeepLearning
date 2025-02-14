import argparse
import gc
import time
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import cuda
from torch.utils.data import DataLoader

# from torchcrf import CRF
from tqdm import tqdm
from transformers import (
    # AutoConfig,
    AutoModel,
    # AutoModelForTokenClassification,
    AutoTokenizer,
    BertPreTrainedModel,
)

from data_loader import load_snli

# Ignore all warnings
warnings.filterwarnings("ignore")


input_path = "./"
log_soft = F.log_softmax
print(torch.version.cuda)
MAX_LEN = 512  # suitable for all datasets
BATCH_SIZE = 64
LEARNING_RATE = 1e-5
num_labels = 3


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
        return output


def save_embeddings_to_text_file(embeddings, output_file_path):
    with open(output_file_path, "a") as file:
        for embedding in embeddings:
            for i, emb in enumerate(embedding):
                if i != len(embedding) - 1:
                    file.write(f"{emb} ")
                else:
                    file.write(f"{emb}\n")


def generate_embeddings(model, dataloader, device, output_file_path):
    embeddings = []
    labels = []
    total_embeddings = 0
    for idx, batch in enumerate(tqdm(dataloader, ncols=100)):
        # indexes = batch["index"]
        input_ids = batch["ids"].to(device, dtype=torch.long)
        mask = batch["mask"].to(device, dtype=torch.long)
        targets = batch["target"].to(device, dtype=torch.long)
        token_type_ids = batch["token_type_ids"].to(device, dtype=torch.long)
        labels.extend([target.item() for target in targets])
        with torch.no_grad():
            output = model(
                input_ids=input_ids,
                attention_mask=mask,
                token_type_ids=token_type_ids,
                labels=targets,
                device=device,
            )
        embeddings = output.tolist()
        save_embeddings_to_text_file(embeddings, output_file_path)
        total_embeddings += len(embeddings)
    if output_file_path == "./Predictions/train_embeddings.txt":
        np.savetxt("./Predictions/train_groundtruth.txt", labels)
    print(f"Total embeddings saved in {output_file_path} : {total_embeddings}")
    return embeddings


def main():
    gc.collect()

    torch.cuda.empty_cache()
    start = time.time()

    parser = argparse.ArgumentParser()

    parser.add_argument("--input_model_path", type=str, required=True)

    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.input_model_path)

    model = MainModel.from_pretrained(args.input_model_path, num_labels=3, loss_fn=None)
    device = "cuda" if cuda.is_available() else "cpu"

    model.to(device)

    train_data = load_snli(
        file_path="./snli_1.0/snli_1.0_train.txt", tokenizer=tokenizer
    )

    train_path = "./Predictions/train_embeddings.txt"
    train_dataloader = DataLoader(train_data, shuffle=False, batch_size=BATCH_SIZE)
    generate_embeddings(model, train_dataloader, device, train_path)

    snli_data = load_snli(file_path="./snli_1.0/snli_1.0_test.txt", tokenizer=tokenizer)
    snli_test_dataloader = DataLoader(snli_data, shuffle=False, batch_size=BATCH_SIZE)
    snli_embedding_path = "./Predictions/SNLI/SNLI_embeddings.txt"
    generate_embeddings(model, snli_test_dataloader, device, snli_embedding_path)

    end = time.time()
    total_time = end - start
    with open("live.txt", "a") as fh:
        fh.write(f"Total training time : {total_time}\n")

    print(f"Total training time : {total_time}")


if __name__ == "__main__":
    main()
