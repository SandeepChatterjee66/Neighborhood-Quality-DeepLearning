import torch
import torch.nn as nn

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('device:',device)

def save_embeddings_to_text_file(embeddings, output_file_path):
    with open(output_file_path, 'a+') as file:
        for embedding in embeddings:
            for i,emb in enumerate(embedding):
                if i != len(embedding) - 1:
                    file.write(f'{emb} ')
                else:
                    file.write(f'{emb}\n')

class TwoSentenceModel(nn.Module):
    def __init__(self, hidden_size, num_classes, emb_size, max_sent_length, weights, embed_path=""):
        super(TwoSentenceModel, self).__init__()

        self.hidden_size = hidden_size
        
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

        # Path to save embeddings
        self.embed_path = embed_path

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
#     def __init__(self, hidden_size, num_classes, emb_size = 300, max_sent_length = 256, embed_path = "", weights = None):
#         super(TwoSentenceModel, self).__init__()

#         self.hidden_size = hidden_size
#         weight = torch.FloatTensor(weights)
#         self.embedding = nn.Embedding.from_pretrained(weight)
#         self.linear1 = nn.Linear(2*emb_size*max_sent_length, hidden_size)
#         self.linear2 = nn.Linear(hidden_size, hidden_size)  # New hidden layer with hidden_size neurons
#         self.linear3 = nn.Linear(hidden_size,hidden_size)
#         self.linear4 = nn.Linear(hidden_size,hidden_size)
#         self.linear5 = nn.Linear(hidden_size, num_classes)
#         logits = nn.Linear(hidden_size, num_classes)  # Output layer
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
        
#         # First hidden layer
#         linear1 = self.linear1(embed)
#         linear1 = torch.tanh(linear1.contiguous().view(-1, linear1.size(-1))).view(linear1.shape)
#         linear1 = self.dropout(linear1)
        
#         # Second hidden layer (added)
#         linear2 = self.linear2(linear1)
#         linear2 = torch.tanh(linear2)
#         linear2 = self.dropout(linear2)
        
#         save_embeddings_to_text_file(linear2, self.embed_path)
#         #need to change the file name to create embeddings after every epoch
        
#         # Output layer
#         logits = self.linear3(linear2)
#         return logits
