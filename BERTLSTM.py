import numpy as np
import torch
from torch import nn
from transformers import BertModel


class BERTLSTM(nn.Module):
    """
    """
    def __init__(
        self, output_dim: int, hidden_dim_size: int, p: float, embedding_layer='bert-base-uncased'
    ):
        super().__init__()

        # maps each token to an embedding_dim vector using our word embeddings
        self.bert = BertModel.from_pretrained(embedding_layer)

        # the LSTM takes an embedded sentence
        self.lstm = nn.LSTM(768, hidden_dim_size, batch_first=True)#dropout=p

        # fc (fully connected) layer transforms the LSTM-output to give the final output layer
        self.fc = nn.Linear(hidden_dim_size, output_dim)

        self.drop_layer = nn.Dropout(p=p)

    def forward(self, X):
        # apply the embedding layer that maps each token to its embedding
        input_ids = np.zeros((len(X), 128))
        attention_mask = np.zeros((len(X), 128))
        for i in range(len(X)):
            input_ids[i][:128] = X[i]['input_ids']
            attention_mask[i][:128] = X[i]['attention_mask']
        # convert to torch LongTensors (integers)
        input_ids, attention_mask = torch.LongTensor(input_ids).to("cuda"), torch.LongTensor(attention_mask).to("cuda")

        x, _ = self.bert(input_ids=input_ids, attention_mask=attention_mask, return_dict=False)

        # run the LSTM along the sentences of length batch_max_len
        x, _ = self.lstm(x)  # dim: batch_size x batch_max_len x lstm_hidden_dim

        # extract last hidden state
        x = x[:, x.shape[1]-1, :]

        # apply the fully connected layer and obtain the output for each token
        x = self.fc(x)

        # apply dropout layer
        x = self.drop_layer(x)

        return torch.sigmoid(x)