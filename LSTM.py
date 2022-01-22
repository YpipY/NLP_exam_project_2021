import torch
from torch import nn


class SentenceLSTM(nn.Module):
    """
    """
    def __init__(
        self, output_dim: int, embedding_layer: nn.Embedding, hidden_dim_size: int, p: float
    ):
        super().__init__()

        # maps each token to an embedding_dim vector using our word embeddings
        self.embedding = embedding_layer

        self.embedding_size = embedding_layer.weight.shape[1]

        # the LSTM takes an embedded sentence
        self.lstm = nn.LSTM(self.embedding_size, hidden_dim_size, batch_first=True)#dropout=p

        # fc (fully connected) layer transforms the LSTM-output to give the final output layer
        self.fc = nn.Linear(hidden_dim_size, output_dim)

        self.drop_layer = nn.Dropout(p=p)

    def forward(self, X):
        # apply the embedding layer that maps each token to its embedding
        x = self.embedding(X)  # dim: batch_size x batch_max_len x embedding_dim

        # run the LSTM along the sentences of length batch_max_len
        x, _ = self.lstm(x)  # dim: batch_size x batch_max_len x lstm_hidden_dim

        # extract last hidden state
        x = x[:, x.shape[1]-1, :]

        # apply the fully connected layer and obtain the output for each token
        x = self.fc(x)

        # apply dropout layer
        x = self.drop_layer(x)

        return torch.sigmoid(x)
