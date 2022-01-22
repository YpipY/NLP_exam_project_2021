import torch
from torch import nn


class NNModel(nn.Module):
    """
    subclass of the nn.module class form PyTorch, used for making neural networks
    """
    def __init__(self, layers, embedding_layer: nn.Embedding, p: float):
        """
        initialize a model

        :param layers: list if int for the number of notes in each layer
        """
        # this is so hack I know, but it was the only way I could think if to have layers defined in the function
        super().__init__()

        # maps each token to an embedding_dim vector using our word embeddings
        self.embedding = embedding_layer

        self.embedding_size = embedding_layer.weight.shape[1]

        # if it is a one layer model
        if layers is None:
            self.linear0 = nn.Linear(self.embedding_size, 1)
        # or a multi layer model
        else:
            # input layer
            self.linear0 = nn.Linear(self.embedding_size, layers[0])
            i = 1
            # middle layers
            while i is not len(layers):
                setattr(self, 'linear%s' % i, nn.Linear(layers[i-1], layers[i]))
                i += 1
            # output layer
            setattr(self, 'linear%s' % i, nn.Linear(layers[i-1], 1))

        self.drop_layer = nn.Dropout(p=p)

    def forward(self, X):
        """
        make prediction using correct model. Uses linear regression and sigmoid

        :param X: some x values corresponding to number of inputs in the first layer
        :return: y value prediction
        """
        # apply the embedding layer that maps each token to its embedding
        x = self.embedding(X)  # dim: batch_size x batch_max_len x embedding_dim

        # for all layers
        i = 0
        while hasattr(self, "linear%s" % i):
            # make predictions
            x = getattr(self, "linear%s" % i)(x)  # make predictions
            x = torch.sigmoid(x)  # run trough sigmoid
            i += 1

        x = self.drop_layer(x)

        x = torch.mean(x, 1)

        return x
