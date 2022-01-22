import numpy as np
import torch
from torch import nn
from transformers import BertModel


class BERTModel(nn.Module):
    """
    subclass of the nn.module class form PyTorch, used for making neural networks
    """
    def __init__(self, layers, p: float):
        """
        initialize a model

        :param layers: list if int for the number of notes in each layer
        """
        # this is so hack I know, but it was the only way I could think if to have layers defined in the function
        super().__init__()

        # maps each token to an embedding_dim vector using our word embeddings
        self.bert = BertModel.from_pretrained('bert-base-uncased')


        # if it is a one layer model
        if layers is None:
            self.linear0 = nn.Linear(768, 1)
        # or a multi layer model
        else:
            # input layer
            self.linear0 = nn.Linear(768, layers[0])
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
        input_ids = np.zeros((len(X), 128))
        attention_mask = np.zeros((len(X), 128))
        for i in range(len(X)):
            input_ids[i][:128] = X[i]['input_ids']
            attention_mask[i][:128] = X[i]['attention_mask']
        # convert to torch LongTensors (integers)
        input_ids, attention_mask = torch.LongTensor(input_ids).to("cuda"), torch.LongTensor(attention_mask).to("cuda")

        _, x = self.bert(input_ids=input_ids, attention_mask=attention_mask, return_dict=False)

        # for all layers
        i = 0
        while hasattr(self, "linear%s" % i):
            # make predictions
            x = getattr(self, "linear%s" % i)(x)  # make predictions
            x = torch.sigmoid(x)  # run trough sigmoid
            i += 1

        x = self.drop_layer(x)

        return x
