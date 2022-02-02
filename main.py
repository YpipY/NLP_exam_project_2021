"""

"""
import collections
import csv
import os
import random
import numpy as np
import string
from typing import Iterable, List, Tuple

import gensim.downloader as api
from gensim.models.keyedvectors import KeyedVectors
from transformers import BertTokenizer, BertForSequenceClassification, TrainingArguments, Trainer

from sklearn import metrics

import torch
from torch import nn

import spacy

from util import batch
from LSTM import SentenceLSTM
from NN import NNModel
from BERTmodel import BERTModel
from dataset import Dataset
from BERTLSTM import BERTLSTM


def file_loader(d: string):
    """Loads the corpus data

    :param d: directory of the corpus data
    :return: a list of dicts
    """
    corpus = []
    reader = csv.DictReader(open(d))
    for row in reader:
        corpus.append(row)
    return corpus


def gensim_to_torch_embedding(gensim_wv: KeyedVectors) -> Tuple[nn.Embedding, dict]:
    """Convert a gensim KeyedVectors object into a pytorch Embedding layer.

    Args:
        gensim_wv: gensim KeyedVectors object

    Returns:
        torch.nn.Embedding: torch Embedding layer
        dict: dictionary of word to index mapping
    """
    embedding_size = gensim_wv.vectors.shape[1]

    # create unknown and padding embedding
    unk_emb = np.mean(gensim_wv.vectors, axis=0).reshape((1, embedding_size))
    pad_emb = np.zeros((1, gensim_wv.vectors.shape[1]))

    # add the new embedding
    embeddings = np.vstack([gensim_wv.vectors, unk_emb, pad_emb])

    weights = torch.FloatTensor(embeddings)

    emb_layer = nn.Embedding.from_pretrained(embeddings=weights, padding_idx=-1)

    # creating vocabulary
    vocab = gensim_wv.key_to_index
    vocab["UNK"] = weights.shape[0] - 2
    vocab["PAD"] = emb_layer.padding_idx

    return emb_layer, vocab


def tokenizer(spacymodel, data):
    """Runs the given data through a Spacy tokenizer

    :param spacymodel: a spacy model
    :param data: list of strings
    :return: a list of Spacy docs
    """
    # running the training though the tokenizer
    sen = []
    c = 0  # counter
    print("Sentences are running though the tokenizer (can take some time)")
    # for all sentences
    for sentence in data:
        # run trough the nlp tokenizer
        sen.append(spacymodel(sentence))
        # print to see the progress
        if (c % round(len(data) / 100)) == 0:
            print(repr(round((c / (len(data))) * 100)) + " %")
        c += 1  # counter
    return sen


def select_lemma(t, p=None, keep=True):
    """Selects (or removes) lemmas with given port of speech tag

    :param t: a spacy doc object
    :param p: a list of part of speech
              ADJ: adjective ADP: adposition ADV: adverb AUX: auxiliary CCONJ: coordinating conjunction DET: determiner
              INTJ: interjection NOUN: noun NUM: numeral PART: particle PRON: pronoun PROPN: proper noun
              PUNCT: punctuation SCONJ: subordinating conjunction SYM: symbol VERB: verb X: other
              (note: X, SYM and FUNCT are not wards)
    :param keep: should words with the given part of speech be kept. Default = True
    :return: list of strings of lemmas
    """
    # selecting lemmas only specific part of speech
    if p is None:
        p = []
    lem = []
    # for all tokens in the text
    for token in t:
        # see if words with the given pos should be kept
        if keep:
            # check if this token is classified as given pos
            if token.pos_ in p:
                # save the lemma
                lem.append(token.lemma_)
        else:
            # check if this token is NOT classified as given pos
            if token.pos_ not in p:
                # save the lemma
                lem.append(token.lemma_)
    return lem


def tokens_to_idx(tokens: List[str], vocab):
    """Convert tokens into the indexes of the embedding model

    Args:
        tokens (List[str]): A list of lists of tokens.
        vocab: Dictionary of word to index mapping

    Returns:
        (List[int]): A list of word to index mapping
    """
    return [vocab.get(t.lower(), vocab["UNK"]) for t in tokens]


def prepare_batch(tokens: List[List[str]], labels: List[List[int]], vocab: dict) -> Tuple[torch.Tensor, torch.Tensor]:
    """Prepare a batch of data for training.

    Args:
        tokens (List[List[str]]): A list of lists of tokens.
        labels (List[List[int]]): A list of lists of labels.
        vocab (dict): Dictionary of word to index mapping

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple of tensors containing the tokens and labels.
    """
    # convert tokens into the indexes of the embedding model
    batch_tok_idx = [tokens_to_idx(sent, vocab) for sent in tokens]
    # number of rows
    batch_size = len(tokens)

    # number of columns = longest sentence in the batch
    batch_max_len = max([len(s) for s in batch_tok_idx])

    # prepare a numpy array with the data, initializing the data with 'PAD' and all labels with -1; initializing labels
    # to -1 differentiates tokens with tags from 'PAD' tokens
    batch_input = vocab["PAD"] * np.ones((batch_size, batch_max_len))
    batch_labels = -1 * np.ones((batch_size, batch_max_len))

    # copy the data to the numpy array
    for i in range(batch_size):
        tok_idx = batch_tok_idx[i]
        tags = labels[i]
        size = len(tok_idx)

        batch_input[i][:size] = tok_idx
        batch_labels[i][:size] = tags

    # convert to torch LongTensors (integers)
    batch_input, batch_labels = torch.LongTensor(batch_input), torch.LongTensor(batch_labels)
    return batch_input, batch_labels


def train_model(model, tokens, tags, optimizer, criterion, batch_size, vocab, val_tokens=None, val_tags=None,
                patience=0, epochs=1000, save_name='model.okl', test_lemmas=None, test_label=None):
    """Trains a given model and saves the best model

    Args:
        model: A torch recurred neural network model
        tokens (List[List[int]]): List of list of integers corresponding to indexes in the embedding model
        tags (List[List[int]]): List of list of integers corresponding to togs
        optimizer: A torch optimizer
        criterion: A torch loss function
        val_tokens (List[List[int]]): List of list of integers corresponding to indexes in the embedding
                                                model. Used for validation
        val_tags (List[List[int]]): List of list of integers corresponding to togs. Used for validation
        batch_size (int): Size of batches
        vocab (dict): Dictionary of word to index mapping
        patience (float): Patience for early stopping
        epochs (int): Number of epochs
        save_name (str): Name of the file the model gets saved to
    """
    # initialize variables for early stopping
    best_average_loss = None
    epochs_since_save = 0
    # training loop
    for epoch in range(epochs):
        sum_loss = 0
        n_batches = 0
        loss = 0
        batches_tokens = batch(tokens, batch_size)
        batches_tags = batch(tags, batch_size)
        model.train()
        for cur_tokens in batches_tokens:
            # get current batch
            cur_tags = next(batches_tags)
            # prepare current batch
            cur_tokens, _ = prepare_batch(cur_tokens, cur_tags, vocab)
            cur_tags = torch.FloatTensor(cur_tags)

            # forward pass
            X = cur_tokens
            y = model(X)
            y = torch.squeeze(y, 1)

            # backwards pass
            loss = criterion(y, cur_tags)  # calculate loss
            loss.backward()  # calculate the gradient
            optimizer.step()  # step in the direction that minimizes loss
            optimizer.zero_grad()  # reset gradient

            # sum up loss
            sum_loss += loss.item()
            # count number of batches
            n_batches += 1
        train_loss = sum_loss / n_batches
        # periodically calculate loss on validation set
        model.eval()
        predict(model, test_lemmas, test_label, batch_size, vocab)
        if patience != 0 and val_tokens is not None and val_tags is not None:
            if epoch % 1 == 0:  # every epoch
                sum_loss = 0
                n_batches = 0
                loss = 0
                batches_tokens = batch(val_tokens, batch_size)
                batches_tags = batch(val_tags, batch_size)
                for cur_tokens in batches_tokens:
                    # get current batch
                    cur_tags = next(batches_tags)
                    # prepare current batch
                    cur_tokens, _ = prepare_batch(cur_tokens, cur_tags, vocab)
                    cur_tags = torch.FloatTensor(cur_tags)

                    # forward pass
                    X = cur_tokens
                    y = model(X)
                    y = torch.squeeze(y, 1)

                    # backwards pass
                    loss = criterion(y, cur_tags)  # calculate loss
                    #loss.backward()  # calculate the gradient

                    # sum up loss
                    sum_loss += loss.item()
                    # count number of batches
                    n_batches += 1
                # calculate average loss
                average_loss = (sum_loss / n_batches)
                # print the average loss to see how the model is doing
                print(f'epoch: {epoch + 1}, train loss = {train_loss:.4f}, eval loss = {average_loss:.4f}, patience: {epochs_since_save}')
                # save the model if it is the best so far
                if best_average_loss is None or average_loss < best_average_loss:
                    best_average_loss = average_loss
                    epochs_since_save = 0
                    torch.save(model, save_name)
                else:
                    epochs_since_save += 1
                # stops training if no loss increase has occurred over patience number of validations
                if epochs_since_save == patience:
                    print("Stopped due to reaching patience threshold")
                    return
        # if no validation set
        elif patience != 0:
            # calculate average loss
            average_loss = (sum_loss / n_batches)
            # print the average loss to see how the model is doing
            print(f'epoch: {epoch + 1}, train loss = {train_loss:.4f}, eval loss = {average_loss:.4f}, patience: {epochs_since_save}')
            # save the model if it is the best so far
            if best_average_loss is None or average_loss < best_average_loss:
                best_average_loss = average_loss
                epochs_since_save = 0
                torch.save(model, save_name)
            else:
                epochs_since_save += 1
            # stops training if no loss increase has occurred over patience number of validations
            if epochs_since_save == patience:
                print("Stopped due to reaching patience threshold")
                return
        else:
            # print the average loss to see how the model is doing
            torch.save(model, save_name)
            print(f"epoch: {epoch+1} of {epochs}, loss = {loss.item():.4f}")


def train_model_bert(model, tokens, tags, optimizer, criterion, batch_size, val_tokens=None, val_tags=None,
                patience=0, epochs=1000, save_name='model.okl', test_lemmas=None, test_label=None):
    """Trains a given model and saves the best model. For BERT models

    Args:
        model: A torch recurred neural network model
        tokens (List[List[int]]): List of list of integers corresponding to indexes in the embedding model
        tags (List[List[int]]): List of list of integers corresponding to togs
        optimizer: A torch optimizer
        criterion: A torch loss function
        val_tokens (List[List[int]]): List of list of integers corresponding to indexes in the embedding
                                                model. Used for validation
        val_tags (List[List[int]]): List of list of integers corresponding to togs. Used for validation
        batch_size (int): Size of batches
        patience (float): Patience for early stopping
        epochs (int): Number of epochs
        save_name (str): Name of the file the model gets saved to
    """
    # initialize variables for early stopping
    best_average_loss = None
    epochs_since_save = 0
    # training loop
    for epoch in range(epochs):
        sum_loss = 0
        n_batches = 0
        loss = 0
        batches_tokens = batch(tokens, batch_size)
        batches_tags = batch(tags, batch_size)
        model.train()
        for cur_tokens in batches_tokens:
            # get current batch
            cur_tags = next(batches_tags)
            cur_tags = torch.FloatTensor(cur_tags).to("cuda")

            # forward pass
            X = cur_tokens
            y = model(X)
            y = torch.squeeze(y, 1)

            # backwards pass
            loss = criterion(y, cur_tags)  # calculate loss
            loss.backward()  # calculate the gradient
            optimizer.step()  # step in the direction that minimizes loss
            optimizer.zero_grad()  # reset gradient

            # sum up loss
            sum_loss += loss.item()
            # count number of batches
            n_batches += 1
        train_loss = sum_loss / n_batches
        # periodically calculate loss on validation set
        model.eval()
        predict_bert(model, test_lemmas, test_label, batch_size)
        if patience != 0 and val_tokens is not None and val_tags is not None:
            if epoch % 1 == 0:  # every epoch
                sum_loss = 0
                n_batches = 0
                loss = 0
                batches_tokens = batch(val_tokens, batch_size)
                batches_tags = batch(val_tags, batch_size)
                for cur_tokens in batches_tokens:
                    # get current batch
                    cur_tags = next(batches_tags)
                    cur_tags = torch.FloatTensor(cur_tags).to("cuda")

                    # forward pass
                    X = cur_tokens
                    y = model(X)
                    y = torch.squeeze(y, 1)

                    # backwards pass
                    loss = criterion(y, cur_tags)  # calculate loss
                    #loss.backward()  # calculate the gradient

                    # sum up loss
                    sum_loss += loss.item()
                    # count number of batches
                    n_batches += 1
                # calculate average loss
                average_loss = (sum_loss / n_batches)
                # print the average loss to see how the model is doing
                print(f'epoch: {epoch + 1}, train loss = {train_loss:.4f}, eval loss = {average_loss:.4f}, patience: {epochs_since_save}')
                # save the model if it is the best so far
                if best_average_loss is None or average_loss < best_average_loss:
                    best_average_loss = average_loss
                    epochs_since_save = 0
                    torch.save(model, save_name)
                else:
                    epochs_since_save += 1
                # stops training if no loss increase has occurred over patience number of validations
                if epochs_since_save == patience:
                    print("Stopped due to reaching patience threshold")
                    return
        # if no validation set
        elif patience != 0:
            # calculate average loss
            average_loss = (sum_loss / n_batches)
            # print the average loss to see how the model is doing
            print(f'epoch: {epoch + 1}, train loss = {train_loss:.4f}, eval loss = {average_loss:.4f}, patience: {epochs_since_save}')
            # save the model if it is the best so far
            if best_average_loss is None or average_loss < best_average_loss:
                best_average_loss = average_loss
                epochs_since_save = 0
                torch.save(model, save_name)
            else:
                epochs_since_save += 1
            # stops training if no loss increase has occurred over patience number of validations
            if epochs_since_save == patience:
                print("Stopped due to reaching patience threshold")
                return
        else:
            # print the average loss to see how the model is doing
            torch.save(model, save_name)
            print(f"epoch: {epoch+1} of {epochs}, loss = {loss.item():.4f}")


def predict(model, tokens, tags, batch_size, vocab):
    """Makes predictions and prints accuracy, balanced accuracy and F1 micro score

        Args:
            model: A torch recurred neural network model
            tokens (List[List[int]]): List of list of integers corresponding to indexes in the embedding model
            tags (List[List[int]]): List of list of integers corresponding to togs
            batch_size (int): Size of batches
            vocab (dict): Dictionary of word to index mapping
        """
    # make predictions
    batches_tokens = batch(tokens, batch_size)
    batches_tags = batch(tags, batch_size)
    y_true = torch.empty(0)
    y_pred = torch.empty(0)
    model.eval()
    for cur_tokens in batches_tokens:
        # get current batch
        cur_tags = next(batches_tags)
        # prepare current batch
        cur_tokens, _ = prepare_batch(cur_tokens, cur_tags, vocab)

        cur_tags = torch.FloatTensor(cur_tags)
        y_true = torch.cat((y_true, cur_tags), 0)

        # forward pass
        X = cur_tokens
        with torch.no_grad():
            y = model(X)
        y = torch.squeeze(y, 1)
        y_pred = torch.cat((y_pred, y), 0)
    y_pred = torch.round(y_pred)
    y_pred = y_pred.detach()
    y_true = y_true.detach()

    # calculate matrices
    accuracy = metrics.accuracy_score(y_true=y_true, y_pred=y_pred)
    balancedaccuracy = metrics.balanced_accuracy_score(y_true=y_true, y_pred=y_pred)
    recall = metrics.recall_score(y_true=y_true, y_pred=y_pred)
    precision = metrics.precision_score(y_true=y_true, y_pred=y_pred)
    f1 = metrics.f1_score(y_true=y_true, y_pred=y_pred, average='micro')

    print(f"accuracy: {accuracy}, balanced accuracy: {balancedaccuracy}, precision: {precision}, recall: {recall}, "
          f"f1: {f1}")

    # calculate and print accuracy
    #print("Accuracy")
    #print(metrics.accuracy_score(y_true, y_pred))
    # calculate and print balanced accuracy
    #print("Balanced accuracy")
    #print(metrics.balanced_accuracy_score(y_true, y_pred))
    # calculate and print F1 scores
    #print("F1 score")
    #print(metrics.f1_score(y_true, y_pred, average='micro'))


def predict_bert(model, tokens, tags, batch_size):
    """Makes predictions and prints accuracy, balanced accuracy and F1 micro score. For BERT models

        Args:
            model: A torch recurred neural network model
            tokens (List[List[int]]): List of list of integers corresponding to indexes in the embedding model
            tags (List[List[int]]): List of list of integers corresponding to togs
            batch_size (int): Size of batches
        """
    # make predictions
    batches_tokens = batch(tokens, batch_size)
    batches_tags = batch(tags, batch_size)
    y_true = torch.empty(0)
    y_pred = torch.empty(0)
    model.eval()
    for cur_tokens in batches_tokens:
        # get current batch
        cur_tags = next(batches_tags)

        cur_tags = torch.FloatTensor(cur_tags)
        y_true = torch.cat((y_true, cur_tags), 0)
        # forward pass
        X = cur_tokens
        with torch.no_grad():
            y = model(X)
        y = torch.squeeze(y, 1)
        y_pred = torch.cat((y_pred, y.to("cpu")), 0)
    y_pred = torch.round(y_pred)
    y_pred = y_pred.detach()
    y_true = y_true.detach()

    # calculate matrices
    accuracy = metrics.accuracy_score(y_true=y_true, y_pred=y_pred)
    balancedaccuracy = metrics.balanced_accuracy_score(y_true=y_true, y_pred=y_pred)
    recall = metrics.recall_score(y_true=y_true, y_pred=y_pred)
    precision = metrics.precision_score(y_true=y_true, y_pred=y_pred)
    f1 = metrics.f1_score(y_true=y_true, y_pred=y_pred, average='micro')

    print(f"accuracy: {accuracy}, balanced accuracy: {balancedaccuracy}, precision: {precision}, recall: {recall}, "
          f"f1: {f1}")

    # calculate and print accuracy
    #print("Accuracy")
    #print(metrics.accuracy_score(y_true, y_pred))
    # calculate and print balanced accuracy
    #print("Balanced accuracy")
    #print(metrics.balanced_accuracy_score(y_true, y_pred))
    # calculate and print F1 scores
    #print("F1 score")
    #print(metrics.f1_score(y_true, y_pred, average='micro'))


def compute_metrics(p):
    pred, labels = p
    pred = np.argmax(pred, axis=1)

    accuracy = metrics.accuracy_score(y_true=labels, y_pred=pred)
    balancedaccuracy = metrics.balanced_accuracy_score(y_true=labels, y_pred=pred)
    recall = metrics.recall_score(y_true=labels, y_pred=pred)
    precision = metrics.precision_score(y_true=labels, y_pred=pred)
    f1 = metrics.f1_score(y_true=labels, y_pred=pred, average='micro')

    return {"accuracy": accuracy, "balancedaccuracy": balancedaccuracy, "precision": precision, "recall": recall, "f1": f1}


class Main:
    runlstm = False
    runnn = False
    runlr = False
    useBERT = True
    runBERTmodel = False
    BERTmodeltest = False
    # interesting models: none
    # Notes:
    # * lstm is too large it is over-fitting
    # * model just will not generalize
    hidden_layer_size = 64
    batch_size = 16
    learning_rate = 0.01
    patience = 100
    epochs = 10000
    dropout_p = 0.1
    L2 = 0.01
    save_name = "BERTpre.pkl"
    embedding_model = "bert-base-uncased-finetuned-rapport/checkpoint-9825"

    use_existing = True

    train = False

    # set a seed to make the results reproducible
    torch.manual_seed(451) #24
    np.random.seed(451)
    random.seed(451)

    # load data
    data = file_loader(os.getcwd() + '\\' + 'affcon_diplomacy_data.csv')

    # shuffle the data
    random.shuffle(data)

    # spilt data and get labels
    test_label = []
    for label in data[0:1000]:
        test_label.append(label["affcon_rapport"])
    test_label = list(map(int, test_label))

    val_label = []
    for label in data[1000:2000]:
        val_label.append(label["affcon_rapport"])
    val_label = list(map(int, val_label))

    train_label = []
    for label in data[2000:-1]:
        train_label.append(label["affcon_rapport"])
    train_label = list(map(int, train_label))

    test_lemmas = []
    val_lemmas = []
    train_lemmas = []

    if runBERTmodel:
        # load pretrained BERT model and BERT tokenizer
        model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

        # tokenize the texts
        dummy = []
        for label in data[1000:2000]:
            dummy.append(label["full_text"])
        val_lemmas = (tokenizer(dummy, padding=True, max_length=512, truncation=True,
                                    return_tensors="pt"))
        dummy = []
        for label in data[2000:-1]:
            dummy.append(label["full_text"])
        train_lemmas = (tokenizer(dummy, padding=True, max_length=512, truncation=True,
                                      return_tensors="pt"))

        # create the datasets
        train_data = Dataset(train_lemmas, train_label)
        val_data = Dataset(val_lemmas, val_label)
        test_data = Dataset(train_lemmas, test_label)


        # create trainer
        metric_name = "f1"
        model_name = 'bert-base-uncased'.split("/")[-1]
        args = TrainingArguments(
            f"{model_name}-finetuned-rapport",
            evaluation_strategy="epoch",
            save_strategy="epoch",
            learning_rate=learning_rate,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            num_train_epochs=5,
            weight_decay=L2,
            load_best_model_at_end=True,
            metric_for_best_model=metric_name,
        )

        trainer = Trainer(
            model,
            args,
            train_dataset=train_data,
            eval_dataset=val_data,
            tokenizer=tokenizer,
            compute_metrics=compute_metrics
        )

        trainer.train()

        trainer.evaluate()

    if BERTmodeltest:
        model_name = 'bert-base-uncased'.split("/")[-1]
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

        # tokenize the data
        dummy = []
        for label in data[0:1000]:
            dummy.append(label["full_text"])
        test_lemmas = (tokenizer(dummy, padding=True, max_length=512, truncation=True,
                                 return_tensors="pt"))

        # create the dataset
        test_dataset = Dataset(test_lemmas)

        # load trained model
        model_path = f"{model_name}-finetuned-rapport/checkpoint-9825"
        model = BertForSequenceClassification.from_pretrained(model_path, num_labels=2)

        # create test trainer
        test_trainer = Trainer(model)

        # make prediction
        raw_pred, _, _ = test_trainer.predict(test_dataset)

        # preprocess raw predictions
        y_pred = np.argmax(raw_pred, axis=1)
        y_true = np.asarray(test_label)

        accuracy = metrics.accuracy_score(y_true=y_true, y_pred=y_pred)
        balancedaccuracy = metrics.balanced_accuracy_score(y_true=y_true, y_pred=y_pred)
        recall = metrics.recall_score(y_true=y_true, y_pred=y_pred)
        precision = metrics.precision_score(y_true=y_true, y_pred=y_pred)
        f1 = metrics.f1_score(y_true=y_true, y_pred=y_pred)

        print(f"accuracy: {accuracy}, balanced accuracy: {balancedaccuracy}, precision: {precision}, recall: {recall}, "
              f"f1: {f1}")

    if useBERT:
        # load BERT tokenizer
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

        # tokenize the texts
        for label in data[0:1000]:
            test_lemmas.append(tokenizer(label["full_text"], padding='max_length', max_length=128, truncation=True,
                                         return_tensors="pt"))
        for label in data[1000:2000]:
            val_lemmas.append(tokenizer(label["full_text"], padding='max_length', max_length=128, truncation=True,
                                        return_tensors="pt"))
        for label in data[2000:-1]:
            train_lemmas.append(tokenizer(label["full_text"], padding='max_length', max_length=128, truncation=True,
                                          return_tensors="pt"))

    if runlstm or runnn or runlr:
        # spilt data and get spacy docs and lemmas
        # loading the spacy english large web model
        nlp = spacy.load("en_core_web_lg")

        test_spacy = []
        for label in data[0:1000]:
            test_spacy.append(label["full_text"])
        test_spacy = tokenizer(nlp, test_spacy)
        for sen in test_spacy:
            test_lemmas.append(select_lemma(sen, keep=False)) # ['VERB', 'NOUN', 'ADJ', 'ADV', 'PRON', 'PROPN']

        val_spacy = []
        for label in data[1000:2000]:
            val_spacy.append(label["full_text"])
        val_spacy = tokenizer(nlp, val_spacy)
        for sen in val_spacy:
            val_lemmas.append(select_lemma(sen, keep=False))

        train_spacy = []
        for label in data[2000:-1]:
            train_spacy.append(label["full_text"])
        train_spacy = tokenizer(nlp, train_spacy)
        for sen in train_spacy:
            train_lemmas.append(select_lemma(sen, keep=False))

        # load embeddings
        embeddings = api.load(embedding_model)

        # convert gensim word embedding to torch word embedding
        embedding_layer, vocab = gensim_to_torch_embedding(embeddings)

    # initializing the model
    num_classes = 1
    if runlstm:
        if use_existing:
            model = torch.load(save_name)
        else:
            model = SentenceLSTM(embedding_layer=embedding_layer, output_dim=num_classes, hidden_dim_size=hidden_layer_size,
                                 p=dropout_p)
    elif runnn:
        if use_existing:
            model = torch.load(save_name)
        else:
            model = NNModel(layers=[128, 64, 32], embedding_layer=embedding_layer, p=dropout_p)
    elif runlr:
        if use_existing:
            model = torch.load(save_name)
        else:
            model = NNModel(layers=None, embedding_layer=embedding_layer, p=dropout_p)
    elif useBERT:
        if use_existing:
            model = torch.load(save_name)
        else:
            model = BERTLSTM(output_dim=num_classes, hidden_dim_size=hidden_layer_size, p=dropout_p,
                             embedding_layer=embedding_model)

    # initializing optimizer and loss
    optimizer = torch.optim.AdamW(model.parameters(), learning_rate, weight_decay=L2)  # AdamW optimizer algorithm, amsgrad=True
    criterion = nn.BCELoss()  # BCE loss

    # training the model
    if train:
        if runlstm or runnn or runlr:
            train_model(model, train_lemmas, train_label, optimizer, criterion, batch_size, vocab, val_lemmas,
                        val_label, patience=patience, epochs=epochs, save_name=save_name, test_lemmas=test_lemmas, test_label=test_label)
        elif useBERT:
            model.cuda()
            criterion.cuda()
            train_model_bert(model, train_lemmas, train_label, optimizer, criterion, batch_size, val_lemmas,
                        val_label, patience=patience, epochs=epochs, save_name=save_name, test_lemmas=test_lemmas,
                        test_label=test_label)

    # loading the best model
    model = torch.load(save_name)

    if runlstm or runnn or runlr:
        print("train set predictions")
        predict(model, train_lemmas, train_label, batch_size, vocab)
        print("validation set predictions")
        predict(model, val_lemmas, val_label, batch_size, vocab)
        print("test set predictions")
        predict(model, test_lemmas, test_label, batch_size, vocab)
    elif useBERT:
        print("train set predictions")
        predict_bert(model, train_lemmas, train_label, batch_size)
        print("validation set predictions")
        predict_bert(model, val_lemmas, val_label, batch_size)
        print("test set predictions")
        predict_bert(model, test_lemmas, test_label, batch_size)
