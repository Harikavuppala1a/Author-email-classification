#!/usr/bin/env python

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

import argparse
import pickle
import os

import numpy as np


class DataFeed():
    """Data feed to the model
    """
    def __init__(self):
        """Constructor
        """
        pass

    def shuffle_data(self, emails, labels):
        """Shuffle the contents of emails and labels together (inplace)
        """
        idxs = np.arange(len(emails))
        np.random.shuffle(idxs)
        emails = emails[idxs]
        labels = labels[idxs]

        return emails, labels

    def yield_sequences(self, emails, labels, shuffle=True):
        """Yield pairs of sequences of emails and labels from data
        """
        assert len(emails) == len(labels)
        if shuffle:
            self.shuffle_data(emails, labels)

        # iterate through the entire dataset
        for idx in range(len(emails)):
            yield emails[idx: idx+1], labels[idx: idx+1]


class LSTMClassifier(nn.Module):
    """LSTM classifier model
    """
    def __init__(self, embedding_dim, vocab_size, LSTM_dim, num_classes):
        """Constructor
        """
        super(LSTMClassifier, self).__init__()

        # parameters of the model
        self.embedding_dim = embedding_dim
        self.vocab_size = vocab_size
        self.LSTM_dim = LSTM_dim
        self.num_classes = num_classes

        # the model
        self.embedding_layer = nn.Embedding(num_embeddings=self.vocab_size, embedding_dim=self.embedding_dim)
        self.lstm_layer = nn.LSTM(input_size=self.embedding_dim, hidden_size=LSTM_dim//2, bidirectional=True)
        self.output_layer = nn.Linear(in_features=LSTM_dim, out_features=num_classes)

        # hidden state of LSTM
        self.h0 = self.init_hidden()

    def init_hidden(self):
        """Initialize the LSTM hidden state
           dims of tensor are: (num_layers * num_directions, batch, hidden_size)
        """
        h0 = Variable(torch.zeros(2, 1, self.LSTM_dim))
        c0 = Variable(torch.zeros(2, 1, self.LSTM_dim))

        result = (h0.cuda(), c0.cuda()) if torch.cuda.is_available() else (h0, c0)

        return result

    def forward(self, x):
        """Forward pass of the model
        """
        x = self.word_embeddings(x)
        x, h = self.lstm(x, self.h0)        # outputs, hidden states
        # get the output from last hidden state of LSTM
        x = x[:, -1, :].squeeze()
        x = self.output_layer(x)

        return x


class Train():
    """Model train class
    """
    def __init__(self, classifier_model, num_epochs, learning_rate):
        """Constructor
        """
        # model to be trained
        self.model = classifier_model

        # number of epochs to train the model
        self.num_epochs = num_epochs

        # learning rate
        self.learning_rate = learning_rate

        # instantiate an instance of data feed
        self.data_feed = DataFeed()

    def init_optimizer_criterion(self):
        """Initialize the criterion and optimizer for training the model
        """
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate, betas=(0.9, 0.999), eps=1e-08,
                               weight_decay=0)
        criterion = nn.CrossEntropyLoss()

        return optimizer, criterion

    def train_model(self, train_emails, train_labels):
        """Train the model
        """
        self.model = self.model.cuda() if torch.cuda.is_available() else self.model
        self.model.train()
        train_loss = 0
        num_updates = 0
        optimizer, criterion = self.init_optimizer_criterion()

        for epoch in range(1, self.num_epochs+1):
            for email, label in self.data_feed(train_emails, train_labels):
                self.model.zero_grad()

                output = self.model(email)
                loss = criterion(output, label)

                loss.backward()
                optimizer.step()

                train_loss += loss.data[0]
                if num_updates == 1500:
                    print("Epoch: %d, Update: %d, Train loss: %f" % (epoch, num_updates, (train_loss/num_updates)))
                num_updates += 1


def run_main():
    """Main
    """
    parser = argparse.ArgumentParser(description="LSTM based email authorship classifier")

    parser.add_argument("--data_dir", help="dir containing the training and testing data", required=True)
    parser.add_argument("--embedding_dim", type=int, help="dimensions of the embeddings", required=True)
    parser.add_argument("--lstm_dim", type=int, help="size of the lstm layer", required=True)
    parser.add_argument("--num_epochs", type=int, help="number of epochs to train the model", required=True)
    parser.add_argument("--learning_rate", type=float, help="learning rate", required=True)

    args = parser.parse_args()

    # load the files
    train_emails = pickle.load(open(os.path.join(args.data_dir, "train_emails.pkl"), "rb"))
    train_labels = pickle.load(open(os.path.join(args.data_dir, "train_labels.pkl"), "rb"))
    test_emails = pickle.load(open(os.path.join(args.data_dir, "test_emails.pkl"), "rb"))
    test_labels = pickle.load(open(os.path.join(args.data_dir, "test_labels.pkl"), "rb"))
    word_index_mapping = pickle.load(open(os.path.join(args.data_dir, "word_index_mapping.pkl"), "rb"))

    vocab_size = len(word_index_mapping)
    num_classes = 7

    # instantiate an instance of model
    classifier_model = LSTMClassifier(embedding_dim=args.embedding_dim, vocab_size=vocab_size, LSTM_dim=args.lstm_dim,
                                      num_classes=num_classes)

    # instantiate an instance of training class
    model_train = Train(classifier_model, args.num_epochs, args.learning_rate)
    model_train.train_model(train_emails, train_labels)


if __name__ == "__main__":
    run_main()
