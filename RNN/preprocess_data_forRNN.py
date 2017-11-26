#!/usr/bin/env python

import pickle
import argparse
import email
import re
import os


class Emails():
    """Emails class
    """
    def __init__(self):
        """Constructor
        """
        pass

    def clean_text(self, text):
        """Clean up the text and return it
        """
        text = text.rstrip()
        text = re.sub(r"[^a-zA-Z]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()

        return text

    def get_text_from_email(self, email_message):
        """Get the text contents from the email message
        """
        text = []
        for section in email_message.walk():
            if section.get_content_type() == "text/plain":
                text.append(section.get_payload())

        return "".join(text)

    def parse_emails(self, data_dir):
        """Parse and return the emails and return them a plain text
        """
        emails_text = []
        emails_label = []
        names = ["kitchen-l/", "farmer-d/", "beck-s/", "kaminski-v/", "sanders-r/", "lokay-m/", "williams-w3/"]
        for name in names:
            for filename in os.listdir(os.path.join(data_dir, name)):
                with open(os.path.join(data_dir, name, filename), "r", encoding="latin-1") as fp:
                    email_message = email.message_from_file(fp)
                    text = self.get_text_from_email(email_message)
                    emails_text.append(self.clean_text(text).split())
                    emails_label.append(name)

        return emails_text, emails_label


class Dataset():
    """Dataset class
    """
    def __init__(self):
        """Constructor
        """
        pass

    def convert_label_onehot(self, email_label):
        """Convert categorical label to one-hot representation
        """
        label_list = ["kitchen-l", "farmer-d", "beck-s", "kaminski-v", "sanders-r", "lokay-m", "williams-w3"]
        one_hot = [0] * len(label_list)
        for idx, label in enumerate(label_list):
            if label == email_label:
                one_hot[idx] == 1

        return one_hot

    def make_vocab(self, emails_text):
        """Make word to index mapping
        """
        # compute the frequency of occurrence of all words in the corpus
        word_freq = {}
        for text in emails_text:
            for word in text:
                if word not in word_freq:
                    word_freq[word] = 1
                else:
                    word_freq[word] += 1

        # make a list of all words whose occurrence is > 100
        word_list = [k for k, v in word_freq.items() if v >= 100]
        word_list.append("UNK")

        # make word-index mapping
        word_index_mapping = {}
        for idx, word in enumerate(word_list):
            word_index_mapping[word] = idx

        return word_index_mapping

    def make_dataset(self, emails_text, emails_labels, word_index_mapping=None):
        """Make the dataset
        """
        if not word_index_mapping:
            word_index_mapping = self.make_vocab(emails_text)

        emails = [[word_index_mapping[word] if word in word_index_mapping else word_index_mapping["UNK"]
                   for word in text] for text in emails_text]
        labels = [self.convert_label_onehot(label) for label in emails_labels]

        return emails, labels, word_index_mapping


def run_main():
    """Main
    """
    # setup a command line parser
    parser = argparse.ArgumentParser("Preprocess email data for author classification using word embeddings + RNNs")

    # specify the command line arguments
    parser.add_argument("--data_dir", help="The root dir containing the data", required=True)

    # parse and get the command line arguments
    args = parser.parse_args()

    data_dir = args.data_dir

    # parse and get the email text and labels from data
    emails = Emails()
    train_emails, train_labels = emails.parse_emails(os.path.join(data_dir, "enron_email_database", "train"))
    test_emails, test_labels = emails.parse_emails(os.path.join(data_dir, "enron_email_database", "test"))

    # make the dataset
    dataset = Dataset()
    trainX, trainY, word2index = dataset.make_dataset(train_emails, train_labels)
    testX, testY, _ = dataset.make_dataset(test_emails, test_labels, word2index)

    # save to disk
    data_dir = os.path.join(data_dir, "RNN_data")
    if not os.path.isdir(data_dir):
        os.makedirs(data_dir)

    pickle.dump(trainX, open(os.path.join(data_dir, "train_emails.pkl"), "wb"))
    pickle.dump(trainY, open(os.path.join(data_dir, "train_labels.pkl"), "wb"))
    pickle.dump(testX, open(os.path.join(data_dir, "test_emails.pkl"), "wb"))
    pickle.dump(testY, open(os.path.join(data_dir, "test_labels.pkl"), "wb"))
    pickle.dump(word2index, open(os.path.join(data_dir, "word_index_mapping.pkl"), "wb"))


if __name__ == "__main__":
    run_main()
