from collections import Counter
import numpy as np
from string import punctuation


class Preprocess:
    def __init__(self):
        self.x = "boo"

    def get_words(self, data):
        """
        removes punctuation, lines and space to return a list of words
        Args:
            data (str): The data from the file
        Returns:
            list: a list of words
        """
        # get rid of punctuation
        data = data.lower()
        all_text = "".join([c for c in data if c not in punctuation])

        # slit by new lines and spaces
        data_split = all_text.split("\n")
        all_text = " ".join(data_split)

        # create a list of words
        words = all_text.split()

        return words, data_split

    def encode(self, words):
        """
        Maps words to integers
        Args:
            words (list): list of words
        Returns:
            Dict: a dictionary consisting of words and their integer mappings
        """
        # get counts of all words
        counts = Counter(words)

        # sort these words according to the word count
        vocab = sorted(counts, key=counts.get, reverse=True)

        # assign each word an integer
        vocab_to_int = {word: ii for ii, word in enumerate(vocab, 1)}

        return vocab_to_int

    def remove_outliers(self, data, labels):
        """
        removes reviews that are too small (empty)
        Args:
            data (list): list of reviews
            labels (ndarray): numpy array of labels in integer form

        Returns:
            list: list of reviews
            ndarray: numpy array of labels
        """
        data_size = len(data)

        non_zero_index = [ii for ii, review in enumerate(data) if len(review) != 0]
        data = [data[ii] for ii in non_zero_index]
        labels = np.array([labels[ii] for ii in non_zero_index])

        print("Number of reviews removed:"+str(data_size - len(data)))

        return data, labels

    def pad_features(self, data, seq_length):
        """
        Return features of review_ints, where each review is padded with 0's
        or truncated to the input seq_length
        Args:
            data(list): encoded integer list of reviews
            seq_length(int): threshold limit of each review

        Returns:
            feature_matrix(ndarray): feature matrix of review integers

        """
        # create an empty matrix
        feature_matrix = np.zeros((len(data), seq_length), dtype=int)

        # for each review
        for i, row in enumerate(data):
            feature_matrix[i, -len(row):] = np.array(row)[:seq_length]

        return feature_matrix

    def tokenize_review(test_review, vocab_to_int):
        """
        tokenize an individual review
        """
        test_review = test_review.lower()  # lowercase
        # get rid of punctuation
        test_text = ''.join([c for c in test_review if c not in punctuation])

        # splitting by spaces
        test_words = test_text.split()

        # tokens
        test_ints = []
        test_ints.append([vocab_to_int[word] for word in test_words])

        return test_ints

