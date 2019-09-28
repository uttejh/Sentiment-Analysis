from string import punctuation
from collections import Counter


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

