"""
Sentiment Analysis with an RNN
"""
from Preprocess import Preprocess
import numpy as np
from Train import Train


def load_data(filepath):
    """
    reads contents of a file
    Args:
        filepath (str): The file location of the data

    Returns:
        str: The contents of the file
    """
    with open(filepath, "r") as f:
        return f.read()


def main():
    """

    :rtype: object
    """

    # get reviews
    reviews = load_data("data/reviews.txt")
    # get labels
    labels = load_data("data/labels.txt")

    # visualize data
    print(reviews[:100])
    print(labels[:10])

    # class object
    preprocess = Preprocess()

    # get words
    words, reviews_split = preprocess.get_words(reviews)

    # map words to integers
    vocab_to_int = preprocess.encode(words)

    # tokenize reviews
    review_ints = []
    for review in reviews_split:
        review_ints.append([vocab_to_int[word] for word in review.split()])

    # encode labels
    labels_split = labels.split("\n")
    encoded_labels = np.array([1 if label == "positive" else 0 for label in labels_split])

    # visualize review integers
    print('Unique words: ', len(vocab_to_int))
    print(review_ints[:1])
    print(encoded_labels[:10])

    # remove outliers
    review_ints, encoded_labels = preprocess.remove_outliers(review_ints, encoded_labels)

    # pad sequences with 0
    seq_length = 200

    features = preprocess.pad_features(review_ints, seq_length)

    # test statements
    assert len(features) == len(review_ints), "Your features should have as many rows as reviews."
    assert len(features[0]) == seq_length, "Each feature row should contain seq_length values."

    # print first 10 values of the first 30 batches
    print(features[:30, :10])

    # initiate train TODO: reformat this comment
    train = Train(features, encoded_labels)

    hyperparameters = {"vocab_to_int": vocab_to_int, "embedding_dim": 400,
                       "hidden_dim": 256, "output_size": 1, "n_layers": 2,
                       "lr": 0.001, "epochs": 4}
    train.train_model(**hyperparameters)


if __name__ == "__main__":
    main()
