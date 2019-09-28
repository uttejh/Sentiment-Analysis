"""
Sentiment Analysis with an RNN
"""
from Preprocess import Preprocess


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

    # visualize review integers
    print('Unique words: ', len(vocab_to_int))
    print(review_ints[:1])

    # TODO: Remove outliers
    # TODO: pad sequences with 0

if __name__ == "__main__":
    main()
