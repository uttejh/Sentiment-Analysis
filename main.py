"""
Sentiment Analysis with an RNN
"""


def load_data(filepath):
    """
    reads a text file
    :type filename: object
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
    # TODO: Visualize data
    # TODO: Data Pre-processing
    # TODO: Encoding


if __name__ == "__main__":
    main()
