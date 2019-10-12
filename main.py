"""
Sentiment Analysis with an RNN
"""
from Preprocess import Preprocess
import numpy as np
from Train import Train
import torch


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


def predict(preprocess, net, test_review, sequence_length=200):
    """
    Predict an individual review
    """
    net.eval()

    # tokenize review
    test_ints = preprocess.tokenize_review(test_review)

    # pad tokenized sequence
    seq_length = sequence_length
    features = preprocess.pad_features(test_ints, seq_length)

    # convert to tensor to pass into your model
    feature_tensor = torch.from_numpy(features)

    batch_size = feature_tensor.size(0)

    # initialize hidden state
    h = net.init_hidden(batch_size)

    # get the output from the model
    output, h = net(feature_tensor, h)

    # convert output probabilities to predicted class (0 or 1)
    pred = torch.round(output.squeeze())
    # printing output value, before rounding
    print('Prediction value, pre-rounding: {:.6f}'.format(output.item()))

    # print custom response
    if (pred.item() == 1):
        print("Positive review detected!")
    else:
        print("Negative review detected.")


def main():
    """
    1. Preprocess data
    2. Create a network
    3. Train and validate Model
    4. Test Model
    %. Predict individual reviews
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

    # initiate train
    train = Train(features, encoded_labels)

    hyperparameters = {"vocab_to_int": vocab_to_int, "embedding_dim": 400,
                       "hidden_dim": 256, "output_size": 1, "n_layers": 2,
                       "lr": 0.001, "epochs": 4}
    model = train.train_model(**hyperparameters)

    train.test_model(model)

    # negative test review
    test_review = "The worst movie I have seen; acting was terrible and I want my money back. This movie had bad " \
                      "acting and the dialogue was slow. "
    test_ints = preprocess.tokenize_review(test_review, vocab_to_int)
    features = preprocess.pad_features(test_ints, seq_length)
    feature_tensor = torch.from_numpy(features)

    predict(preprocess, model, test_review, seq_length)


if __name__ == "__main__":
    main()
