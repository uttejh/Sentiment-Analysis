class Train:
    def __init__(self, features, labels):
        # percentage of training data
        split_frac = 0.8

        # split data into train and rest
        split_idx = int(len(features)*split_frac)

        self.train_x, remaining_x = features[:split_idx], features[split_idx:]
        self.train_y, remaining_y = labels[:split_idx], labels[split_idx:]

        # split rest data into test and validation
        test_idx = int(len(remaining_x)*0.5)
        self.val_x, self.test_x = remaining_x[:test_idx], remaining_x[test_idx:]
        self.val_y, self.test_y = remaining_y[:test_idx], remaining_y[test_idx:]

        print("\t\t\tFeature Shapes:")
        print("Train set: \t\t{}".format(self.train_x.shape),
              "\nValidation set: \t{}".format(self.val_x.shape),
              "\nTest set: \t\t{}".format(self.test_x.shape))