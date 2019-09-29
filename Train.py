class Train:
    def __init__(self, features, labels):
        # percentage of training data
        split_frac = 0.8

        # split data into train and rest
        split_idx = int(len(features)*split_frac)

        self.train_x, self.remaining_x = features[:split_idx], features[split_idx:]
        self.train_y, self.remaining_y = labels[:split_idx], labels[split_idx:]

        # split rest data into test and validation
        test_idx = int(len(self.remaining_x)*0.5)
        self.val_x, self.test_x = self.remaining_x[:test_idx], self.remaining_x[test_idx:]
        self.val_y, self.test_y = self.remaining_y[:test_idx], self.remaining_y[test_idx:]