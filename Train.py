import torch
from torch.utils.data import TensorDataset, DataLoader
from Network import Network

class Train:
    """
    TODO
    """
    def __init__(self, features, labels):
        # percentage of training data
        split_frac = 0.8
        # batch size
        self.batch_size = 50

        # split data into train and rest
        split_idx = int(len(features)*split_frac)

        train_x, remaining_x = features[:split_idx], features[split_idx:]
        train_y, remaining_y = labels[:split_idx], labels[split_idx:]

        # split rest data into test and validation
        test_idx = int(len(remaining_x)*0.5)
        val_x, test_x = remaining_x[:test_idx], remaining_x[test_idx:]
        val_y, test_y = remaining_y[:test_idx], remaining_y[test_idx:]

        print("\t\t\tFeature Shapes:")
        print("Train set: \t\t{}".format(train_x.shape),
              "\nValidation set: \t{}".format(val_x.shape),
              "\nTest set: \t\t{}".format(test_x.shape))

        # data loader
        self.train_loader = self.create_dataloader(train_x, train_y)
        self.val_loader = self.create_dataloader(val_x, val_y)
        self.test_loader = self.create_dataloader(test_x, test_y)

        # visualize a batch
        self.visualize_batch()

        # TODO: create a func train_model that uses network

    def create_dataloader(self, data_x, data_y):
        """
        creates a dataloader useful for training testing and validation
        Args:
            data_x(ndarray): review integers
            data_y(ndarray: labels

        Returns:
            dataset_loader(tensor): reviews and labels in form of tensors
        """
        # create tensors of dataset
        dataset = TensorDataset(torch.from_numpy(data_x), torch.from_numpy(data_y))

        # data loaders
        dataset_loader = DataLoader(dataset, shuffle=True, batch_size=self.batch_size)

        return dataset_loader

    def visualize_batch(self):
        """
        Helper function to print batch details
        """
        dataiter = iter(self.train_loader)
        sample_x, sample_y = dataiter.next()

        print('Sample input size: ', sample_x.size())  # batch_size, seq_length
        print('Sample input: \n', sample_x)
        print()
        print('Sample label size: ', sample_y.size())  # batch_size
        print('Sample label: \n', sample_y)

    def instantiate_model(self, vocab_to_int, embedding_dim, hidden_dim, output_size, n_layers):
        vocab_size = len(vocab_to_int) + 1

        net = Network(vocab_size, output_size, embedding_dim, hidden_dim, n_layers)

        return net

    def train_model(self, vocab_to_int, embedding_dim, hidden_dim, output_size, n_layers, lr, epochs):
        net = self.instantiate_model(vocab_to_int, embedding_dim, hidden_dim, output_size, n_layers)
        print(net)

