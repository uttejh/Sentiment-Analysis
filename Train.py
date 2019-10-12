import torch
from torch.utils.data import TensorDataset, DataLoader
from Network import Network
import torch.nn as nn
import numpy as np

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
        """
        Train the model
        """
        net = self.instantiate_model(vocab_to_int, embedding_dim, hidden_dim, output_size, n_layers)
        print(net)

        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(net.parameters(), lr=lr)

        # training params

        epochs = 4  # 3-4 is approx where I noticed the validation loss stop decreasing

        counter = 0
        print_every = 100
        clip = 5  # gradient clipping

        net.train()

        for e in range(epochs):
            # initialize hidden state
            h = net.init_hidden(self.batch_size)

            # batch loop
            for inputs, labels in self.train_loader:
                counter += 1

                # Creating new variables for the hidden state, otherwise
                # we'd backprop through the entire training history
                h = tuple([each.data for each in h])

                # zero accumulated gradients
                net.zero_grad()

                # get the output from the model
                output, h = net(inputs, h)

                # calculate the loss and perform backprop
                loss = criterion(output.squeeze(), labels.float())
                loss.backward()
                # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
                nn.utils.clip_grad_norm_(net.parameters(), clip)
                optimizer.step()

                # loss stats
                if counter % print_every == 0:
                    # Get validation loss
                    val_h = net.init_hidden(self.batch_size)
                    val_losses = []
                    net.eval()
                    for inputs, labels in self.val_loader:

                        # Creating new variables for the hidden state, otherwise
                        # we'd backprop through the entire training history
                        val_h = tuple([each.data for each in val_h])

                        output, val_h = net(inputs, val_h)
                        val_loss = criterion(output.squeeze(), labels.float())

                        val_losses.append(val_loss.item())

                    net.train()
                    print("Epoch: {}/{}...".format(e + 1, epochs),
                          "Step: {}...".format(counter),
                          "Loss: {:.6f}...".format(loss.item()),
                          "Val Loss: {:.6f}".format(np.mean(val_losses)))

        return net

    def test_model(self, net):
        """
        Test overall performance of the model
        """
        # Get test data loss and accuracy

        test_losses = []  # track loss
        num_correct = 0

        # init hidden state
        h = net.init_hidden(self.batch_size)

        net.eval()
        # iterate over test data
        for inputs, labels in self.test_loader:

            # Creating new variables for the hidden state, otherwise
            # we'd backprop through the entire training history
            h = tuple([each.data for each in h])

            # get predicted outputs
            output, h = net(inputs, h)
            criterion = nn.BCELoss()

            # calculate loss
            test_loss = criterion(output.squeeze(), labels.float())
            test_losses.append(test_loss.item())

            # convert output probabilities to predicted class (0 or 1)
            pred = torch.round(output.squeeze())  # rounds to the nearest integer

            # compare predictions to true label
            correct_tensor = pred.eq(labels.float().view_as(pred))
            correct = np.squeeze(correct_tensor.numpy())
            num_correct += np.sum(correct)

        # -- stats! -- ##
        # avg test loss
        print("Test loss: {:.3f}".format(np.mean(test_losses)))

        # accuracy over all test data
        test_acc = num_correct / len(self.test_loader.dataset)
        print("Test accuracy: {:.3f}".format(test_acc))
