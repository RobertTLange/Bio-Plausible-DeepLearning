import time
import numpy as np

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
# from torchsummary import summary

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score

from utils.logger import Logger, update_logger
from utils.helpers import init_weights, get_data, get_test_error, train_model, update_tensor_dim, train_model_slim

# Dont print depreciation warning
import warnings
warnings.filterwarnings("ignore")

class CNN(nn.Module):
    def __init__(self, ch_sizes, k_sizes, stride, padding, out_size, verbose=False):
        super(CNN, self).__init__()
        self.layers = nn.ModuleList()

        W_in = 28

        for k in range(len(ch_sizes) - 1):
            self.layers.append(nn.Conv2d(in_channels=ch_sizes[k],
                                         out_channels=ch_sizes[k+1],
                                         kernel_size=k_sizes[k],
                                         stride=stride,
                                         padding=padding))

            W_in = update_tensor_dim(W_in, k_sizes[k], padding, stride)
            self.layers.append(nn.BatchNorm2d(ch_sizes[k+1]))
            self.layers.append(nn.ReLU())
            self.layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            W_in = update_tensor_dim(W_in, 2, 0, 2)
        self.layers.append(nn.Linear(W_in**2*ch_sizes[-1], out_size))
        # Xavier initialization of first
        init_weights(self.layers)
        if verbose:
            self.print_architecture()

    def forward(self, x):
        out = x
        for layer in self.layers[:-1]:
            out = layer(out)

        out = out.reshape(out.size(0), -1)
        out = self.layers[-1](out)
        return out

    def print_architecture(self):
        for layer in self.layers:
            print(layer)


def eval_cnn(dataset, batch_size, learning_rate, num_layers=2,
             ch_1=16, ch_2=32, ch_3=0, ch_4=0, ch_5=0,
             k_1=5, k_2=5, k_3=0, k_4=0, k_5=0,
             stride=1, padding=2,
             num_epochs=1, k_fold=2, verbose=False):

    # be careful with feeding only 1 channel - cifar has 3rgb
    ch_sizes = [1, ch_1, ch_2, ch_3, ch_4, ch_5][:(num_layers+1)]
    k_sizes = [k_1, k_2, k_3, k_4, k_5][:num_layers]

    if verbose:
        print("Batchsize: {}".format(batch_size))
        print("Learning Rate: {}".format(learning_rate))
        print("Architecture of Cross-Validated Network:")
        for i in range(len(k_sizes)):
            print("\t Layer {}: {} Channels, {} Kernel Size".format(i+1, ch_sizes[i+1], k_sizes[i]))

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Initialize list to store cross_val accuracies
    scores = []
    # Load dataset
    X, y = get_data(70000, dataset)

    # Split original dataset into folds (return idx)
    kf = StratifiedKFold(n_splits=k_fold, random_state=0)
    kf.get_n_splits(X)
    counter = 1

    for sub_index, test_index in kf.split(X, y):
        X_sub, X_test = X[sub_index], X[test_index]
        y_sub, y_test = y[sub_index], y[test_index]

        # Instantiate the model with layersize and Logging directory
        cnn_model = CNN(ch_sizes, k_sizes,
                        stride, padding, out_size=10)
        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(cnn_model.parameters(), lr=learning_rate)

        # Train the network
        model = train_model_slim("cnn", cnn_model, num_epochs,
                                 X_sub, y_sub, batch_size,
                                 device, optimizer, criterion)

        # Compute accuracy on hold-out set
        score_temp = get_test_error("cnn", device, model, X_test, y_test)
        scores.append(score_temp)

        # Clean memory after eval!
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if verbose:
            print("Cross-Validation Score Fold {}: {}".format(counter,
                                                              score_temp))
            counter += 1
    return np.mean(scores)

if __name__ == "__main__":
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Define batchsize for data-loading
    batch_size = 100

    # MNIST dataset
    X, y = get_data(num_samples=70000)
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        stratify=y,
                                                        random_state=0)

    # Feedforward Neural Network Parameters
    num_epochs = 2

    # Instantiate the model with layersize and Logging directory
    cnn_model = CNN(ch_sizes=[1, 16, 25], k_sizes=[3, 5],
                    stride=1, padding=2, out_size=10).to(device)

    print(cnn_model)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(cnn_model.parameters(), lr=0.001)

    model = train_model("cnn", cnn_model, num_epochs,
                        X, y, batch_size,
                        device, optimizer, criterion,
                        model_fname ="models/temp_model_cnn.ckpt",
                        verbose=True, logging=True)

    # Get test error
    score = get_test_error("cnn", device, model, X_test, y_test)
    print("Test Accuracy: {}".format(score))
