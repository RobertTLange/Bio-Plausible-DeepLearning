import torch
import torch.nn as nn
import torchvision
from torchvision import transforms

import time
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold, train_test_split

from helpers import init_weights, get_data, get_test_error, train_model

# Dont print depreciation warning
import warnings
warnings.filterwarnings("ignore")

# Fully connected neural network with hidden layers
class DNN(nn.Module):
    def __init__(self, h_sizes=[784, 500], out_size=10, verbose=False):
        super(DNN, self).__init__()

        self.layers = nn.ModuleList()
        for k in range(len(h_sizes)-1):
            self.layers.append(nn.Linear(h_sizes[k], h_sizes[k+1]))
            self.layers.append(nn.ReLU())
        self.layers.append(nn.Linear(h_sizes[k+1], out_size))
        # Xavier initialization of first
        init_weights(self.layers)
        if verbose:
            self.print_architecture()

    def forward(self, x):
        out = x
        for layer in self.layers:
            out = layer(out)
        return out

    def print_architecture(self):
        for layer in self.layers:
            print(layer)

"""
# Reload model and set into eval/prediction mode
model = DNN(*args, **kwargs)
model.load_state_dict(torch.load(PATH))
model.eval()
"""

def eval_dnn(dataset, batch_size, learning_rate, num_layers=2,
             h_l_1=500, h_l_2=0, h_l_3=0, h_l_4=0, h_l_5=0, h_l_6=0,
             num_epochs=5, k_fold=3, verbose=False):

    if dataset == "mnist" or dataset == "fashion":
        h_in = 784
    elif dataset == "cifar10":
        h_in = 32*32*3
    h_sizes = [h_in, h_l_1, h_l_2, h_l_3, h_l_4, h_l_5, h_l_6][:(num_layers+1)]

    if verbose:
        print("Dataset: {}".format(dataset))
        print("Batchsize: {}".format(batch_size))
        print("Learning Rate: {}".format(learning_rate))
        print("Architecture of Cross-Validated Network:")
        for i in range(len(h_sizes)):
            print("\t Layer {}: {} Units".format(i, h_sizes[i]))

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
        dnn_model = DNN(h_sizes, out_size=10)
        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(dnn_model.parameters(), lr=learning_rate)

        # Train the network
        model = train_model("dnn", dnn_model, num_epochs,
                            X_sub, y_sub, batch_size,
                            device, optimizer, criterion,
                            model_fname ="models/temp_model_dnn.ckpt",
                            verbose=False, logging=False)

        # Compute accuracy on hold-out set
        score_temp = get_test_error("dnn", device, model, X_test, y_test)
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
    num_epochs = 5

    # Instantiate the model with layersize and Logging directory
    dnn_model = DNN(h_sizes=[784, 500, 300, 100], out_size=10)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(dnn_model.parameters(), lr=0.001)

    model = train_model("dnn", dnn_model, num_epochs,
                        X, y, batch_size,
                        device, optimizer, criterion,
                        model_fname ="models/temp_model_dnn.ckpt",
                        verbose=True, logging=True)

    # Get test error
    score = get_test_error("dnn", device, model, X_test, y_test)
    print("Test Accuracy: {}".format(score))
