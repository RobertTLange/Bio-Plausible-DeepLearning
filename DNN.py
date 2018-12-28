import torch
import torch.nn as nn
import torchvision
from torchvision import transforms

import time
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score

from logger import Logger, update_logger
from helpers import init_weights, get_data, report


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


def train_dnn_model(model, num_epochs,
                    X, y, batch_size,
                    device, optimizer, criterion,
                    model_fname ="temp_model.ckpt",
                    verbose=True, logging=True):
    logger = Logger('./logs')

    model.to(device)

    # Select data
    idx_train, idx_valid = next(iter(StratifiedKFold(5, random_state=0).split(np.arange(len(X)), y)))
    X_train, X_valid, y_train, y_valid = (np.squeeze(X[idx_train]),
                                          np.squeeze(X[idx_valid]),
                                          y[idx_train], y[idx_valid])

    dataset_train = torch.utils.data.TensorDataset(torch.tensor(X_train),
                                                   torch.tensor(y_train))
    dataset_valid = torch.utils.data.TensorDataset(torch.tensor(X_valid),
                                                   torch.tensor(y_valid))

    for epoch in range(num_epochs):

        train_out = train_step(model, dataset_train, batch_size=batch_size,
                               device=device, criterion=criterion,
                               optimizer=optimizer)
        train_loss, train_acc = report(verbose, y=y_train, epoch=epoch,
                                       training=True, **train_out)

        valid_out = valid_step(model, dataset_valid, batch_size=batch_size,
                               device=device, criterion=criterion)
        valid_loss, valid_acc = report(verbose, y=y_valid, epoch=epoch,
                                       training=False, **valid_out)

        # Save the model checkpoint
        torch.save(model.state_dict(), model_fname)

        if logging:
            update_logger(logger, epoch+1, (epoch+1)*len(dataset_train),
                          train_loss, valid_loss, train_acc, valid_acc,
                          model)

    return model

def train_step(model, dataset, device, criterion, batch_size, optimizer):

    model.train()
    y_preds = []
    losses = []
    batch_sizes = []
    tic = time.time()

    for Xi, yi in torch.utils.data.DataLoader(dataset, batch_size=batch_size):
        Xi, yi = Xi.reshape(Xi.shape[0], 28*28).to(device), yi.to(device)
        optimizer.zero_grad()
        y_pred = model(Xi)
        loss = criterion(y_pred, yi)
        loss.backward()
        optimizer.step()

        y_preds.append(y_pred)
        losses.append(loss.item())
        batch_sizes.append(len(Xi))
    toc = time.time()

    return {
        'losses': losses,
        'batch_sizes': batch_sizes,
        'y_proba': torch.cat(y_preds).cpu().detach().numpy(),
        'time': toc - tic,
    }


def valid_step(model, dataset, device, criterion, batch_size):

    model.eval()
    y_preds = []
    losses = []
    batch_sizes = []
    tic = time.time()
    with torch.no_grad():
        for Xi, yi in torch.utils.data.DataLoader(dataset, batch_size=batch_size):
            Xi, yi = Xi.reshape(Xi.shape[0], 28*28).to(device), yi.to(device)
            y_pred = model(Xi)
            loss = criterion(y_pred, yi)

            y_preds.append(y_pred)
            loss = loss.item()
            losses.append(loss)
            batch_sizes.append(len(Xi))
    toc = time.time()

    return {
        'losses': losses,
        'batch_sizes': batch_sizes,
        'y_proba': torch.cat(y_preds).cpu().detach().numpy(),
        'time': toc - tic,
    }

"""
model = DNN(*args, **kwargs)
model.load_state_dict(torch.load(PATH))
model.eval()
"""

def eval_dnn(batch_size, h_sizes, learning_rate, k_fold=5, verbose=False):
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Feedforward Neural Network Parameters
    num_epochs = 5
    # Initialize list to store cross_val accuracies
    scores = []
    # Load dataset
    X, y = get_data(num_samples=70000)

    # Split original dataset into folds (return idx)
    kf = StratifiedKFold(n_splits=k_fold, random_state=0)
    kf.get_n_splits(X)

    for sub_index, test_index in kf.split(X, y):
        X_sub, X_test = X[sub_index], X[test_index]
        y_sub, y_test = y[sub_index], y[test_index]

        # Instantiate the model with layersize and Logging directory
        dnn_model = DNN(h_sizes, out_size=10)
        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(dnn_model.parameters(), lr=learning_rate)

        # Train the network
        model = train_dnn_model(dnn_model, num_epochs,
                                X_sub, y_sub, batch_size,
                                device, optimizer, criterion,
                                model_fname ="models/temp_model_dnn.ckpt",
                                verbose=False, logging=False)

        # Compute accuracy on hold-out set
        score_temp = get_test_error(device, model, X_test, y_test)
        print(score_temp)
        scores.append(score_temp)
    return np.mean(scores)


def get_test_error(device, model, X_test, y_test):
    X_test = X_test.reshape(X_test.shape[0], 28*28)
    X_test = torch.tensor(X_test).to(device)
    with torch.no_grad():
        y_pred = model(X_test).cpu().numpy().argmax(1)
    return accuracy_score(y_test, y_pred)


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

    model = train_dnn_model(dnn_model, num_epochs,
                            X, y, batch_size,
                            device, optimizer, criterion,
                            model_fname ="models/temp_model_dnn.ckpt",
                            verbose=True, logging=True)

    # Get test error
    score = get_test_error(device, model, X_test, y_test)
    print("Test Accuracy: {}".format(score))
