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

# Fully connected neural network with one hidden layer
class DNN(nn.Module):
    def __init__(self, h_sizes=[784, 500], out_size=10):
        super(DNN, self).__init__()

        self.layers = nn.ModuleList()
        for k in range(len(h_sizes)-1):
            self.layers.append(nn.Linear(h_sizes[k], h_sizes[k+1]))
            self.layers.append(nn.ReLU())
        self.layers.append(nn.Linear(h_sizes[k+1], out_size))
        # Xavier initialization of first
        init_weights(self.layers)
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
                    X, y, device, optimizer, criterion,
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
        report(y=y_train, epoch=epoch, training=True, **train_out)

        valid_out = valid_step(model, dataset_valid, batch_size=batch_size,
                               device=device, criterion=criterion)
        report(y=y_valid, epoch=epoch, training=False, **valid_out)

        print('-' * 50)
        # Save the model checkpoint
        torch.save(model.state_dict(), model_fname)

        # if verbose:
        #     print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Train Acc: {:.2f}, Test Acc: {:.2f}'
        #            .format(epoch+1, num_epochs,
        #                    i+1, len(train_loader),
        #                    loss.item(),
        #                    train_accuracy.item(),
        #                    test_accuracy))
        #
        # if logging:
        #     update_logger(logger, epoch, i, loss,
        #                   train_accuracy, test_accuracy,
        #                   model, images, train_loader)

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

if __name__ == "__main__":
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Define batchsize for data-loading
    batch_size = 100

    # MNIST dataset
    X, X_test, y, y_test = get_data(num_samples=20000)

    # Feedforward Neural Network Parameters
    num_epochs = 5

    # Instantiate the model with layersize and Logging directory
    dnn_model = DNN(h_sizes=[784, 500, 300, 100], out_size=10)
    logger = Logger('./logs')

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(dnn_model.parameters(), lr=0.001)

    model = train_dnn_model(dnn_model, num_epochs,
                    X, y,
                    device, optimizer, criterion,
                    model_fname ="models/temp_model_dnn.ckpt",
                    verbose=True, logging=True)

    # X_test = torch.tensor(X_test).to(device)
    # with torch.no_grad():
    #     y_pred = model(X_test).cpu().numpy().argmax(1)
    # print(accuracy_score(y_test, y_pred))
