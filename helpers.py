import os
import gzip
import time
import numpy as np

import torch
import torch.nn as nn

from sklearn.datasets import fetch_mldata
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold, train_test_split

from logger import Logger, update_logger

"""
- Dataset specific helpers
"""
def load_mnist(path, kind='train'):
    """Load MNIST data from `path`"""
    labels_path = os.path.join(path,
                               '%s-labels-idx1-ubyte.gz'
                               % kind)
    images_path = os.path.join(path,
                               '%s-images-idx3-ubyte.gz'
                               % kind)

    with gzip.open(labels_path, 'rb') as lbpath:
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8,
                               offset=8)

    with gzip.open(images_path, 'rb') as imgpath:
        images = np.frombuffer(imgpath.read(), dtype=np.uint8,
                               offset=16).reshape(len(labels), 784)
    return images, labels



def get_data(num_samples, dataset="MNIST"):

    if dataset == "MNIST":
        data = fetch_mldata('MNIST original')
    elif dataset == "Fashion-MNIST":
        data = fetch_mldata('MNIST original')

    torch.manual_seed(0)
    X = data.data.astype('float32').reshape(-1, 1, 28, 28)
    y = data.target.astype('int64')
    X, y = shuffle(X, y)
    X, y = X[:num_samples], y[:num_samples]
    X /= 255
    return X, y


"""
- Modeling specific helpers
    a. Xavier initialization of all layers
    b. Accuracy computation on hold out set
    c. Compute and report learning stats
    d. Perform training of model
    e. Compute dimension of filtered image at next layer
"""
def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.01)


def get_test_error(model_type, device, model, X_test, y_test):
    if model_type == "dnn":
        X_test = X_test.reshape(X_test.shape[0], 28*28)
    X_test = torch.tensor(X_test).to(device)
    with torch.no_grad():
        y_pred = model(X_test).cpu().numpy().argmax(1)
    return accuracy_score(y_test, y_pred)


def report(verbose, losses, batch_sizes, y, y_proba, epoch, time, training=True):
    template = "{} | epoch {:>2} | "

    loss = np.average(losses, weights=batch_sizes)
    y_pred = np.argmax(y_proba, axis=1)
    acc = accuracy_score(y, y_pred)

    template += "acc: {:.4f} | loss: {:.4f} | time: {:.2f}"

    if verbose:
        print(template.format(
              'train' if training else 'valid', epoch + 1, acc, loss, time))
        if not training:
            print('-' * 50)

    return loss, acc


def train_model(model_type, model, num_epochs,
                X, y, batch_size,
                device, optimizer, criterion,
                model_fname ="temp_model_dnn.ckpt",
                verbose=True, logging=True):
    logger = Logger('./logs')

    model.to(device)

    # Select data
    idx_train, idx_valid = next(iter(StratifiedKFold(5, random_state=0).split(np.arange(len(X)), y)))

    if model_type == "dnn":
        X_train, X_valid, y_train, y_valid = (np.squeeze(X[idx_train]),
                                              np.squeeze(X[idx_valid]),
                                              y[idx_train], y[idx_valid])
    elif model_type == "cnn":
        # No squeze here needed - keep full dimensionality
        X_train, X_valid, y_train, y_valid = (X[idx_train],
                                              X[idx_valid],
                                              y[idx_train], y[idx_valid])

    dataset_train = torch.utils.data.TensorDataset(torch.tensor(X_train),
                                                   torch.tensor(y_train))
    dataset_valid = torch.utils.data.TensorDataset(torch.tensor(X_valid),
                                                   torch.tensor(y_valid))

    for epoch in range(num_epochs):

        train_out = train_step(model_type, model, dataset_train, batch_size=batch_size,
                               device=device, criterion=criterion,
                               optimizer=optimizer)
        train_loss, train_acc = report(verbose, y=y_train, epoch=epoch,
                                       training=True, **train_out)

        valid_out = valid_step(model_type, model, dataset_valid, batch_size=batch_size,
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


def train_step(model_type, model, dataset, device, criterion, batch_size, optimizer):

    model.train()
    y_preds = []
    losses = []
    batch_sizes = []
    tic = time.time()

    for Xi, yi in torch.utils.data.DataLoader(dataset, batch_size=batch_size):
        if model_type == "dnn":
            Xi, yi = Xi.reshape(Xi.shape[0], 28*28).to(device), yi.to(device)
        elif model_type == "cnn":
            Xi, yi = Xi.to(device), yi.to(device)
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


def valid_step(model_type, model, dataset, device, criterion, batch_size):

    model.eval()
    y_preds = []
    losses = []
    batch_sizes = []
    tic = time.time()
    with torch.no_grad():
        for Xi, yi in torch.utils.data.DataLoader(dataset, batch_size=batch_size):
            if model_type == "dnn":
                Xi, yi = Xi.reshape(Xi.shape[0], 28*28).to(device), yi.to(device)
            elif model_type == "cnn":
                Xi, yi = Xi.to(device), yi.to(device)
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


def update_tensor_dim(W_in, k_size, padding, stride):
    return (W_in - k_size + 2*padding)/stride + 1

if __name__ == "__main__":
    get_data(num_samples=100, dataset="MNIST")
