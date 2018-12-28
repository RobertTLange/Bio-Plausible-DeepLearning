import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_mldata
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score


def get_data(num_samples):
    mnist = fetch_mldata('MNIST original')
    torch.manual_seed(0)
    X = mnist.data.astype('float32').reshape(-1, 1, 28, 28)
    y = mnist.target.astype('int64')
    X, y = shuffle(X, y)
    X, y = X[:num_samples], y[:num_samples]
    X /= 255
    return X, y


def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.01)


def plot_learning(its, train_acc, val_acc, train_loss, val_loss, title):
    fig, ax1 = plt.subplots()
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Accuracy')
    l1 = ax1.plot(its, train_acc, c="r", label="Train Accuracy")
    l2 = ax1.plot(its, val_acc, c="g", label="Validation Accuracy")

    ax2 = ax1.twinx()
    ax2.set_ylabel('Loss')
    l3 = ax2.plot(its, train_loss, c="b", label="Train Loss")
    l4 = ax2.plot(its, val_loss, c="y", label="Validation Loss")

    lns = l1 + l2 + l3 + l4
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs, loc=7)

    plt.title(title)
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.show()


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
