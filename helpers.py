import torch
import torch.nn as nn
import matplotlib.pyplot as plt

def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.01)


def plot_learning(its, train_acc, test_acc, loss, title):
    fig, ax1 = plt.subplots()
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Accuracy')
    l1 = ax1.plot(its, train_acc, c="r", label="Train Accuracy")
    l2 = ax1.plot(its, test_acc, c="g", label="Test Accuracy")

    ax2 = ax1.twinx()
    ax2.set_ylabel('Loss')
    l3 = ax2.plot(its, loss, c="b", label="Train Loss")

    lns = l1 + l2 + l3
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs, loc=7)

    plt.title(title)
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.show()
