import numpy as np
import matplotlib.pyplot as plt

"""
- Define plot helper functions
"""
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


def plot_images(x, y, row_id, labels):

    l_id = y[row_id]
    pixels = x[row_id, :]
    pixels = np.array(pixels, dtype='uint8')
    if pixels.shape == (1, 28, 28):
        pixels = pixels.reshape((28, 28))
        temp_fig = plt.imshow(pixels, plt.get_cmap('gray_r'))
    elif pixels.shape == (3, 32, 32):
        pixels = pixels.reshape((32, 32, 3))
        temp_fig = plt.imshow(pixels)
    temp_fig.axes.get_xaxis().set_visible(False)
    temp_fig.axes.get_yaxis().set_visible(False)
    plt.title('{xyz}'.format(xyz=labels[l_id]))
    # plt.show()


def plot_labels(X, y, labels, save_fname=None):
    X *= 255
    plt.figure(figsize=(10, 8))

    u, indices = np.unique(y, return_index=True)
    counter = 0

    for i in indices:
        counter += 1
        plt.subplot(1, 10, counter)
        plot_images(X, y, i, labels)

    plt.tight_layout()
    if save_fname is not None:
        plt.savefig(save_fname, dpi=300)
        print("Saved figure to {}".format(save_fname))
    plt.show()
