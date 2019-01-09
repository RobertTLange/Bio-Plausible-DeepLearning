import numpy as np
import matplotlib.pyplot as plt

"""
- Define plot helper functions
1. Plot learning curves for single dataset
2. Plot learning curves for all datasets
3. Plot Individual image from either (Fashion-)MNIST or CIFAR-10
4. Plot one image per label
5. Plot Frobenius norm of relative weight changes and gradient changes
"""

def plot_learning(its, train_acc, val_acc, train_loss, val_loss, pl_type, title):
    if pl_type == "accuracy":
        plt.plot(its, train_acc, c="r", label="Train Accuracy")
        plt.plot(its, val_acc, c="g", label="Validation Accuracy")
        plt.ylabel('Accuracy')
        plt.title(title)
    elif pl_type == "loss":
        plt.plot(its, train_loss, c="b", label="Train Loss")
        plt.plot(its, val_loss, c="y", label="Validation Loss")
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
    plt.legend(loc=7)

def plot_all_learning(its, train_accs, val_accs,
                      train_losses, val_losses,
                      sub_titles, save_fname=None):
    plt.figure(figsize=(10, 8), dpi=200)

    counter = 0

    for i in range(len(train_losses)):
        counter += 1
        plt.subplot(2, len(train_losses), counter)
        plot_learning(its[i], train_accs[i], val_accs[i],
                      train_losses[i], val_losses[i], "accuracy",
                      sub_titles[i])
        plt.subplot(2, len(train_losses), counter+len(train_losses))
        plot_learning(its[i], train_accs[i], val_accs[i],
                      train_losses[i], val_losses[i], "loss",
                      sub_titles[i])

    plt.tight_layout()
    if save_fname is not None:
        plt.savefig(save_fname, dpi=300)
        print("Saved figure to {}".format(save_fname))
    #plt.show()


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


def plot_weight_dev(its, fr_n_weights_ch, fr_n_weight_grad_ch,
                    fr_n_biases_ch, fr_n_bias_grad_ch,
                    title='Learning Dynamics and Convergence of Optimization'):

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(18, 16), dpi=200, sharey='row')
    fig.suptitle(title, fontsize=18)

    ax1.plot(its, fr_n_weights_ch, label="DNN - Backprop")
    ax1.set_ylabel(r"$\frac{||W_{t+1} - W_{t}||_F}{||W_{t}||_F}$", fontsize=15)
    ax1.legend()

    ax2.plot(its, fr_n_biases_ch, label="DNN - Backprop")
    ax2.set_ylabel(r"$\frac{||b_{t+1} - b_{t}||_F}{||b_{t}||_F}$", fontsize=15)
    ax2.legend()

    ax3.plot(its, fr_n_weight_grad_ch, label="DNN - Backprop")
    ax3.set_xlabel("Iteration", fontsize=15)
    ax3.set_ylabel(r"$\frac{||\nabla W_{t+1} - \nabla W_{t}||_F}{||\nabla W_{t}||_F}$", fontsize=15)
    ax3.legend()

    ax4.plot(its, fr_n_bias_grad_ch, label="DNN - Backprop")
    ax4.set_xlabel("Iteration", fontsize=15)
    ax4.set_ylabel(r"$\frac{||\nabla b_{t+1} - \nabla b_{t}||_F}{||\nabla b_{t}||_F}$", fontsize=15)
    ax4.legend()
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
