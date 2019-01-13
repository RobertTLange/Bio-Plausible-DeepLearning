import numpy as np
import matplotlib.pyplot as plt

# Define color blind-friendly color cycle
CB_color_cycle = ['#377eb8', '#ff7f00', '#4daf4a',
                  '#f781bf', '#a65628', '#984ea3',
                  '#999999', '#e41a1c', '#dede00']


"""
- Define plot helper functions
0. Smooth Time Series
1. Plot learning curves for single dataset
2. Plot learning curves for all datasets
3. Plot Individual image from either (Fashion-)MNIST or CIFAR-10
4. Plot one image per label
5. Plot Frobenius norm of relative weight changes and gradient changes
"""


def smooth(ts, windowSize):
    # Perform smoothed moving average with specified window to time series
    weights = np.repeat(1.0, windowSize) / windowSize
    ts_MA = np.convolve(ts, weights, 'valid')
    return ts_MA


def plot_learning(its, train_acc, val_acc,
                  train_loss, val_loss,
                  sm_window, pl_type, title):
    # Transform ticks such that they have numb x 10^5 shape
    its_ticks = np.arange(100000, np.max(its), 100000)
    its_labels_temp = [str(int(it/100000)) for it in its_ticks]
    its_labels = [it_l + r"$\times 10^5$" for it_l in its_labels_temp]
    its_labels[0] = r"$10^5$"

    if sm_window > 1:
        train_acc = smooth(train_acc, sm_window)
        val_acc = smooth(val_acc, sm_window)
        train_loss = smooth(train_loss, sm_window)
        val_loss = smooth(val_loss, sm_window)

    if pl_type == "accuracy":
        plt.plot(its[sm_window-1:], train_acc, c="r", label="BP-MLP: Train")
        plt.plot(its[sm_window-1:], val_acc, c="g", label="BP-MLP: Val.")
        plt.ylabel('Accuracy')
        plt.title(title)
        plt.xticks(its_ticks, [])
        plt.legend(loc=4, fontsize=8)
    elif pl_type == "loss":
        plt.plot(its[sm_window-1:], train_loss, c="b", label="BP-MLP: Train")
        plt.plot(its[sm_window-1:], val_loss, c="y", label="BP-MLP: Val.")
        plt.xlabel('Data Points')
        plt.ylabel('Loss')
        plt.xticks(its_ticks, its_labels)
        plt.legend(loc=1, fontsize=8)


def plot_all_learning(its, train_accs, val_accs,
                      train_losses, val_losses, sm_window,
                      sub_titles, save_fname=None):
    plt.figure(figsize=(10, 8), dpi=200)

    counter = 0

    for i in range(len(train_losses)):
        counter += 1
        plt.subplot(2, len(train_losses), counter)
        plot_learning(its[i], train_accs[i], val_accs[i],
                      train_losses[i], val_losses[i],
                      sm_window, "accuracy", sub_titles[i])
        plt.subplot(2, len(train_losses), counter+len(train_losses))
        plot_learning(its[i], train_accs[i], val_accs[i],
                      train_losses[i], val_losses[i],
                      sm_window, "loss", sub_titles[i])

    plt.tight_layout()
    if save_fname is not None:
        plt.savefig(save_fname, dpi=300)
        print("Saved figure to {}".format(save_fname))


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


def plot_weight_dev(its, l_id, fr_n_weights, fr_n_weights_ch, fr_n_weights_grad_ch,
                    title='Learning Dynamics and Convergence of Optimization',
                    save_fname=None):

    its_ticks = np.arange(100000, np.max(its[2]), 100000)
    its_labels_temp = [str(int(it/100000)) for it in its_ticks]
    its_labels = [it_l + r"$\times 10^5$" for it_l in its_labels_temp]
    its_labels[0] = r"$10^5$"

    fig, axs = plt.subplots(3, 3, figsize=(10, 8), dpi=200, sharey='row')
    fig.suptitle(title, fontsize=18)
    data_labels = ["MNIST", "Fashion-MNIST", "CIFAR-10"]

    for i in range(len(fr_n_weights)):
        axs[0, i].plot(its[i], fr_n_weights[i][l_id], label="DNN - Backprop")
        axs[0, i].set_ylabel(r"$||W_{t}||_F$", fontsize=15)
        axs[0, i].legend()
        axs[0, i].set_title(data_labels[i])
        axs[0, i].set_xticks(its_ticks)
        axs[0, i].set_xticklabels([])

        axs[1, i].plot(its[i], fr_n_weights_ch[i][l_id], label="DNN - Backprop")
        axs[1, i].set_ylabel(r"$\frac{||W_{t} - W_{t-1}||_F}{||W_{t-1}||_F}$",
                             fontsize=15)
        axs[1, i].set_xticks(its_ticks)
        axs[1, i].set_xticklabels([])
        axs[1, i].legend()

        axs[2, i].plot(its[i], fr_n_weights_grad_ch[i][l_id], label="DNN - Backprop")
        axs[2, i].set_ylabel(r"$\frac{||\nabla W_{t+1} - \nabla W_{t}||_F}{||\nabla W_{t}||_F}$", fontsize=15)
        axs[2, i].set_xticks(its_ticks)
        axs[2, i].set_xticklabels(its_labels)
        axs[2, i].legend()

    fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    if save_fname is not None:
        plt.savefig(save_fname, dpi=300)
        print("Saved figure to {}".format(save_fname))


def plot_bo_results(bo_data, title, save_fname=None):
    fig, axs = plt.subplots(3, 3, figsize=(10, 8), dpi=200, sharey='row')
    fig.suptitle(title, fontsize=18)

    data_labels = ["MNIST", "Fashion-MNIST", "CIFAR-10"]
    algo_labels = ["BP: MLP", "BP: CNN", "SegComp:MLP"]
    # Run Loop over different Algorithms
    for j in range(len(bo_data)):
        # Plot the time-series of BO evaluations
        for i in range(len(data_labels)):
            axs[0, i].plot(np.arange(1, len(bo_data[j][i])+1), bo_data[j][i])
            axs[0, i].set_title("Evaluations: " + data_labels[i])
            axs[0, i].set_xlabel("BO Iteration")
            if i == 0:
                axs[0, i].set_ylabel("k-fold CV Test Accuracy")

            axs[1, i].hist(bo_data[j][i], 50, density=1, alpha=0.75)
            axs[1, i].set_title("Histogramm: " + data_labels[i])
            axs[1, i].set_xlabel("k-fold CV Test Accuracy")

            axs[2, i].boxplot(bo_data[j][i], vert=True,
                              patch_artist=True)
            axs[2, i].set_title("Boxplot: " + data_labels[i])
            axs[2, i].set_xticklabels(algo_labels[:(j+1)],
                                      rotation=45, fontsize=8)
            if i == 0:
                axs[2, i].set_ylabel("k-fold CV Test Accuracy")

    fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    if save_fname is not None:
        plt.savefig(save_fname, dpi=300)
        print("Saved figure to {}".format(save_fname))
    return
