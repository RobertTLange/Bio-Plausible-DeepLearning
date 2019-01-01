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


def plot_kernels(tensor, num_cols=6):
    num_kernels = tensor.shape[0]
    num_rows = 1+ num_kernels // num_cols
    fig = plt.figure(figsize=(num_cols,num_rows))
    for i in range(num_kernels):
        ax1 = fig.add_subplot(num_rows,num_cols,i+1)
        ax1.imshow(tensor[i][0,:,:], cmap='gray')
        ax1.axis('off')
        ax1.set_xticklabels([])
        ax1.set_yticklabels([])

# plt.subplots_adjust(wspace=0.1, hspace=0.1)
#     plt.show()
# filters = cnn.modules();
# model_layers = [i for i in cnn.children()];
# first_layer = model_layers[0];
# second_layer = model_layers[1];
# first_kernels = first_layer[0].weight.data.numpy()
# plot_kernels(first_kernels, 8)
# second_kernels = second_layer[0].weight.data.numpy()
# plot_kernels(second_kernels, 8)
