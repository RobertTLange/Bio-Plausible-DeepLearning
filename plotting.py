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
