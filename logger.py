# Code referenced from https://gist.github.com/gyglim/1f8dfb1b5c82627ae3efcfbbadb9f514
import os
import glob
import tensorflow as tf
import numpy as np
import scipy.misc
try:
    from StringIO import StringIO  # Python 2.7
except ImportError:
    from io import BytesIO         # Python 3.x


class Logger(object):

    def __init__(self, log_dir):
        """Create a summary writer logging to log_dir."""
        self.writer = tf.summary.FileWriter(log_dir)

    def scalar_summary(self, tag, value, step):
        """Log a scalar variable."""
        summary = tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=value)])
        self.writer.add_summary(summary, step)

    def image_summary(self, tag, images, step):
        """Log a list of images."""

        img_summaries = []
        for i, img in enumerate(images):
            # Write the image to a string
            try:
                s = StringIO()
            except:
                s = BytesIO()
            scipy.misc.toimage(img).save(s, format="png")

            # Create an Image object
            img_sum = tf.Summary.Image(encoded_image_string=s.getvalue(),
                                       height=img.shape[0],
                                       width=img.shape[1])
            # Create a Summary value
            img_summaries.append(tf.Summary.Value(tag='%s/%d' % (tag, i), image=img_sum))

        # Create and write Summary
        summary = tf.Summary(value=img_summaries)
        self.writer.add_summary(summary, step)

    def histo_summary(self, tag, values, step, bins=1000):
        """Log a histogram of the tensor of values."""

        # Create a histogram using numpy
        counts, bin_edges = np.histogram(values, bins=bins)

        # Fill the fields of the histogram proto
        hist = tf.HistogramProto()
        hist.min = float(np.min(values))
        hist.max = float(np.max(values))
        hist.num = int(np.prod(values.shape))
        hist.sum = float(np.sum(values))
        hist.sum_squares = float(np.sum(values**2))

        # Drop the start of the first bin
        bin_edges = bin_edges[1:]

        # Add bin edges and counts
        for edge in bin_edges:
            hist.bucket_limit.append(edge)
        for c in counts:
            hist.bucket.append(c)

        # Create and write Summary
        summary = tf.Summary(value=[tf.Summary.Value(tag=tag, histo=hist)])
        self.writer.add_summary(summary, step)
        self.writer.flush()


def update_logger(logger, epoch, i, loss, train_accuracy, test_accuracy,
                  model, images, train_loader):
    info = {'loss': loss.item(),
            'train_accuracy': train_accuracy.item(),
            'test_accuracy': test_accuracy}

    for tag, value in info.items():
        logger.scalar_summary(tag, value, epoch*len(train_loader) + i+1)

    # 2. Log values and gradients of the parameters (histogram summary)
    for tag, value in model.named_parameters():
        tag = tag.replace('.', '/')
        logger.histo_summary(tag, value.data.cpu().numpy(), epoch*len(train_loader) + i+1)
        logger.histo_summary(tag+'/grad', value.grad.data.cpu().numpy(), epoch*len(train_loader) + i+1)

    # 3. Log training images (image summary)
    info = { 'images': images.view(-1, 28, 28)[:10].cpu().numpy() }

    for tag, images in info.items():
        logger.image_summary(tag, images, epoch*len(train_loader) + i+1)
    return


def get_latest_log_fname(log_dir):
    list_of_files = glob.glob(log_dir + "/*") # * means all if need specific format then *.csv
    latest_file = max(list_of_files, key=os.path.getctime)
    return latest_file


def process_logger(log_fname, save_fname=None):
    iterations = []
    losses = []
    train_accuracies = []
    test_accuracies = []

    for e in tf.train.summary_iterator(log_fname):
        for v in e.summary.value:
            if v.tag == 'loss':
                losses.append(v.simple_value)
            elif v.tag == 'train_accuracy':
                train_accuracies.append(v.simple_value)
            elif v.tag == 'test_accuracy':
                test_accuracies.append(v.simple_value)
        iterations.append(int(e.step))

    iterations = np.unique(iterations)[1:]

    if save_fname is not None:
        out_array = np.array([iterations, losses,
                              train_accuracies, test_accuracies])
        np.savetxt(save_fname, out_array)
    return iterations, losses, train_accuracies, test_accuracies
