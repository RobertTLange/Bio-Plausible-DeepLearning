# Code referenced from https://gist.github.com/gyglim/1f8dfb1b5c82627ae3efcfbbadb9f514
import os
import glob
import tensorflow as tf
import numpy as np
import scipy.misc
import _pickle as pickle

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


def update_logger(logger, epoch, its,
                  train_loss, valid_loss,
                  train_accuracy, valid_accuracy,
                  model):
    info = {'train_loss': train_loss.item(),
            'valid_loss': valid_loss.item(),
            'train_accuracy': train_accuracy.item(),
            'valid_accuracy': valid_accuracy}

    for tag, value in info.items():
        logger.scalar_summary(tag, value, its)

    # 2. Log values and gradients of the parameters (histogram summary)
    for tag, value in model.named_parameters():
        tag = tag.replace('.', '/')
        logger.histo_summary(tag, value.data.cpu().numpy(), its)
        logger.histo_summary(tag+'/grad', value.grad.data.cpu().numpy(), its)

    # 3. Log training images (image summary)
    # info = { 'images': images.view(-1, 28, 28)[:10].cpu().numpy() }

    # for tag, images in info.items():
    #     logger.image_summary(tag, images, its)
    return


def get_latest_log_fname(log_dir):
    list_of_files = glob.glob(log_dir + "/*")
    latest_file = max(list_of_files, key=os.path.getctime)
    return latest_file


def process_logger_learning(log_fnames, save_fname=None):
    iterations = []
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    for log_fname in log_fnames:
        its = []
        train_loss = []
        val_loss = []
        train_acc = []
        val_acc = []

        for e in tf.train.summary_iterator(log_fname):
            for v in e.summary.value:
                if v.tag == 'train_loss':
                    train_loss.append(v.simple_value)
                elif v.tag == 'valid_loss':
                    val_loss.append(v.simple_value)
                elif v.tag == 'train_accuracy':
                    train_acc.append(v.simple_value)
                elif v.tag == 'valid_accuracy':
                    val_acc.append(v.simple_value)
            its.append(int(e.step))

        its = np.unique(its)[1:]

        iterations.append(its)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)

    if save_fname is not None:
        out_array = np.array([iterations, train_losses, val_losses,
                              train_accuracies, val_accuracies])
        np.savetxt(save_fname, out_array)
    return iterations, train_losses, val_losses, train_accuracies, val_accuracies


class WeightLogger():
    def __init__(self, log_dir, wlog_fname, layer_ids):
        self.iterations = []
        self.weights = {key: None for key in layer_ids}
        self.weight_gradients = {key: None for key in layer_ids}
        self.biases = {key: None for key in layer_ids}
        self.bias_gradients = {key: None for key in layer_ids}

        self.fr_n_weights = {key: [] for key in layer_ids}
        self.fr_n_weights_ch = {key: [] for key in layer_ids}
        self.fr_n_weights_grad_ch = {key: [] for key in layer_ids}

        self.fr_n_biases = {key: [] for key in layer_ids}
        self.fr_n_biases_ch = {key: [] for key in layer_ids}
        self.fr_n_biases_grad_ch = {key: [] for key in layer_ids}

        self.save_fname = log_dir + wlog_fname

    def update_weight_logger(self, iter, model):
        self.iterations.append(iter)
        for tag, value in model.named_parameters():
            l_id = int(tag[7])
            if tag.startswith('layers') and tag.endswith('weight'):
                if len(self.iterations) > 1:
                    self.compute_stats(l_id, value, "weights")
                self.weights[l_id] = value.data.cpu().numpy().copy()
                self.weight_gradients[l_id] = value.grad.data.cpu().numpy().copy()

            elif tag.startswith('layers') and tag.endswith('bias'):
                if len(self.iterations) > 1:
                    self.compute_stats(l_id, value, "biases")
                self.biases[l_id] = value.data.cpu().numpy().copy()
                self.bias_gradients[l_id] = value.grad.data.cpu().numpy().copy()

    def compute_stats(self, l_id, value, param_type):
        temp_param = value.data.cpu().numpy().copy()
        temp_grad_param = value.grad.data.cpu().numpy().copy()

        if param_type == "weights":
            temp_param_old = self.weights[l_id]
            temp_grad_param_old = self.weight_gradients[l_id]
        elif param_type == "biases":
            temp_param_old = self.biases[l_id]
            temp_grad_param_old = self.bias_gradients[l_id]

        fr_n_param = np.linalg.norm(temp_param)
        fr_n_ch = np.linalg.norm(temp_param - temp_param_old)/np.linalg.norm(temp_param_old)
        fr_n_grad_ch = np.linalg.norm(temp_grad_param - temp_grad_param_old)/np.linalg.norm(temp_grad_param_old)

        if param_type == "weights":
            self.fr_n_weights[l_id].append(fr_n_param)
            self.fr_n_weights_ch[l_id].append(fr_n_ch)
            self.fr_n_weights_grad_ch[l_id].append(fr_n_grad_ch)
        elif param_type == "biases":
            self.fr_n_biases[l_id].append(fr_n_param)
            self.fr_n_biases_ch[l_id].append(fr_n_ch)
            self.fr_n_biases_grad_ch[l_id].append(fr_n_grad_ch)

    def dump_data(self):
        with open(self.save_fname, 'wb') as fp:
            pickle.dump(self.iterations, fp)

            pickle.dump(self.fr_n_weights, fp)
            pickle.dump(self.fr_n_weights_ch, fp)
            pickle.dump(self.fr_n_weights_grad_ch, fp)

            pickle.dump(self.fr_n_biases, fp)
            pickle.dump(self.fr_n_biases_ch, fp)
            pickle.dump(self.fr_n_biases_grad_ch, fp)


def process_logger_weights(log_fname):
    # Load in the Weight Logs
    with open(log_fname, 'rb') as fp:
        iterations = pickle.load(fp, encoding='latin1')

        fr_n_weights = pickle.load(fp, encoding='latin1')
        fr_n_weights_ch = pickle.load(fp, encoding='latin1')
        fr_n_weights_grad_ch = pickle.load(fp, encoding='latin1')

        fr_n_biases = pickle.load(fp, encoding='latin1')
        fr_n_biases_ch = pickle.load(fp, encoding='latin1')
        fr_n_biases_grad_ch = pickle.load(fp, encoding='latin1')

    # return iterations, weights, weight_grad, biases, bias_grad
    return iterations[1:], fr_n_weights, fr_n_weights_ch, fr_n_weights_grad_ch, fr_n_biases, fr_n_biases_ch, fr_n_biases_grad_ch
