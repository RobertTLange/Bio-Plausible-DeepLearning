import _pickle as pickle
import numpy as np

class CompDNN_Logger():
    def __init__(self, log_dir, log_fname):
        self.save_fname = log_dir + log_fname
        # Init empty lists to store results in
        self.iterations = []
        self.train_losses = []
        self.val_losses = []
        self.train_accs = []
        self.val_accs = []

    def update(self, its, train_loss, val_loss,
               train_acc, val_acc):
        self.iterations.append(its)
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)
        self.train_accs.append(train_acc)
        self.val_accs.append(val_acc)
        self.dump_data()

    def dump_data(self):
        with open(self.save_fname, 'wb') as fp:
            pickle.dump(self.iterations, fp)
            pickle.dump(self.train_losses, fp)
            pickle.dump(self.val_losses, fp)
            pickle.dump(self.train_accs, fp)
            pickle.dump(self.val_accs, fp)
        return


class Weight_CompDNN_Logger():
    def __init__(self, log_dir, wlog_fname, layer_ids):
        self.save_fname = log_dir + wlog_fname
        # Init empty lists to store results in
        self.iterations = []
        self.weights = None
        self.weight_gradients = None
        self.fr_n_weights = {key: [] for key in layer_ids}
        self.fr_n_weights_ch = {key: [] for key in layer_ids}
        self.fr_n_weights_grad_ch = {key: [] for key in layer_ids}

    def update(self, its, weights, weight_gradients):
        self.iterations.append(its)

        if self.weights is not None:
            for l_id in range(len(weights)):
                self.compute_stats(l_id, weights[l_id], weight_gradients[l_id])

        self.weights = weights
        self.weight_gradients = weight_gradients
        self.dump_data()

    def compute_stats(self, l_id, temp_param, temp_grad_param):

        temp_param_old = self.weights[l_id]
        temp_grad_param_old = self.weight_gradients[l_id]

        fr_n_param = np.linalg.norm(temp_param)
        fr_n_ch = np.linalg.norm(temp_param - temp_param_old)/np.linalg.norm(temp_param_old)
        fr_n_grad_ch = np.linalg.norm(temp_grad_param - temp_grad_param_old)/np.linalg.norm(temp_grad_param_old)

        self.fr_n_weights[l_id].append(fr_n_param)
        self.fr_n_weights_ch[l_id].append(fr_n_ch)
        self.fr_n_weights_grad_ch[l_id].append(fr_n_grad_ch)

    def dump_data(self):
        with open(self.save_fname, 'wb') as fp:
            pickle.dump(self.iterations, fp)
            pickle.dump(self.fr_n_weights, fp)
            pickle.dump(self.fr_n_weights_ch, fp)
            pickle.dump(self.fr_n_weights_grad_ch, fp)
        return



def plot_weights(W_list, save_dir=None, suffix=None, normalize=False):
    '''
    Plots receptive fields given by weight matrices in W_list.

    Arguments:
        W_list (list of ndarrays) : List of weight matrices to plot.
        save_dir (string)         : Directory in which to save the plot.
        suffix (string)           : Suffix to add to the end of the filename of the plot.
        normalize (bool)          : Whether to normalize each receptive field. If True,
                                    the vmin and vmax of each receptive field subplot will
                                    be independent from the vmin and vmax of the other subplots.
    '''

    def prime_factors(n):
        # Get all prime factors of a number n.
        factors = []
        lastresult = n
        if n == 1: # 1 is a special case
            return [1]
        while 1:
            if lastresult == 1:
                break
            c = 2
            while 1:
                if lastresult % c == 0:
                    break
                c += 1
            factors.append(c)
            lastresult /= c
        print("Factors of %d: %s" % (n, str(factors)))
        return factors

    def find_closest_divisors(n):
        # Find divisors of a number n that are closest to its square root.
        a_max = np.floor(np.sqrt(n))
        if n % a_max == 0:
            a = a_max
            b = n/a
        else:
            p_fs = prime_factors(n)
            candidates = np.array([1])
            for i in xrange(len(p_fs)):
                f = p_fs[i]
                candidates = np.union1d(candidates, f*candidates)
                candidates[candidates > a_max] = 0
            a = candidates.max()
            b = n/a
        print("Closest divisors of %d: %s" % (n, str((int(b), int(a)))))
        return (int(a), int(b))

    plt.close('all')

    fig = plt.figure(figsize=(18, 9))

    M = len(W_list)

    n = [W.shape[0] for W in W_list]
    n_in = W_list[0].shape[-1]

    print(M, n)

    grid_specs = [0]*M
    axes = [ [0]*i for i in n ]

    max_Ws = [ np.amax(W) for W in W_list ]

    min_Ws = [ np.amin(W) for W in W_list ]

    W_sds = [ np.std(W) for W in W_list ]
    W_avgs = [ np.mean(W) for W in W_list ]

    for m in xrange(M):
        print("Layer {0} | W_avg: {1:.6f}, W_sd: {2:.6f}.".format(m, np.mean(W_list[m]), np.std(W_list[m])))

    for m in xrange(M):
        if m == 0:
            img_Bims = find_closest_divisors(n_in)
        else:
            img_Bims = grid_dims

        grid_dims = find_closest_divisors(n[m])
        grid_dims = (grid_dims[1], grid_dims[0]) # tanspose grid dimensions, to better fit the space

        grid_specs[m] = gs.GridSpec(grid_dims[0], grid_dims[1])

        for k in xrange(n[m]):
            row = k // grid_dims[1]
            col = k - row*grid_dims[1]

            axes[m][k] = fig.add_subplot(grid_specs[m][row, col])
            if normalize:
                heatmap = axes[m][k].imshow(W_list[m][k].reshape(img_Bims).T, interpolation="nearest", cmap=weight_cmap)
            else:
                heatmap = axes[m][k].imshow(W_list[m][k].reshape(img_Bims).T, interpolation="nearest", vmin=W_avgs[m] - 3.465*W_sds[m], vmax=W_avgs[m] + 3.465*W_sds[m], cmap=weight_cmap)
            axes[m][k].set_xticklabels([])
            axes[m][k].set_yticklabels([])

            axes[m][k].tick_params(axis='both',  # changes apply to the x-axis
                                   which='both', # both major and minor ticks are affected
                                   bottom='off', # ticks along the bottom edge are off
                                   top='off',    # ticks along the top edge are off
                                   left='off',   # ticks along the left edge are off
                                   right='off')  # ticks along the right edge are off

            if m == M-1 and k == 0:
                plt.colorbar(heatmap)

        grid_specs[m].update(left=float(m)/M,
                             right=(m+1.0)/M,
                             hspace=1.0/(grid_dims[0]),
                             wspace=0.05,
                             bottom=0.02,
                             top=0.98)

    if save_dir != None:
        if suffix != None:
            plt.savefig(save_dir + 'weights' + suffix + '.png')
        else:
            plt.savefig(save_dir + 'weights.png')
    else:
        plt.show()
