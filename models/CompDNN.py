# encoding=utf8

from __future__ import print_function
import numpy as np
import copy
import datetime
import os
import pdb
import sys
import time
import shutil
import json
from scipy.special import expit

from sklearn.utils import shuffle
from sklearn.model_selection import StratifiedKFold

from utils.helpers import to_one_hot, prep_data_guergiev, get_data, load_guergiev_params
from utils.logger import CompDNN_Logger, Weight_CompDNN_Logger


def sigma(x):
    return expit(x)

def deriv_sigma(x):
    return expit(x)*(1.0 - expit(x))

def kappa(x, tau_L, tau_s):
    return (np.exp(-x/tau_L) - np.exp(-x/tau_s))/(tau_L - tau_s)

def get_kappas(n, tau_L, tau_s):
    return np.array([kappa(i+1, tau_L, tau_s) for i in range(n)])

def shuffle_arrays(*args):
    # Shuffle multiple arrays using the same random permutation.
    p = np.random.permutation(args[0].shape[1])
    results = (a[:, p] for a in args)
    return results


class CompDNN:
    def __init__(self, X, y, param_fname="logs/comp_dnn_base_params.json"):

        # Load in hyperparameters of the model
        self.p = load_guergiev_params(param_fname)

        n = [self.p["h_l_1"], self.p["h_l_2"],
             self.p["h_l_3"], self.p["h_l_4"], self.p["h_l_5"],
             self.p["h_l_6"]][:self.p["num_layers"]]
        self.n = tuple(n + [10])
        self.M = len(self.n)  # number of layers

        self.n_neurons_per_category = int(self.n[-1]/10)

        # Select data - use 20% as validation set
        idx_train, idx_valid = next(iter(StratifiedKFold(5, random_state=0).split(np.arange(len(X)), y)))
        X_train, X_valid, y_train, y_valid = (X[idx_train],
                                              X[idx_valid],
                                              y[idx_train],
                                              y[idx_valid])

        self.X_train, self.y_train = prep_data_guergiev(X_train, y_train)
        self.X_valid, self.y_valid = prep_data_guergiev(X_valid, y_valid)

        self.num_train = self.X_train.shape[1]
        self.num_valid = self.X_valid.shape[1]

        self.n_in = self.X_train.shape[0]  # input size
        self.n_out = self.n[-1]            # output size

        self.x_hist = np.zeros((self.n_in, self.p["mem"]))  # initialize input spike hist
        self.current_epoch = None  # current epoch of simulation

        # Initialize weights and layer instances
        self.init_weights()
        self.init_layers()



    def init_weights(self):
        if self.p["use_weight_optimization"]:
            # initial weight optimization parameters
            V_avg = 3  # average of dendritic potential
            V_sd = 3  # standard deviation of dendritic potential
            b_avg = 0.8  # desired average of bias
            b_sd = 0.001  # desired standard deviation of bias
            nu = self.p["lambda_max"]*0.25  # slope of linear region of activation fct
            V_sm = V_sd**2 + V_avg**2  # second moment of dendritic potential

        # initialize lists of weight matrices & bias vectors
        self.W = [0]*self.M
        self.b = [0]*self.M
        self.Y = [0]*(self.M-1)
        if self.p["use_feedback_bias"]:
            self.c = [0]*(self.M-1)

        for m in range(self.M-1, -1, -1):
            # get number of units in the layer below
            if m != 0:
                N = self.n[m-1]
            else:
                N = self.n_in

            # generate feedforward weights & biases
            if self.p["use_weight_optimization"]:
                # calc weight vars needed to get desired avg & sd of somatic p.
                W_avg = (V_avg - b_avg)/(nu*N*V_avg)
                W_sm = (V_sm + (nu**2)*(N - N**2)*(W_avg**2)*(V_avg**2) - 2*N*nu*b_avg*V_avg*W_avg - (b_avg**2))/(N*(nu**2)*V_sm)
                W_sd = np.sqrt(W_sm - W_avg**2)

                self.W[m] = W_avg + 3.465*W_sd*np.random.uniform(-1, 1, size=(self.n[m], N))
                self.b[m] = b_avg + 3.465*b_sd*np.random.uniform(-1, 1, size=(self.n[m], 1))
            else:
                self.W[m] = 0.1*np.random.uniform(-1, 1, size=(self.n[m], N))
                self.b[m] = 1.0*np.random.uniform(-1, 1, size=(self.n[m], 1))

            if m != 0:
                if self.p["use_broadcast"]:
                    if self.p["use_weight_optimization"]:
                        self.Y[m-1] = W_avg + 3.465*W_sd*np.random.uniform(-1, 1, size=(N, self.n[-1]))

                        if self.p["use_feedback_bias"]:
                            self.c[m-1] = b_avg + 3.465*b_sd*np.random.uniform(-1, 1, size=(N, 1))
                    else:
                        self.Y[m-1] = np.random.uniform(-1, 1, size=(N, self.n[-1]))

                        if self.p["use_feedback_bias"]:
                            self.c[m-1] = np.random.uniform(-1, 1, size=(N, 1))
                else:
                    if self.p["use_weight_optimization"]:
                        self.Y[m-1] = W_avg + 3.465*W_sd*np.random.uniform(-1, 1, size=(N, self.n[m]))

                        if self.p["use_feedback_bias"]:
                            self.c[m-1] = b_avg + 3.465*b_sd*np.random.uniform(-1, 1, size=(N, 1))
                    else:
                        self.Y[m-1] = np.random.uniform(-1, 1, size=(N, self.n[m]))

                        if self.p["use_feedback_bias"]:
                            self.c[m-1] = np.random.uniform(-1, 1, size=(N, 1))

        if self.p["use_symmetric_weights"] is True:  # enforce symm weights
            self.make_weights_symmetric()

        if self.p["use_sparse_feedback"]:
            # Dropout weights, increase magnitude of remaining to keep avg Volt
            self.Y_dropout_indices = [0]*(self.M-1)
            for m in range(self.M-1):
                self.Y_dropout_indices[m] = np.random.choice(len(self.Y[m].ravel()),
                                                             int(0.8*len(self.Y[m].ravel())), False)
                self.Y[m].ravel()[self.Y_dropout_indices[m]] = 0
                self.Y[m] *= 5

    def make_weights_symmetric(self):
        # Feedback weights = Transposes feedforward weights
        if self.p["use_broadcast"]:
            for m in range(self.M-2, -1, -1):
                # make a copy if we're altering the feedback weights after
                if self.p["use_sparse_feedback"]:
                    W_above = self.W[m+1].T.copy()
                else:
                    W_above = self.W[m+1].T

                if m == self.M - 2:
                    # for final hidden l - feedforward weights of output l
                    if self.p["noisy_symmetric_weights"]:
                        self.Y[m] = W_above + np.random.normal(0, 0.05,
                                                               size=W_above.shape)
                    else:
                        self.Y[m] = W_above
                else:
                    # for other hidden l - prod of all feedf weights downstream
                    if self.p["noisy_symmetric_weights"]:
                        self.Y[m] = np.dot(W_above + np.random.normal(0, 0.05, size=W_above.shape), self.Y[m+1])
                    else:
                        self.Y[m] = np.dot(W_above, self.Y[m+1])
        else:
            for m in range(self.M-2, -1, -1):
                # make a copy if we're altering the feedback weights after
                if self.p["use_sparse_feedback"]:
                    W_above = self.W[m+1].T.copy()
                else:
                    W_above = self.W[m+1].T

                # use feedforward weights of the layer downstream
                if self.p["noisy_symmetric_weights"]:
                    self.Y[m] = W_above + np.random.normal(0, 0.05)
                else:
                    self.Y[m] = W_above

    def init_layers(self):
        # Initialize layers list and create all layers
        self.l = []
        if self.M == 1:
            self.l.append(finalLayer(net=self, m=-1, f_input_size=self.n_in,
                                     params=self.p))
        else:
            if self.p["use_broadcast"]:
                self.l.append(hiddenLayer(net=self, m=0,
                                          f_input_size=self.n_in,
                                          b_input_size=self.n[-1],
                                          params=self.p))
                for m in range(1, self.M-1):
                    self.l.append(hiddenLayer(net=self, m=m,
                                              f_input_size=self.n[m-1],
                                              b_input_size=self.n[-1],
                                              params=self.p))
            else:
                self.l.append(hiddenLayer(net=self, m=0,
                                          f_input_size=self.n_in,
                                          b_input_size=self.n[1],
                                          params=self.p))
                for m in range(1, self.M-1):
                    self.l.append(hiddenLayer(net=self, m=m,
                                              f_input_size=self.n[m-1],
                                              b_input_size=self.n[m+1],
                                              params=self.p))
            self.l.append(finalLayer(net=self, m=self.M-1,
                                     f_input_size=self.n[-2],
                                     params=self.p))

    def out_f(self, training=False):
        # Perform a forward phase pass through the network.
        if self.p["use_spiking_feedforward"]:
            x = self.x_hist
        else:
            x = self.x

        if self.M == 1:
            self.l[0].out_f(x, None)
        else:
            if self.p["use_broadcast"]:
                if self.p["use_spiking_feedback"]:
                    self.l[0].out_f(x, self.l[-1].S_hist)

                    for m in range(1, self.M-1):
                        if self.p["use_spiking_feedforward"]:
                            self.l[m].out_f(self.l[m-1].S_hist,
                                            self.l[-1].S_hist)
                        else:
                            self.l[m].out_f(self.l[m-1].lambda_C,
                                            self.l[-1].S_hist)

                    if self.p["use_spiking_feedforward"]:
                        self.l[-1].out_f(self.l[-2].S_hist, None)
                    else:
                        self.l[-1].out_f(self.l[-2].lambda_C, None)
                else:
                    self.l[0].out_f(x, self.l[-1].lambda_C)

                    for m in range(1, self.M-1):
                        if self.p["use_spiking_feedforward"]:
                            self.l[m].out_f(self.l[m-1].S_hist,
                                            self.l[-1].lambda_C)
                        else:
                            self.l[m].out_f(self.l[m-1].lambda_C,
                                            self.l[-1].lambda_C)

                    if self.p["use_spiking_feedforward"]:
                        self.l[-1].out_f(self.l[-2].S_hist, None)
                    else:
                        self.l[-1].out_f(self.l[-2].lambda_C, None)
            else:
                if self.p["use_spiking_feedback"]:
                    self.l[0].out_f(x, self.l[1].S_hist)

                    for m in range(1, self.M-1):
                        if self.p["use_spiking_feedforward"]:
                            self.l[m].out_f(self.l[m-1].S_hist,
                                            self.l[m+1].S_hist)
                        else:
                            self.l[m].out_f(self.l[m-1].lambda_C,
                                            self.l[m+1].S_hist)

                    if self.p["use_spiking_feedforward"]:
                        self.l[-1].out_f(self.l[-2].S_hist, None)
                    else:
                        self.l[-1].out_f(self.l[-2].lambda_C, None)
                else:
                    self.l[0].out_f(x, self.l[1].lambda_C)

                    for m in range(1, self.M-1):
                        if self.p["use_spiking_feedforward"]:
                            self.l[m].out_f(self.l[m-1].S_hist,
                                            self.l[m+1].lambda_C)
                        else:
                            self.l[m].out_f(self.l[m-1].lambda_C,
                                            self.l[m+1].lambda_C)

                    if self.p["use_spiking_feedforward"]:
                        self.l[-1].out_f(self.l[-2].S_hist, None)
                    else:
                        self.l[-1].out_f(self.l[-2].lambda_C, None)

    def out_t(self):
        # Perform a target phase pass through net - target introduced at top l
        if self.p["use_spiking_feedforward"]:
            x = self.x_hist
        else:
            x = self.x

        if self.M == 1:
            self.l[0].out_t(x, self.t)
        else:
            if self.p["use_broadcast"]:
                if self.p["use_spiking_feedback"]:
                    self.l[0].out_t(x, self.l[-1].S_hist)

                    for m in range(1, self.M-1):
                        if self.p["use_spiking_feedforward"]:
                            self.l[m].out_t(self.l[m-1].S_hist,
                                            self.l[-1].S_hist)
                        else:
                            self.l[m].out_t(self.l[m-1].lambda_C,
                                            self.l[-1].S_hist)

                    if self.p["use_spiking_feedforward"]:
                        self.l[-1].out_t(self.l[-2].S_hist, self.t)
                    else:
                        self.l[-1].out_t(self.l[-2].lambda_C, self.t)
                else:
                    self.l[0].out_t(x, self.l[-1].lambda_C)

                    for m in range(1, self.M-1):
                        if self.p["use_spiking_feedforward"]:
                            self.l[m].out_t(self.l[m-1].S_hist,
                                            self.l[-1].lambda_C)
                        else:
                            self.l[m].out_t(self.l[m-1].lambda_C,
                                            self.l[-1].lambda_C)

                    if self.p["use_spiking_feedforward"]:
                        self.l[-1].out_t(self.l[-2].S_hist, self.t)
                    else:
                        self.l[-1].out_t(self.l[-2].lambda_C, self.t)
            else:
                if self.p["use_spiking_feedback"]:
                    self.l[0].out_t(x, self.l[1].S_hist)

                    for m in range(1, self.M-1):
                        if self.p["use_spiking_feedforward"]:
                            self.l[m].out_t(self.l[m-1].S_hist,
                                            self.l[m+1].S_hist)
                        else:
                            self.l[m].out_t(self.l[m-1].lambda_C,
                                            self.l[m+1].S_hist)

                    if self.p["use_spiking_feedforward"]:
                        self.l[-1].out_t(self.l[-2].S_hist, self.t)
                    else:
                        self.l[-1].out_t(self.l[-2].lambda_C, self.t)
                else:
                    self.l[0].out_t(x, self.l[1].lambda_C)

                    for m in range(1, self.M-1):
                        if self.p["use_spiking_feedforward"]:
                            self.l[m].out_t(self.l[m-1].S_hist,
                                            self.l[m+1].lambda_C)
                        else:
                            self.l[m].out_t(self.l[m-1].lambda_C,
                                            self.l[m+1].lambda_C)

                    if self.p["use_spiking_feedforward"]:
                        self.l[-1].out_t(self.l[-2].S_hist, self.t)
                    else:
                        self.l[-1].out_t(self.l[-2].lambda_C, self.t)

    def f_phase(self, x, t, training_num, training=False):
        '''
        Perform a forward phase.

        Arguments:
            x (ndarray)        : Input array of size (X, 1) where X is the size of the input, eg. (784, 1).
            t (ndarray)        : Target array of size (T, 1) where T is the size of the target, eg. (10, 1).
            training_num (int) : Number (from start of the epoch) of the training example being shown.
            training (bool)    : Whether the network is in training (True) or testing (False) mode.
        '''

        for time in range(self.p["l_f_phase"]):
            # update input spike history
            self.x_hist = np.concatenate([self.x_hist[:, 1:], np.random.poisson(x)], axis=-1)

            # do a forward pass
            self.out_f(training=training)

        for m in range(self.M-2, -1, -1):
            plateau_indices = np.arange(self.n[m])
            # calculate plateau potentials for hidden layer neurons
            self.l[m].plateau_f(plateau_indices=plateau_indices)

        for m in range(self.M-1, -1, -1):
            self.l[m].calc_averages(phase="forward")

    def t_phase(self, x, t, training_num):
        '''
        Perform a target phase.

        Arguments:
            x (ndarray)        : Input array of size (X, 1) where X is the size of the input, eg. (784, 1).
            t (ndarray)        : Target array of size (T, 1) where T is the size of the target, eg. (10, 1).
            training_num (int) : Number (from start of the epoch) of the training example being shown.
        '''

        for time in range(self.p["l_t_phase"]):
            # update input history
            self.x_hist = np.concatenate([self.x_hist[:, 1:],
                                          np.random.poisson(x)], axis=-1)

            # do a target pass
            self.out_t()

        for m in range(self.M-2, -1, -1):
            plateau_indices = np.arange(self.n[m])

            # calculate plateau potentials for hidden layer neurons
            self.l[m].plateau_t(plateau_indices=plateau_indices)

        for m in range(self.M-1, -1, -1):
            # calculate averages
            self.l[m].calc_averages(phase="target")

            if self.p["update_feedback_weights"] and m < self.M-1:
                # update feedback weights
                self.l[m].update_Y()

            # update weights
            self.l[m].update_W()

        self.loss = ((self.l[-1].average_lambda_C_t - self.p["lambda_max"]*sigma(self.l[-1].average_C_f)) ** 2).mean()

        for m in range(self.M-1, -1, -1):
            # reset averages
            self.l[m].average_C_f *= 0
            self.l[m].average_C_t *= 0
            self.l[m].average_PSP_B_f *= 0

            if m == self.M-1:
                self.l[m].average_lambda_C_f *= 0
                self.l[m].average_lambda_C_t *= 0
            else:
                self.l[m].average_A_f *= 0
                self.l[m].average_A_t *= 0
                self.l[m].average_lambda_C_f *= 0
                if self.p["update_feedback_weights"]:
                    self.l[m].average_PSP_A_f *= 0

        if self.p["use_symmetric_weights"]:
            # make feedback weights symmetric to new feedforward weights
            self.make_weights_symmetric()

        if self.p["use_sparse_feedback"] and (self.p["use_symmetric_weights"] or self.p["update_feedback_weights"]):
            for m in range(self.M-1):
                # zero out the inactive weights
                self.Y[m].ravel()[self.Y_dropout_indices[m]] = 0

                # increase magnitude of surviving weights
                self.Y[m] *= 5

    def train(self, n_epochs, log_freq, verbose, logging, dataset):
        '''
        Train the network. Checkpoints will be saved at the end of every epoch if save_simulation is True.

        Arguments:
            f_etas (tuple)              : Learning rates for each layer's feedforward weights, eg. (0.21, 0.21).
            b_etas (tuple/None)         : Learning rates for each layer's feedback weights.
                                          If None, no backward weight updates occur.
            n_epochs (int)              : Number of epochs of training.
            save_simulation (bool)      : Whether to save data from this simulation.
            simulations_folder (string) : Name of the parent folder that can contain data from multiple simulations.
            folder_name (string)        : Name of the subfolder in the parent folder that will contain data from this simulation.
            overwrite (bool)            : Whether to overwrite the folder given by folder_name if it already exists.
            simulation_notes (string)   : Notes about this simulation to save in the parameters text file that will be generated.
            current_epoch (int/None)    : The current epoch of this simulation. This sets the value of the network's current_epoch attribute.
                                          If 0, this is a new simulation.
                                          If > 0, this is a continuation of a previously-started simulation.
                                          If None, the current value of the network's 'current_epoch' attribute
                                          determines the state of the simulation.
        '''

        if logging:
            logger = CompDNN_Logger("logs", "/" + dataset + "_guergiev.pkl")
            wlogger = Weight_CompDNN_Logger("logs",
                                            "/" + dataset + "_guergiev_weights.pkl",
                                            list(np.arange(self.M)))

        if self.p["b_etas"] == 0 and self.p["update_feedback_weights"]:
            raise ValueError("No feedback learning rates provided, but 'update_feedback_weights' is True.")


        self.current_epoch = 0
        continuing = False

        l_f_phases = np.zeros(n_epochs*self.num_train) + self.p["l_f_phase"]
        l_t_phases = np.zeros(n_epochs*self.num_train) + self.p["l_t_phase"]

        # get array of total length of both phases for all training examples
        l_phases_tot = l_f_phases + l_t_phases

        # set learning rate instance variables
        self.f_etas = tuple(self.M*[self.p["f_etas"]])
        self.b_etas = tuple(self.M*[self.p["b_etas"]])

        self.losses = np.zeros(n_epochs*self.num_train)

        self.plateau_times_full = [np.zeros((n_epochs*2*self.num_train, self.n[m])) for m in range(self.M)]
        # initialize input spike history
        self.x_hist = np.zeros((self.n_in, self.p["mem"]))

        # start time used for timing how long each 1000 examples take
        start_time = None

        for k in range(n_epochs):
            # shuffle the training data
            self.X_train, self.y_train = shuffle_arrays(self.X_train, self.y_train)

            # generate arrays of forward phase plateau potential times (time until plateau potential from start of forward phase) for individual neurons
            self.plateau_times_f = [np.zeros((self.num_train, self.n[m])) + l_f_phases[k*self.num_train:(k+1)*self.num_train, np.newaxis] - 1 for m in range(self.M)]

            # generate arrays of target phase plateau potential times (time until plateau potential from start of target phase) for individual neurons

            self.plateau_times_t = [np.zeros((self.num_train, self.n[m])) + l_t_phases[k*self.num_train:(k+1)*self.num_train, np.newaxis] - 1 for m in range(self.M)]

            for n in range(self.num_train):
                # set start time
                if start_time == None:
                    start_time = time.time()

                l_phases_tot = self.p["l_f_phase"] + self.p["l_t_phase"]

                # get training example data
                self.x = self.p["lambda_max"]*self.X_train[:, n][:, np.newaxis]
                self.t = self.y_train[:, n][:, np.newaxis]

                # do forward & target phases
                self.f_phase(self.x, None, n, training=True)
                self.t_phase(self.x,
                             self.t.repeat(self.n_neurons_per_category,
                                           axis=0), n)


                self.losses[k*self.num_train + n] = self.loss

                # Sum together gradients over a 100 sample batch
                if ((n+1) % log_freq == log_freq - 100) and logging:
                    weight_gradients = {key: None for key in range(self.M)}
                    for l_id in range(len(self.l)):
                        weight_gradients[l_id] = self.l[l_id].delta_W.copy()
                if ((n+1) % log_freq  > log_freq - 100) and logging:
                    for l_id in range(len(self.l)):
                        weight_gradients[l_id] += self.l[l_id].delta_W.copy()

                if (n+1) % log_freq == 0 and logging:
                    template = "{}| epoch {:>2}| batch {:>2}/{:>2}|"
                    template += " acc: {:.4f}| loss: {:.4f}| time: {:.2f}"
                    if n != self.num_train - 1:
                        train_acc, train_loss = self.get_test_error(self.X_train, self.y_train)
                        end_time = time.time()
                        time_elapsed = end_time - start_time

                        if verbose:
                            print(template.format("Train", self.current_epoch + 1, n+1,
                                                  self.num_train, train_acc, train_loss, time_elapsed))

                        start_time = time.time()
                        test_acc, test_loss = self.get_test_error(self.X_valid, self.y_valid)
                        end_time = time.time()
                        time_elapsed = end_time - start_time
                        start_time = None

                        if verbose:
                            print(template.format("Valid", self.current_epoch + 1, n+1,
                                                  self.num_train, test_acc, test_loss, time_elapsed))

                            print('-' * 73)

                        if logging:
                            logger.update(self.current_epoch*self.num_train + n,
                                          train_loss, test_loss,
                                          train_acc, test_acc)

                            wlogger.update(self.current_epoch*self.num_train + n,
                                           self.W, weight_gradients)

            # update latest epoch counter
            self.current_epoch += 1

    def get_test_error(self, X, y):
        '''
        Test the network's current weights on the test set.
        The network's layers are copied
        and restored to their previous state after testing.

        Arguments:
            n_test (int) : The number of test examples to use.
        '''
        # save old length of forward phase
        old_l_f_phase = self.p["l_f_phase"]
        # set new length of forward phase
        self.p["l_f_phase"] = self.p["l_f_phase_test"]

        # save old integration time
        old_integration_time = self.p["integration_time"]
        # set new integration time
        self.p["integration_time"] = self.p["integration_time_test"]

        old_x_hist = self.x_hist

        # initialize count of correct classifications/loss (cross-entropy)
        num_correct = 0
        cross_ent_loss = 0

        # shuffle testing data
        X_temp, y_temp = shuffle_arrays(X, y)
        n_test = X.shape[1]

        digits = np.arange(10)

        # create new integration recording variables
        for m in range(self.M):
            self.l[m].create_integration_vars()

        for n in range(n_test):
            # clear all layer variables
            for m in range(self.M):
                self.l[m].clear_vars()

            # clear input spike history
            self.x_hist *= 0

            # get testing example data
            self.x = self.p["lambda_max"]*X_temp[:, n][:, np.newaxis]
            self.t = y_temp[:, n][:, np.newaxis]

            # do a forward phase & get the unit with maximum average somatic potential
            self.f_phase(self.x,
                         self.t.repeat(self.n_neurons_per_category, axis=0),
                         None, training=False)

            sel_num = np.argmax(np.mean(self.l[-1].average_C_f.reshape(-1, self.n_neurons_per_category), axis=-1))

            # get the target number from testing example data
            target_num = np.dot(digits, self.t)

            # increment correct classification counter if they match
            if sel_num == target_num:
                num_correct += 1
            else:
                prob_corr_num = np.mean(self.l[-1].average_C_f.reshape(-1, self.n_neurons_per_category), axis=-1)[int(target_num)]
                prob_corr_num = np.clip(prob_corr_num, 1e-12, 1. - 1e-12)
                cross_ent_loss += -np.log(prob_corr_num)

        # calculate percent error
        err_rate = (1.0 - float(num_correct)/n_test)*100.0

        if old_x_hist is not None:
            self.x_hist = old_x_hist

        self.p["integration_time"] = old_integration_time
        self.p["l_f_phase"] = old_l_f_phase

        # create new integration recording variables
        for m in range(self.M):
            self.l[m].create_integration_vars()

        # clear all layer variables
        for m in range(self.M):
            self.l[m].clear_vars()

        return 1-err_rate/100, cross_ent_loss/n_test

    def train_slim(self, n_epochs):
        if self.p["b_etas"] == 0 and self.p["update_feedback_weights"]:
            raise ValueError("No feedback learning rates provided, but 'update_feedback_weights' is True.")


        self.current_epoch = 0
        continuing = False

        l_f_phases = np.zeros(n_epochs*self.num_train) + self.p["l_f_phase"]
        l_t_phases = np.zeros(n_epochs*self.num_train) + self.p["l_t_phase"]

        # get array of total length of both phases for all training examples
        l_phases_tot = l_f_phases + l_t_phases

        # set learning rate instance variables
        self.f_etas = tuple(self.M*[self.p["f_etas"]])
        self.b_etas = tuple(self.M*[self.p["b_etas"]])


        self.losses = np.zeros(n_epochs*self.num_train)

        self.plateau_times_full = [np.zeros((n_epochs*2*self.num_train, self.n[m])) for m in range(self.M)]
        # initialize input spike history
        self.x_hist = np.zeros((self.n_in, self.p["mem"]))

        # start time used for timing how long each 1000 examples take


        for k in range(n_epochs):
            # shuffle the training data
            self.X_train, self.y_train = shuffle_arrays(self.X_train, self.y_train)

            # generate arrays of forward phase plateau potential times (time until plateau potential from start of forward phase) for individual neurons

            self.plateau_times_f = [np.zeros((self.num_train, self.n[m])) + l_f_phases[k*self.num_train:(k+1)*self.num_train, np.newaxis] - 1 for m in range(self.M)]

            # generate arrays of target phase plateau potential times (time until plateau potential from start of target phase) for individual neurons
            self.plateau_times_t = [np.zeros((self.num_train, self.n[m])) + l_t_phases[k*self.num_train:(k+1)*self.num_train, np.newaxis] - 1 for m in range(self.M)]

            for n in range(self.num_train):
                l_phases_tot = self.p["l_f_phase"] + self.p["l_t_phase"]

                # get training example data
                self.x = self.p["lambda_max"]*self.X_train[:, n][:, np.newaxis]
                self.t = self.y_train[:, n][:, np.newaxis]

                # do forward & target phases
                self.f_phase(self.x, None, n, training=True)
                self.t_phase(self.x,
                             self.t.repeat(self.n_neurons_per_category,
                                           axis=0), n)


                self.losses[k*self.num_train + n] = self.loss

            # update latest epoch counter
            self.current_epoch += 1


class Layer:
    def __init__(self, net, m):
        '''
        Initialize the layer.

        Arguments:
            net (Network) : The network that the layer belongs to.
            m (int)       : The layer number, eg. m = 0 for the first layer.
        '''

        self.net = net
        self.m = m
        self.size = self.net.n[m]

    def spike(self):
        '''
        Generate Poisson spikes based on the firing rates of the neurons.
        '''

        self.S_hist = np.concatenate([self.S_hist[:, 1:],
                                      np.random.poisson(self.lambda_C)],
                                     axis=-1)


class hiddenLayer(Layer):
    def __init__(self, net, m, f_input_size, b_input_size, params):
        '''
        Initialize the hidden layer.

        Arguments:
            net (Network)      : The network that the layer belongs to.
            m (int)            : The layer number, eg. m = 0 for the first hidden layer.
            f_input_size (int) : The size of feedforward input, eg. 784 for MNIST input.
            b_input_size (int) : The size of feedback input. This is the same as the
                                 the number of units in the next layer.
        '''

        Layer.__init__(self, net, m)

        self.params = params
        # Kernel filtering initialize kappas array
        self.kappas = np.flipud(get_kappas(self.params["mem"],
                                           self.params["tau_L"],
                                           self.params["tau_s"]))[:, np.newaxis]
        self.f_input_size = f_input_size
        self.b_input_size = b_input_size

        self.A = np.zeros((self.size, 1))
        self.B = np.zeros((self.size, 1))
        self.C = np.zeros((self.size, 1))
        self.lambda_C = np.zeros((self.size, 1))

        self.S_hist = np.zeros((self.size, self.params["mem"]), dtype=np.int8)

        self.E = np.zeros((self.size, 1))
        self.delta_W = np.zeros(self.net.W[self.m].shape)
        self.delta_Y = np.zeros(self.net.Y[self.m].shape)
        self.delta_b = np.zeros((self.size, 1))

        self.average_C_f = np.zeros((self.size, 1))
        self.average_C_t = np.zeros((self.size, 1))
        self.average_A_f = np.zeros((self.size, 1))
        self.average_A_t = np.zeros((self.size, 1))
        self.average_lambda_C_f = np.zeros((self.size, 1))
        self.average_PSP_B_f = np.zeros((self.f_input_size, 1))
        if params["update_feedback_weights"]:
            self.average_PSP_A_f = np.zeros((self.b_input_size, 1))

        self.alpha_f = np.zeros((self.size, 1))
        self.alpha_t = np.zeros((self.size, 1))

        # set integration counter
        self.integration_counter = 0

        # create integration variables
        self.create_integration_vars()

    def create_integration_vars(self):
        self.A_hist = np.zeros((self.size, self.params["integration_time"]))
        self.PSP_A_hist = np.zeros((self.b_input_size, self.params["integration_time"]))
        self.PSP_B_hist = np.zeros((self.f_input_size, self.params["integration_time"]))
        self.C_hist = np.zeros((self.size, self.params["integration_time"]))
        self.lambda_C_hist = np.zeros((self.size,
                                       self.params["integration_time"]))

    def clear_vars(self):
        '''
        Clear all layer variables.
        '''

        self.A *= 0
        self.B *= 0
        self.C *= 0
        self.lambda_C *= 0

        self.S_hist *= 0
        self.A_hist *= 0
        self.PSP_A_hist *= 0
        self.PSP_B_hist *= 0
        self.C_hist *= 0
        self.lambda_C_hist *= 0

        self.E *= 0
        self.delta_W *= 0
        self.delta_Y *= 0
        self.delta_b *= 0

        self.average_C_f *= 0
        self.average_C_t *= 0
        self.average_A_f *= 0
        self.average_A_t *= 0
        self.average_lambda_C_f *= 0
        self.average_PSP_B_f *= 0
        if self.params["update_feedback_weights"]:
            self.average_PSP_A_f *= 0

        self.alpha_f *= 0
        self.alpha_t *= 0

        self.integration_counter = 0

    def update_W(self):
        '''
        Update feedforward weights.
        '''

        self.E = (self.alpha_t - self.alpha_f)*-self.params["k_B"]*self.params["lambda_max"]*deriv_sigma(self.average_C_f)

        self.delta_W = np.dot(self.E, self.average_PSP_B_f.T)
        self.net.W[self.m] += -self.net.f_etas[self.m]*self.params["P_hidden"]*self.delta_W

        self.delta_b = self.E
        self.net.b[self.m] += -self.net.f_etas[self.m]*self.params["P_hidden"]*self.delta_b

    def update_Y(self):
        '''
        Update feedback weights.
        '''

        E_inv = (self.params["lambda_max"]*sigma(self.average_C_f) - self.alpha_f)*-deriv_sigma(self.average_A_f)

        self.delta_Y        = np.dot(E_inv, self.average_PSP_A_f.T)
        self.net.Y[self.m] += -self.net.b_etas[self.m]*self.delta_Y

        if self.params["use_feedback_bias"]:
            self.delta_c        = E_inv
            self.net.c[self.m] += -self.net.b_etas[self.m]*self.delta_c

    def update_A(self, b_input):
        '''
        Update apical potentials.

        Arguments:
            b_input (ndarray) : Feedback input.
        '''

        if self.params["use_spiking_feedback"]:
            self.PSP_A = np.dot(b_input, self.kappas)
        else:
            self.PSP_A = b_input

        self.PSP_A_hist[:, self.integration_counter % self.params["integration_time"]] = self.PSP_A[:, 0]

        if self.params["use_feedback_bias"]:
            self.A = np.dot(self.net.Y[self.m], self.PSP_A) + self.net.c[self.m]
        else:
            self.A = np.dot(self.net.Y[self.m], self.PSP_A)
        self.A_hist[:, self.integration_counter % self.params["integration_time"]] = self.A[:, 0]

    def update_B(self, f_input):
        '''
        Update basal potentials.

        Arguments:
            f_input (ndarray) : Feedforward input.
        '''

        if self.params["use_spiking_feedforward"]:
            self.PSP_B = np.dot(f_input, self.kappas)
        else:
            self.PSP_B = f_input

        self.PSP_B_hist[:, self.integration_counter % self.params["integration_time"]] = self.PSP_B[:, 0]

        self.B = np.dot(self.net.W[self.m], self.PSP_B) + self.net.b[self.m]

    def update_C(self):
        '''
        Update somatic potentials & calculate firing rates.
        '''

        if self.params["use_conductances"]:
            if self.params["use_apical_conductance"]:
                self.C_dot = -self.params["g_L"]*self.C + self.params["g_B"]*(self.B - self.C) + self.params["g_A"]*(self.A - self.C)
            else:
                self.C_dot = -self.params["g_L"]*self.C + self.params["g_B"]*(self.B - self.C)
            self.C += self.C_dot*self.params["dt"]
        else:
            self.C = self.params["k_B"]*self.B

        self.C_hist[:, self.integration_counter % self.params["integration_time"]] = self.C[:, 0]

        self.lambda_C = self.params["lambda_max"]*sigma(self.C)
        self.lambda_C_hist[:, self.integration_counter % self.params["integration_time"]] = self.lambda_C[:, 0]

    def out_f(self, f_input, b_input):
        '''
        Perform a forward phase pass.

        Arguments:
            f_input (ndarray) : Feedforward input.
            b_input (ndarray) : Feedback input.
        '''

        self.update_B(f_input)
        self.update_A(b_input)
        self.update_C()
        self.spike()

        self.integration_counter = (self.integration_counter + 1) % self.params["integration_time"]

    def out_t(self, f_input, b_input):
        '''
        Perform a target phase pass.

        Arguments:
            f_input (ndarray) : Feedforward input.
            b_input (ndarray) : Feedback input.
        '''

        self.update_B(f_input)
        self.update_A(b_input)
        self.update_C()
        self.spike()

        self.integration_counter = (self.integration_counter + 1) % self.params["integration_time"]

    def plateau_f(self, plateau_indices):
        '''
        Calculate forward phase apical plateau potentials.

        Arguments:
            plateau_indices (ndarray) : Indices of neurons that are undergoing apical plateau potentials.
        '''

        # calculate average apical potentials for neurons undergoing plateau potentials
        self.average_A_f[plateau_indices] = np.mean(self.A_hist[plateau_indices], axis=-1)[:, np.newaxis]

        # calculate apical calcium spike nonlinearity
        self.alpha_f[plateau_indices] = sigma(self.average_A_f[plateau_indices])

    def plateau_t(self, plateau_indices):
        '''
        Calculate target phase apical plateau potentials.

        Arguments:
            plateau_indices (ndarray) : Indices of neurons that are undergoing apical plateau potentials.
        '''

        # calculate average apical potentials for neurons undergoing plateau potentials
        self.average_A_t[plateau_indices] = np.mean(self.A_hist[plateau_indices], axis=-1)[:, np.newaxis]

        # calculate apical calcium spike nonlinearity
        self.alpha_t[plateau_indices] = sigma(self.average_A_t[plateau_indices])

    def calc_averages(self, phase):
        '''
        Calculate averages of dynamic variables. This is done at the end of each
        forward & target phase.

        Arguments:
            phase (string) : Current phase of the network, "forward" or "target".
        '''

        if phase == "forward":
            self.average_C_f        = np.mean(self.C_hist, axis=-1)[:, np.newaxis]
            self.average_lambda_C_f = np.mean(self.lambda_C_hist, axis=-1)[:, np.newaxis]
            self.average_PSP_B_f    = np.mean(self.PSP_B_hist, axis=-1)[:, np.newaxis]

            if self.params["update_feedback_weights"]:
                self.average_PSP_A_f = np.mean(self.PSP_A_hist, axis=-1)[:, np.newaxis]
        elif phase == "target":
            self.average_C_t        = np.mean(self.C_hist, axis=-1)[:, np.newaxis]
            self.average_lambda_C_t = np.mean(self.lambda_C_hist, axis=-1)[:, np.newaxis]

            if self.params["update_feedback_weights"]:
                self.average_PSP_A_t = np.mean(self.PSP_A_hist, axis=-1)[:, np.newaxis]

"""
NOTE: In the paper, we denote the output layer's somatic & dendritic potentials
      as U and V. Here, we use C & B purely in order to simplify the code.
"""
class finalLayer(Layer):
    def __init__(self, net, m, f_input_size, params):
        '''
        Initialize the final layer.

        Arguments:
            net (Network)      : The network that the layer belongs to.
            m (int)            : The layer number, ie. m = M - 1 where M is the total number of layers.
            f_input_size (int) : The size of feedforward input. This is the same as the
                                 the number of units in the previous layer.
        '''

        Layer.__init__(self, net, m)

        self.params = params
        # Kernel filtering initialize kappas array
        self.kappas = np.flipud(get_kappas(self.params["mem"],
                                           self.params["tau_L"],
                                           self.params["tau_s"]))[:, np.newaxis]
        self.f_input_size = f_input_size

        self.B = np.zeros((self.size, 1))
        self.I = np.zeros((self.size, 1))
        self.C = np.zeros((self.size, 1))
        self.lambda_C = np.zeros((self.size, 1))

        self.S_hist = np.zeros((self.size, self.params["mem"]), dtype=np.int8)

        self.E = np.zeros((self.size, 1))
        self.delta_W = np.zeros(self.net.W[self.m].shape)
        self.delta_b = np.zeros((self.size, 1))

        self.average_C_f = np.zeros((self.size, 1))
        self.average_C_t = np.zeros((self.size, 1))
        self.average_lambda_C_f = np.zeros((self.size, 1))
        self.average_lambda_C_t = np.zeros((self.size, 1))
        self.average_PSP_B_f = np.zeros((self.f_input_size, 1))

        # set integration counter
        self.integration_counter = 0

        # create integration variables
        self.create_integration_vars()

    def create_integration_vars(self):
        self.PSP_B_hist = np.zeros((self.f_input_size,
                                    self.params["integration_time"]))
        self.C_hist = np.zeros((self.size, self.params["integration_time"]))
        self.lambda_C_hist = np.zeros((self.size,
                                       self.params["integration_time"]))

    def clear_vars(self):
        '''
        Clear all layer variables.
        '''

        self.B *= 0
        self.I *= 0
        self.C *= 0
        self.lambda_C *= 0

        self.S_hist *= 0
        self.PSP_B_hist *= 0
        self.C_hist *= 0
        self.lambda_C_hist *= 0

        self.E *= 0
        self.delta_W *= 0
        self.delta_b *= 0

        self.average_C_f *= 0
        self.average_C_t *= 0
        self.average_lambda_C_f *= 0
        self.average_lambda_C_t *= 0
        self.average_PSP_B_f *= 0

        self.integration_counter = 0

    def update_W(self):
        '''
        Update feedforward weights.
        '''

        self.E = (self.average_lambda_C_t - self.params["lambda_max"]*sigma(self.average_C_f))*-self.params["k_D"]*self.params["lambda_max"]*deriv_sigma(self.average_C_f)

        self.delta_W = np.dot(self.E, self.average_PSP_B_f.T)
        self.net.W[self.m] += -self.net.f_etas[self.m]*self.params["P_final"]*self.delta_W

        self.delta_b = self.E
        self.net.b[self.m] += -self.net.f_etas[self.m]*self.params["P_final"]*self.delta_b

    def update_B(self, f_input):
        '''
        Update basal potentials.

        Arguments:
            f_input (ndarray) : Feedforward input.
        '''

        if self.params["use_spiking_feedforward"]:
            self.PSP_B = np.dot(f_input, self.kappas)
        else:
            self.PSP_B = f_input

        self.PSP_B_hist[:, self.integration_counter % self.params["integration_time"]] = self.PSP_B[:, 0]

        self.B = np.dot(self.net.W[self.m], self.PSP_B) + self.net.b[self.m]

    def update_I(self, b_input=None):
        '''
        Update injected perisomatic currents.

        Arguments:
            b_input (ndarray) : Target input, eg. if the target label is 8,
                                b_input = np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 0]).
        '''

        if b_input is None:
            self.I *= 0
        else:
            g_E = b_input
            g_I = -g_E + 1
            if self.params["use_conductances"]:
                self.I = g_E*(self.params["E_E"] - self.C) + g_I*(self.params["E_I"] - self.C)
            else:
                self.k_D2 = self.params["g_D"]/(self.params["g_L"] + self.params["g_D"] + g_E + g_I)
                self.k_E = g_E/(self.params["g_L"] + self.params["g_D"] + g_E + g_I)
                self.k_I = g_I/(self.params["g_L"] + self.params["g_D"] + g_E + g_I)

    def update_C(self, phase):
        '''
        Update somatic potentials & calculate firing rates.

        Arguments:
            phase (string) : Current phase of the network, "forward" or "target".
        '''

        if self.params["use_conductances"]:
            if phase == "forward":
                self.C_dot = -self.params["g_L"]*self.C + self.params["g_D"]*(self.B - self.C)
            elif phase == "target":
                self.C_dot = -self.params["g_L"]*self.C + self.params["g_D"]*(self.B - self.C) + self.I
            self.C += self.C_dot*self.params["dt"]
        else:
            if phase == "forward":
                self.C = self.params["k_D"]*self.B
            elif phase == "target":
                self.C = self.k_D2*self.B + self.k_E*self.params["E_E"] + self.k_I*self.params["E_I"]

        self.C_hist[:, self.integration_counter % self.params["integration_time"]] = self.C[:, 0]

        self.lambda_C = self.params["lambda_max"]*sigma(self.C)
        self.lambda_C_hist[:, self.integration_counter % self.params["integration_time"]] = self.lambda_C[:, 0]

    def out_f(self, f_input, b_input):
        '''
        Perform a forward phase pass.

        Arguments:
            f_input (ndarray) : Feedforward input.
            b_input (ndarray) : Target input. b_input = None during this phase.
        '''

        self.update_B(f_input)
        self.update_I(b_input)
        self.update_C(phase="forward")
        self.spike()

        self.integration_counter = (self.integration_counter + 1) % self.params["integration_time"]

    def out_t(self, f_input, b_input):
        '''
        Perform a target phase pass.

        Arguments:
            f_input (ndarray) : Feedforward input.
            b_input (ndarray) : Target input.
        '''

        self.update_B(f_input)
        self.update_I(b_input)
        self.update_C(phase="target")
        self.spike()

        self.integration_counter = (self.integration_counter + 1) % self.params["integration_time"]

    def calc_averages(self, phase):
        '''
        Calculate averages of dynamic variables. This is done at the end of each
        forward & target phase.

        Arguments:
            phase (string) : Current phase of the network, "forward" or "target".
        '''

        if phase == "forward":
            self.average_C_f        = np.mean(self.C_hist, axis=-1)[:, np.newaxis]
            self.average_lambda_C_f = np.mean(self.lambda_C_hist, axis=-1)[:, np.newaxis]
            self.average_PSP_B_f    = np.mean(self.PSP_B_hist, axis=-1)[:, np.newaxis]
        elif phase == "target":
            self.average_C_t        = np.mean(self.C_hist, axis=-1)[:, np.newaxis]
            self.average_lambda_C_t = np.mean(self.lambda_C_hist, axis=-1)[:, np.newaxis]


def eval_comp_dnn(dataset, param_fname, num_epochs, k_fold, verbose=False):
    if verbose:
        params = load_guergiev_params(param_fname)
        print("Dataset: {}".format(dataset))
        print("Learning Rate Weights: {}".format(params["f_etas"]))
        print("Architecture of Cross-Validated Network:")
        for i in range(params["num_layers"]):
            l_str = "h_l_" + str(i+1)
            print("\t Layer {}: {} Units".format(i, params[l_str]))
        print("\t Layer {}: 10 Units".format(params["num_layers"]))
    # Initialize list to store cross_val accuracies
    scores = []
    # Load dataset
    X, y = get_data(70000, dataset)

    # Split original dataset into folds (return idx)
    kf = StratifiedKFold(n_splits=k_fold, random_state=0)
    kf.get_n_splits(X)
    counter = 1

    for sub_index, test_index in kf.split(X, y):
        X_sub, X_test = X[sub_index], X[test_index]
        y_sub, y_test = y[sub_index], y[test_index]

        # Instantiate the model with layersize and Logging directory
        comp_dnn_model = CompDNN(X_sub, y_sub, param_fname)
        comp_dnn_model.train_slim(num_epochs)
        # Compute accuracy on hold-out set
        X_test, y_test = prep_data_guergiev(X_test, y_test)
        score_temp, _ = comp_dnn_model.get_test_error(X_test, y_test)
        scores.append(score_temp)

        if verbose:
            print("Cross-Validation Score Fold {}: {}".format(counter,
                                                              score_temp))
            counter += 1
    return np.mean(scores)
