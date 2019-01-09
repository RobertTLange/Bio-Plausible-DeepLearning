# encoding=utf8

from __future__ import print_function
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs
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
from CompDNN_hyperparameters import *

def sigma(x):
    return expit(x)

def deriv_sigma(x):
    return expit(x)*(1.0 - expit(x))

def kappa(x):
    return (np.exp(-x/tau_L) - np.exp(-x/tau_s))/(tau_L - tau_s)

def get_kappas(n):
    return np.array([kappa(i+1) for i in xrange(n)])

# Kernel filtering initialize kappas array
kappas = np.flipud(get_kappas(mem))[:, np.newaxis]

def shuffle_arrays(*args):
    '''
    Shuffle multiple arrays using the same random permutation.

    Arguments:
        args (tuple of ndarrays) : Arrays to shuffle.

    Returns:
        results (tuple of ndarrays) : Shuffled arrays.
    '''

    p = np.random.permutation(args[0].shape[1])
    results = (a[:, p] for a in args)
    return results


class Network:
    def __init__(self, n, X_train, y_train, X_test, y_test):

        if type(n) == int:
            n = (n,)

        self.n = n           # layer sizes - eg. (500, 100, 10)
        self.M = len(self.n) # number of layers

        self.n_neurons_per_category = int(self.n[-1]/10)

        self.x_train, self.t_train, self.x_test, self.t_test = X_train, y_train, X_test, y_test

        self.n_in  = self.x_train.shape[0] # input size
        self.n_out = self.n[-1]            # output size

        # initialize input spike history
        self.x_hist = np.zeros((self.n_in, mem))

        self.current_epoch = None # current epoch of simulation

        self.init_weights()
        self.init_layers()

    def init_weights(self):
        if use_weight_optimization:
            # initial weight optimization parameters
            V_avg = 3                  # desired average of dendritic potential
            V_sd  = 3                  # desired standard deviation of dendritic potential
            b_avg = 0.8                # desired average of bias
            b_sd  = 0.001              # desired standard deviation of bias
            nu    = lambda_max*0.25    # slope of linear region of activation function
            V_sm  = V_sd**2 + V_avg**2 # second moment of dendritic potential

        # initialize lists of weight matrices & bias vectors
        self.W = [0]*self.M
        self.b = [0]*self.M
        self.Y = [0]*(self.M-1)
        if use_feedback_bias:
            self.c = [0]*(self.M-1)

        if use_sparse_feedback:
            # initialize list of indices of zeroed-out weights
            self.Y_dropout_indices = [0]*(self.M-1)

        for m in xrange(self.M-1, -1, -1):
            # get number of units in the layer below
            if m != 0:
                N = self.n[m-1]
            else:
                N = self.n_in

            # generate feedforward weights & biases
            if use_weight_optimization:
                # calculate weight variables needed to get desired average & standard deviations of somatic potentials
                W_avg = (V_avg - b_avg)/(nu*N*V_avg)
                W_sm  = (V_sm + (nu**2)*(N - N**2)*(W_avg**2)*(V_avg**2) - 2*N*nu*b_avg*V_avg*W_avg - (b_avg**2))/(N*(nu**2)*V_sm)
                W_sd  = np.sqrt(W_sm - W_avg**2)

                self.W[m] = W_avg + 3.465*W_sd*np.random.uniform(-1, 1, size=(self.n[m], N))
                self.b[m] = b_avg + 3.465*b_sd*np.random.uniform(-1, 1, size=(self.n[m], 1))
            else:
                self.W[m] = 0.1*np.random.uniform(-1, 1, size=(self.n[m], N))
                self.b[m] = 1.0*np.random.uniform(-1, 1, size=(self.n[m], 1))

            # generate feedback weights & biases; in the paper, we do not use feedback biases
            if m != 0:
                if use_broadcast:
                    if use_weight_optimization:
                        self.Y[m-1] = W_avg + 3.465*W_sd*np.random.uniform(-1, 1, size=(N, self.n[-1]))

                        if use_feedback_bias:
                            self.c[m-1] = b_avg + 3.465*b_sd*np.random.uniform(-1, 1, size=(N, 1))
                    else:
                        self.Y[m-1] = np.random.uniform(-1, 1, size=(N, self.n[-1]))

                        if use_feedback_bias:
                            self.c[m-1] = np.random.uniform(-1, 1, size=(N, 1))
                else:
                    if use_weight_optimization:
                         self.Y[m-1] = W_avg + 3.465*W_sd*np.random.uniform(-1, 1, size=(N, self.n[m]))

                         if use_feedback_bias:
                             self.c[m-1] = b_avg + 3.465*b_sd*np.random.uniform(-1, 1, size=(N, 1))
                    else:
                        self.Y[m-1] = np.random.uniform(-1, 1, size=(N, self.n[m]))

                        if use_feedback_bias:
                            self.c[m-1] = np.random.uniform(-1, 1, size=(N, 1))

        if use_symmetric_weights == True:
            # enforce symmetric weights
            self.make_weights_symmetric()

        if use_sparse_feedback:
            # randomly zero out 80% of weights, increase magnitude of surviving weights to keep desired average voltages
            for m in xrange(self.M-1):
                self.Y_dropout_indices[m] = np.random.choice(len(self.Y[m].ravel()), int(0.8*len(self.Y[m].ravel())), False)
                self.Y[m].ravel()[self.Y_dropout_indices[m]] = 0
                self.Y[m] *= 5

    def make_weights_symmetric(self):
        '''
        Make the feedback weights in the network symmetric to (transposes of) the feedforward weights.
        '''

        if use_broadcast:
            for m in xrange(self.M-2, -1, -1):
                # make a copy if we're altering the feedback weights after
                if use_sparse_feedback:
                    W_above = self.W[m+1].T.copy()
                else:
                    W_above = self.W[m+1].T

                if m == self.M - 2:
                    # for final hidden layer, use feedforward weights of output layer
                    if noisy_symmetric_weights:
                        self.Y[m] = W_above + np.random.normal(0, 0.05, size=W_above.shape)
                    else:
                        self.Y[m] = W_above
                else:
                    # for other hidden layers, use product of all feedforward weights downstream
                    if noisy_symmetric_weights:
                        self.Y[m] = np.dot(W_above + np.random.normal(0, 0.05, size=W_above.shape), self.Y[m+1])
                    else:
                        self.Y[m] = np.dot(W_above, self.Y[m+1])
        else:
            for m in xrange(self.M-2, -1, -1):
                # make a copy if we're altering the feedback weights after
                if use_sparse_feedback:
                    W_above = self.W[m+1].T.copy()
                else:
                    W_above = self.W[m+1].T

                # use feedforward weights of the layer downstream
                if noisy_symmetric_weights:
                    self.Y[m] = W_above + np.random.normal(0, 0.05)
                else:
                    self.Y[m] = W_above

    def init_layers(self):
        # initialize layers list
        self.l = []

        # create all layers
        if self.M == 1:
            self.l.append(finalLayer(net=self, m=-1, f_input_size=self.n_in))
        else:
            if use_broadcast:
                self.l.append(hiddenLayer(net=self, m=0, f_input_size=self.n_in, b_input_size=self.n[-1]))
                for m in xrange(1, self.M-1):
                    self.l.append(hiddenLayer(net=self, m=m, f_input_size=self.n[m-1], b_input_size=self.n[-1]))
            else:
                self.l.append(hiddenLayer(net=self, m=0, f_input_size=self.n_in, b_input_size=self.n[1]))
                for m in xrange(1, self.M-1):
                    self.l.append(hiddenLayer(net=self, m=m, f_input_size=self.n[m-1], b_input_size=self.n[m+1]))
            self.l.append(finalLayer(net=self, m=self.M-1, f_input_size=self.n[-2]))

    def out_f(self, training=False):
        '''
        Perform a forward phase pass through the network.

        Arguments:
            training (bool) : Whether the network is in training (True) or testing (False) mode.
        '''

        if use_spiking_feedforward:
            x = self.x_hist
        else:
            x = self.x

        if self.M == 1:
            self.l[0].out_f(x, None)
        else:
            if use_broadcast:
                if use_spiking_feedback:
                    self.l[0].out_f(x, self.l[-1].S_hist)

                    for m in xrange(1, self.M-1):
                        if use_spiking_feedforward:
                            self.l[m].out_f(self.l[m-1].S_hist, self.l[-1].S_hist)
                        else:
                            self.l[m].out_f(self.l[m-1].lambda_C, self.l[-1].S_hist)

                    if use_spiking_feedforward:
                        self.l[-1].out_f(self.l[-2].S_hist, None)
                    else:
                        self.l[-1].out_f(self.l[-2].lambda_C, None)
                else:
                    self.l[0].out_f(x, self.l[-1].lambda_C)

                    for m in xrange(1, self.M-1):
                        if use_spiking_feedforward:
                            self.l[m].out_f(self.l[m-1].S_hist, self.l[-1].lambda_C)
                        else:
                            self.l[m].out_f(self.l[m-1].lambda_C, self.l[-1].lambda_C)

                    if use_spiking_feedforward:
                        self.l[-1].out_f(self.l[-2].S_hist, None)
                    else:
                        self.l[-1].out_f(self.l[-2].lambda_C, None)
            else:
                if use_spiking_feedback:
                    self.l[0].out_f(x, self.l[1].S_hist)

                    for m in xrange(1, self.M-1):
                        if use_spiking_feedforward:
                            self.l[m].out_f(self.l[m-1].S_hist, self.l[m+1].S_hist)
                        else:
                            self.l[m].out_f(self.l[m-1].lambda_C, self.l[m+1].S_hist)

                    if use_spiking_feedforward:
                        self.l[-1].out_f(self.l[-2].S_hist, None)
                    else:
                        self.l[-1].out_f(self.l[-2].lambda_C, None)
                else:
                    self.l[0].out_f(x, self.l[1].lambda_C)

                    for m in xrange(1, self.M-1):
                        if use_spiking_feedforward:
                            self.l[m].out_f(self.l[m-1].S_hist, self.l[m+1].lambda_C)
                        else:
                            self.l[m].out_f(self.l[m-1].lambda_C, self.l[m+1].lambda_C)

                    if use_spiking_feedforward:
                        self.l[-1].out_f(self.l[-2].S_hist, None)
                    else:
                        self.l[-1].out_f(self.l[-2].lambda_C, None)

    def out_t(self):
        '''
        Perform a target phase pass through the network. This is the same as a forward phase pass,
        but with a target introduced at the top layer.
        '''

        # same as forward pass, but with a target introduced at the top layer
        if use_spiking_feedforward:
            x = self.x_hist
        else:
            x = self.x

        if self.M == 1:
            self.l[0].out_t(x, self.t)
        else:
            if use_broadcast:
                if use_spiking_feedback:
                    self.l[0].out_t(x, self.l[-1].S_hist)

                    for m in xrange(1, self.M-1):
                        if use_spiking_feedforward:
                            self.l[m].out_t(self.l[m-1].S_hist, self.l[-1].S_hist)
                        else:
                            self.l[m].out_t(self.l[m-1].lambda_C, self.l[-1].S_hist)

                    if use_spiking_feedforward:
                        self.l[-1].out_t(self.l[-2].S_hist, self.t)
                    else:
                        self.l[-1].out_t(self.l[-2].lambda_C, self.t)
                else:
                    self.l[0].out_t(x, self.l[-1].lambda_C)

                    for m in xrange(1, self.M-1):
                        if use_spiking_feedforward:
                            self.l[m].out_t(self.l[m-1].S_hist, self.l[-1].lambda_C)
                        else:
                            self.l[m].out_t(self.l[m-1].lambda_C, self.l[-1].lambda_C)

                    if use_spiking_feedforward:
                        self.l[-1].out_t(self.l[-2].S_hist, self.t)
                    else:
                        self.l[-1].out_t(self.l[-2].lambda_C, self.t)
            else:
                if use_spiking_feedback:
                    self.l[0].out_t(x, self.l[1].S_hist)

                    for m in xrange(1, self.M-1):
                        if use_spiking_feedforward:
                            self.l[m].out_t(self.l[m-1].S_hist, self.l[m+1].S_hist)
                        else:
                            self.l[m].out_t(self.l[m-1].lambda_C, self.l[m+1].S_hist)

                    if use_spiking_feedforward:
                        self.l[-1].out_t(self.l[-2].S_hist, self.t)
                    else:
                        self.l[-1].out_t(self.l[-2].lambda_C, self.t)
                else:
                    self.l[0].out_t(x, self.l[1].lambda_C)

                    for m in xrange(1, self.M-1):
                        if use_spiking_feedforward:
                            self.l[m].out_t(self.l[m-1].S_hist, self.l[m+1].lambda_C)
                        else:
                            self.l[m].out_t(self.l[m-1].lambda_C, self.l[m+1].lambda_C)

                    if use_spiking_feedforward:
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

        if record_voltages and training:
            # initialize voltage arrays
            self.A_hists = [ np.zeros((l_f_phase, self.l[m].size)) for m in xrange(self.M-1) ]
            self.B_hists = [ np.zeros((l_f_phase, self.l[m].size)) for m in xrange(self.M) ]
            self.C_hists = [ np.zeros((l_f_phase, self.l[m].size)) for m in xrange(self.M) ]

        for time in xrange(l_f_phase):
            # update input spike history
            self.x_hist = np.concatenate([self.x_hist[:, 1:], np.random.poisson(x)], axis=-1)

            # do a forward pass
            self.out_f(training=training)

            if use_rand_plateau_times and training:
                # calculate plateau potentials for hidden layer neurons
                for m in xrange(self.M-2, -1, -1):
                    plateau_indices = np.nonzero(time == self.plateau_times_f[m][training_num])

                    self.l[m].plateau_f(plateau_indices=plateau_indices)

            if record_voltages and training:
                # record voltages for this timestep
                for m in xrange(self.M):
                    if m != self.M-1:
                        self.A_hists[m][time, :] = self.l[m].A[:, 0]
                    self.B_hists[m][time, :] = self.l[m].B[:, 0]
                    self.C_hists[m][time, :] = self.l[m].C[:, 0]

        if (not use_rand_plateau_times) or (not training):
            for m in xrange(self.M-2, -1, -1):
                plateau_indices = np.arange(self.n[m])

                # calculate plateau potentials for hidden layer neurons
                self.l[m].plateau_f(plateau_indices=plateau_indices)

        for m in xrange(self.M-1, -1, -1):
             # calculate averages
            self.l[m].calc_averages(phase="forward")

        if record_eigvals:
            # calculate Jacobians & update lists of last 100 Jacobians
            if len(self.J_betas) >= 100:
                self.J_betas = self.J_betas[1:]
                self.J_gammas = self.J_gammas[1:]

            self.J_betas.append(np.multiply(lambda_max*deriv_sigma(self.l[-1].average_C_f), k_D*self.W[-1]))
            self.J_gammas.append(np.multiply(deriv_sigma(np.dot(self.Y[-2], lambda_max*sigma(self.l[-1].average_C_f))), self.Y[-2]))

        if record_voltages and training:
            # append voltages to files
            for m in xrange(self.M):
                if m != self.M-1:
                    with open(os.path.join(self.simulation_path, 'A_hist_{}.csv'.format(m)), 'a') as A_hist_file:
                        np.savetxt(A_hist_file, self.A_hists[m])
                with open(os.path.join(self.simulation_path, 'B_hist_{}.csv'.format(m)), 'a') as B_hist_file:
                    np.savetxt(B_hist_file, self.B_hists[m])
                with open(os.path.join(self.simulation_path, 'C_hist_{}.csv'.format(m)), 'a') as C_hist_file:
                    np.savetxt(C_hist_file, self.C_hists[m])

    def t_phase(self, x, t, training_num):
        '''
        Perform a target phase.

        Arguments:
            x (ndarray)        : Input array of size (X, 1) where X is the size of the input, eg. (784, 1).
            t (ndarray)        : Target array of size (T, 1) where T is the size of the target, eg. (10, 1).
            training_num (int) : Number (from start of the epoch) of the training example being shown.
        '''

        if record_voltages:
            # initialize voltage arrays
            self.A_hists = [ np.zeros((l_t_phase, self.l[m].size)) for m in xrange(self.M-1)]
            self.B_hists = [ np.zeros((l_t_phase, self.l[m].size)) for m in xrange(self.M)]
            self.C_hists = [ np.zeros((l_t_phase, self.l[m].size)) for m in xrange(self.M)]

        for time in xrange(l_t_phase):
            # update input history
            self.x_hist = np.concatenate([self.x_hist[:, 1:], np.random.poisson(x)], axis=-1)

            # calculate backprop angle at the end of the target phase
            calc_E_bp = record_backprop_angle and time == l_t_phase - 1

            # do a target pass
            self.out_t()

            if use_rand_plateau_times:
                # calculate plateau potentials & perform weight updates
                for m in xrange(self.M-2, -1, -1):
                    plateau_indices = np.nonzero(time == self.plateau_times_t[m][training_num])

                    self.l[m].plateau_t(plateau_indices=plateau_indices)

            if record_voltages:
                # record voltages for this timestep
                for m in xrange(self.M):
                    if m != self.M-1:
                        self.A_hists[m][time, :] = self.l[m].A[:, 0]
                    self.B_hists[m][time, :] = self.l[m].B[:, 0]
                    self.C_hists[m][time, :] = self.l[m].C[:, 0]

        if not use_rand_plateau_times:
            for m in xrange(self.M-2, -1, -1):
                plateau_indices = np.arange(self.n[m])

                # calculate plateau potentials for hidden layer neurons
                self.l[m].plateau_t(plateau_indices=plateau_indices)

        for m in xrange(self.M-1, -1, -1):
            # calculate averages
            self.l[m].calc_averages(phase="target")

            if update_feedback_weights and m < self.M-1:
                # update feedback weights
                self.l[m].update_Y()

            # update weights
            self.l[m].update_W()

        if record_loss:
            self.loss = ((self.l[-1].average_lambda_C_t - lambda_max*sigma(self.l[-1].average_C_f)) ** 2).mean()

        for m in xrange(self.M-1, -1, -1):
            # reset averages
            self.l[m].average_C_f     *= 0
            self.l[m].average_C_t     *= 0
            self.l[m].average_PSP_B_f *= 0

            if m == self.M-1:
                self.l[m].average_lambda_C_f *= 0
                self.l[m].average_lambda_C_t *= 0
            else:
                self.l[m].average_A_f     *= 0
                self.l[m].average_A_t     *= 0
                self.l[m].average_lambda_C_f *= 0
                if update_feedback_weights:
                    self.l[m].average_PSP_A_f *= 0

        if use_symmetric_weights:
            # make feedback weights symmetric to new feedforward weights
            self.make_weights_symmetric()

        if use_sparse_feedback and (use_symmetric_weights or update_feedback_weights):
            for m in xrange(self.M-1):
                # zero out the inactive weights
                self.Y[m].ravel()[self.Y_dropout_indices[m]] = 0

                # increase magnitude of surviving weights
                self.Y[m] *= 5

        if record_voltages:
            # append voltages to files
            for m in xrange(self.M):
                if m != self.M-1:
                    with open(os.path.join(self.simulation_path, 'A_hist_{}.csv'.format(m)), 'a') as A_hist_file:
                        np.savetxt(A_hist_file, self.A_hists[m])
                with open(os.path.join(self.simulation_path, 'B_hist_{}.csv'.format(m)), 'a') as B_hist_file:
                    np.savetxt(B_hist_file, self.B_hists[m])
                with open(os.path.join(self.simulation_path, 'C_hist_{}.csv'.format(m)), 'a') as C_hist_file:
                    np.savetxt(C_hist_file, self.C_hists[m])

    def train(self, f_etas, b_etas, n_epochs, n_training_examples, save_simulation, simulations_folder=default_simulations_folder, folder_name="", overwrite=False, simulation_notes=None, current_epoch=None):
        '''
        Train the network. Checkpoints will be saved at the end of every epoch if save_simulation is True.

        Arguments:
            f_etas (tuple)              : Learning rates for each layer's feedforward weights, eg. (0.21, 0.21).
            b_etas (tuple/None)         : Learning rates for each layer's feedback weights.
                                          If None, no backward weight updates occur.
            n_epochs (int)              : Number of epochs of training.
            n_training_examples (int)   : Number of training examples per epoch.
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

        if b_etas == None and update_feedback_weights:
            raise ValueError("No feedback learning rates provided, but 'update_feedback_weights' is True.")

        if current_epoch != None:
            self.current_epoch == current_epoch
        elif self.current_epoch == None:
            # set current epoch
            self.current_epoch = 0

        if self.current_epoch == 0:
            continuing = False
        else:
            continuing = True

        if use_rand_phase_lengths:
            # generate phase lengths for all training examples
            global l_f_phase, l_t_phase
            l_f_phases = min_l_f_phase + np.random.wald(2, 1, n_epochs*n_training_examples).astype(int)
            l_t_phases = min_l_t_phase + np.random.wald(2, 1, n_epochs*n_training_examples).astype(int)
        else:
            l_f_phases = np.zeros(n_epochs*n_training_examples) + l_f_phase
            l_t_phases = np.zeros(n_epochs*n_training_examples) + l_t_phase

        # get array of total length of both phases for all training examples
        l_phases_tot = l_f_phases + l_t_phases

        # get current date/time and create simulation directory
        if save_simulation:
            sim_start_time = datetime.datetime.now()

            if folder_name == "":
                self.simulation_path = os.path.join(simulations_folder, "{}.{}.{}-{}.{}".format(sim_start_time.year,
                                                                                 sim_start_time.month,
                                                                                 sim_start_time.day,
                                                                                 sim_start_time.hour,
                                                                                 sim_start_time.minute))
            else:
                self.simulation_path = os.path.join(simulations_folder, folder_name)

            # make simulation directory
            if not os.path.exists(self.simulation_path):
                os.makedirs(self.simulation_path)
            elif not continuing:
                if overwrite == False:
                    print("Error: Simulation directory \"{}\" already exists.".format(self.simulation_path))
                    return
                else:
                    shutil.rmtree(self.simulation_path, ignore_errors=True)
                    os.makedirs(self.simulation_path)

            # copy current script to simulation directory
            filename = os.path.basename(__file__)
            if filename.endswith('pyc'):
                filename = filename[:-1]
            shutil.copyfile(filename, os.path.join(self.simulation_path, filename))

            params = {
                'nonspiking_mode'        : nonspiking_mode,
                'n_full_test'            : n_full_test,
                'n_quick_test'           : n_quick_test,
                'use_rand_phase_lengths' : use_rand_phase_lengths,
                'use_rand_plateau_times' : use_rand_plateau_times,
                'use_conductances'       : use_conductances,
                'use_broadcast'          : use_broadcast,
                'use_spiking_feedback'   : use_spiking_feedback,
                'use_spiking_feedforward': use_spiking_feedforward,
                'use_symmetric_weights'  : use_symmetric_weights,
                'noisy_symmetric_weights': noisy_symmetric_weights,
                'use_sparse_feedback'    : use_sparse_feedback,
                'update_feedback_weights': update_feedback_weights,
                'use_backprop'           : use_backprop,
                'use_apical_conductance' : use_apical_conductance,
                'use_weight_optimization': use_weight_optimization,
                'use_feedback_bias'      : use_feedback_bias,
                'initial_test'           : initial_test,
                'record_backprop_angle'  : record_backprop_angle,
                'record_loss'            : record_loss,
                'record_voltages'        : record_voltages,
                'record_training_error'  : record_training_error,
                'record_training_labels' : record_training_labels,
                'record_phase_times'     : record_phase_times,
                'record_plateau_times'   : record_plateau_times,
                'record_eigvals'         : record_eigvals,
                'record_matrices'        : record_matrices,
                'plot_eigvals'           : plot_eigvals,
                'dt'                     : dt,
                'mem'                    : mem,
                'integration_time'       : integration_time,
                'integration_time_test'  : integration_time_test,
                'l_f_phase'              : l_f_phase,
                'l_t_phase'              : l_t_phase,
                'l_f_phase_test'         : l_f_phase_test,
                'lambda_max'             : lambda_max,
                'tau_s'                  : tau_s,
                'tau_L'                  : tau_L,
                'g_B'                    : g_B,
                'g_A'                    : g_A,
                'g_L'                    : g_L,
                'g_D'                    : g_D,
                'k_B'                    : k_B,
                'k_D'                    : k_D,
                'k_I'                    : k_I,
                'P_hidden'               : P_hidden,
                'P_final'                : P_final,
                'n'                      : self.n,
                'f_etas'                 : f_etas,
                'b_etas'                 : b_etas,
                'n_training_examples'    : n_training_examples,
                'n_epochs'               : n_epochs
            }

            # save simulation params
            if not continuing:
                with open(os.path.join(self.simulation_path, 'simulation.txt'), 'w') as simulation_file:
                    print("Simulation done on {}.{}.{}-{}.{}.".format(sim_start_time.year,
                                                                     sim_start_time.month,
                                                                     sim_start_time.day,
                                                                     sim_start_time.hour,
                                                                     sim_start_time.minute), file=simulation_file)
                    if simulation_notes:
                        print(simulation_notes, file=simulation_file)
                    print("Start time: {}".format(sim_start_time), file=simulation_file)
                    print("-----------------------------", file=simulation_file)
                    for key, value in sorted(params.items()):
                        line = '{}: {}'.format(key, value)
                        print(line, file=simulation_file)

                with open(os.path.join(self.simulation_path, 'simulation.json'), 'w') as simulation_file:
                    simulation_file.write(json.dumps(params))
            else:
                # load previously saved recording arrays
                self.prev_full_test_errs   = np.load(os.path.join(self.simulation_path, "full_test_errors.npy"))
                self.prev_quick_test_errs  = np.load(os.path.join(self.simulation_path, "quick_test_errors.npy"))

                if record_backprop_angle and not use_backprop:
                    self.prev_bp_angles = np.load(os.path.join(self.simulation_path, "bp_angles.npy"))

                if record_loss:
                    self.prev_losses = np.load(os.path.join(self.simulation_path, "final_layer_loss.npy"))

                if record_training_error:
                    self.prev_training_errors = np.load(os.path.join(self.simulation_path, "training_errors.npy"))

                if record_training_labels:
                    self.prev_training_labels = np.load(os.path.join(self.simulation_path, "training_labels.npy"))

                if record_plateau_times:
                    self.prev_plateau_times_full = [ np.load(os.path.join(self.simulation_path, "plateau_times_{}.npy".format(m))) for m in range(self.M)]

                if record_phase_times:
                    self.prev_phase_times = np.load(os.path.join(self.simulation_path, "phase_times.npy"))

                if record_eigvals:
                    self.prev_max_jacobian_eigvals   = np.load(os.path.join(self.simulation_path, "max_jacobian_eigvals.npy"))
                    self.prev_max_weight_eigvals     = np.load(os.path.join(self.simulation_path, "max_weight_eigvals.npy"))
                    if record_matrices:
                        self.prev_jacobian_prod_matrices = np.load(os.path.join(self.simulation_path, "jacobian_prod_matrices.npy"))
                        self.prev_weight_prod_matrices   = np.load(os.path.join(self.simulation_path, "weight_prod_matrices.npy"))

        # set learning rate instance variables
        self.f_etas = f_etas
        self.b_etas = b_etas

        if save_simulation and not continuing:
            # save initial weights
            self.save_weights(self.simulation_path, prefix='initial_')

        if not continuing:
            # initialize full test error recording array
            self.full_test_errs  = np.zeros(n_epochs + 1)

            # initialize quick test error recording array
            self.quick_test_errs = np.zeros(n_epochs*int(n_training_examples/1000.0) + 1)
        else:
            self.full_test_errs  = np.zeros(n_epochs)
            self.quick_test_errs = np.zeros(n_epochs*int(n_training_examples/1000.0))

        if record_loss:
            self.losses = np.zeros(n_epochs*n_training_examples)

        if record_training_error:
            self.training_errors = np.zeros(n_epochs)

        if record_plateau_times:
            self.plateau_times_full = [ np.zeros((n_epochs*2*n_training_examples, self.n[m])) for m in range(self.M) ]

        if record_phase_times:
            self.phase_times = np.zeros(n_epochs*n_training_examples*2)

            self.phase_times[0] = l_f_phases[0]
            for i in xrange(1, 2*n_epochs*n_training_examples):
                if i % 2 == 0:
                    self.phase_times[i] = self.phase_times[i-1] + l_f_phases[int(i/2)]
                else:
                    self.phase_times[i] = self.phase_times[i-1] + l_t_phases[int((i-1)/2)]

            if save_simulation:
                if not continuing:
                    phase_times = self.phase_times
                else:
                    phase_times = np.concatenate(self.prev_phase_times, self.phase_times, axis=0)

                np.save(os.path.join(self.simulation_path, "phase_times.npy"), phase_times)

        if record_training_labels:
            self.training_labels = np.zeros(n_epochs*n_training_examples)

        if record_eigvals:
            # initialize arrays for Jacobian testing
            self.max_jacobian_eigvals = np.zeros(n_epochs*n_training_examples)
            if record_matrices:
                self.jacobian_prod_matrices = np.zeros((n_epochs*n_training_examples, self.n[-1], self.n[-1]))

            if not continuing:
                self.max_weight_eigvals = np.zeros(n_epochs*n_training_examples + 1)
                if record_matrices:
                    self.weight_prod_matrices = np.zeros((n_epochs*n_training_examples + 1, self.n[-1], self.n[-1]))
            else:
                self.max_weight_eigvals = np.zeros(n_epochs*n_training_examples)
                if record_matrices:
                    self.weight_prod_matrices = np.zeros((n_epochs*n_training_examples, self.n[-1], self.n[-1]))

            # create identity matrix
            I = np.eye(self.n[-1])

            # get max eigenvalues for weights
            U = np.dot(self.W[-1], self.Y[-2])
            p = np.dot((I - U).T, I - U)

            if not continuing:
                if record_matrices:
                    self.weight_prod_matrices[0] = U
                self.max_weight_eigvals[0] = np.amax(np.real(np.linalg.eigvals(p)))

            # initialize lists for storing last 100 Jacobians
            self.J_betas = []
            self.J_gammas = []

        if record_backprop_angle and not use_backprop:
            # initialize backprop angles recording array
            if self.M > 1:
                self.bp_angles = np.zeros(n_epochs*n_training_examples)

        if initial_test and not continuing:
            # do an initial weight test
            print("Start of epoch {}.".format(self.current_epoch + 1))

            # set start time
            start_time = time.time()

            test_err = self.test_weights(n_test=n_full_test)

            # get end time & elapsed time
            end_time = time.time()
            time_elapsed = end_time - start_time

            sys.stdout.write("\x1b[2K\rFE: {0:05.2f}%. T: {1:.3f}s.\n\n".format(test_err, time_elapsed))

            self.full_test_errs[0] = test_err

            if save_simulation:
                # save full test error
                np.save(os.path.join(self.simulation_path, "full_test_errors.npy"), self.full_test_errs)

                with open(os.path.join(self.simulation_path, "full_test_errors.txt"), 'a') as test_err_file:
                    line = "%.10f" % test_err
                    print(line, file=test_err_file)

            self.quick_test_errs[0] = test_err

            if save_simulation:
                # save quick test error
                np.save(os.path.join(self.simulation_path, "quick_test_errors.npy"), self.quick_test_errs)

                with open(os.path.join(self.simulation_path, "quick_test_errors.txt"), 'a') as test_err_file:
                    line = "%.10f" % test_err
                    print(line, file=test_err_file)
        else:
            # don't do an initial weight test
            print("Start of epoch {}.\n".format(self.current_epoch + 1))

        # initialize input spike history
        self.x_hist   = np.zeros((self.n_in, mem))

        # start time used for timing how long each 1000 examples take
        start_time = None

        if record_eigvals and plot_eigvals:
            plt.close("all")
            fig = plt.figure(figsize=(13, 8))
            ax1 = fig.add_subplot(311)
            ax2 = fig.add_subplot(321)
            ax3 = fig.add_subplot(312)
            plt.show(block=False)

        if record_training_error:
            num_correct = 0

        for k in xrange(n_epochs):
            # shuffle the training data
            self.x_train, self.t_train = shuffle_arrays(self.x_train, self.t_train)

            # generate arrays of forward phase plateau potential times (time until plateau potential from start of forward phase) for individual neurons
            if use_rand_plateau_times:
                self.plateau_times_f = [ np.zeros((n_training_examples, self.n[m])) + l_f_phases[k*n_training_examples:(k+1)*n_training_examples, np.newaxis] - 1 - np.minimum(np.abs(np.random.normal(0, 3, size=(n_training_examples, self.n[m])).astype(int)), 5) for m in range(self.M) ]
            else:
                self.plateau_times_f = [ np.zeros((n_training_examples, self.n[m])) + l_f_phases[k*n_training_examples:(k+1)*n_training_examples, np.newaxis] - 1 for m in range(self.M) ]

            # generate arrays of target phase plateau potential times (time until plateau potential from start of target phase) for individual neurons
            if use_rand_plateau_times:
                self.plateau_times_t = [ np.zeros((n_training_examples, self.n[m])) + l_t_phases[k*n_training_examples:(k+1)*n_training_examples, np.newaxis] - 1 - np.minimum(np.abs(np.random.normal(0, 3, size=(n_training_examples, self.n[m])).astype(int)), 5) for m in range(self.M) ]
            else:
                self.plateau_times_t = [ np.zeros((n_training_examples, self.n[m])) + l_t_phases[k*n_training_examples:(k+1)*n_training_examples, np.newaxis] - 1 for m in range(self.M) ]

            for n in xrange(n_training_examples):
                # set start time
                if start_time == None:
                    start_time = time.time()

                if use_rand_phase_lengths:
                    l_f_phase = int(l_f_phases[k*n_training_examples + n])
                    l_t_phase = int(l_t_phases[k*n_training_examples + n])

                l_phases_tot = l_f_phase + l_t_phase

                # get plateau potential times from the beginning of the simulation
                if record_plateau_times:
                    total_time_to_forward_phase = np.sum(l_f_phases[:k*n_training_examples + n]) + np.sum(l_t_phases[:k*n_training_examples + n])
                    total_time_to_target_phase  = np.sum(l_f_phases[:k*n_training_examples + n + 1]) + np.sum(l_t_phases[:k*n_training_examples + n])
                    for m in range(self.M):
                        self.plateau_times_full[m][k*n_training_examples + 2*n]     = total_time_to_forward_phase + self.plateau_times_f[m][n]
                        self.plateau_times_full[m][k*n_training_examples + 2*n + 1] = total_time_to_target_phase + self.plateau_times_t[m][n]

                # print every 100 examples
                if (n+1) % 100 == 0:
                    sys.stdout.write("\x1b[2K\rEpoch {0}, example {1}/{2}.".format(self.current_epoch + 1, n+1, n_training_examples))
                    sys.stdout.flush()

                # get training example data
                self.x = lambda_max*self.x_train[:, n][:, np.newaxis]
                self.t = self.t_train[:, n][:, np.newaxis]

                if record_voltages:
                    # initialize voltage arrays
                    self.A_hists = [ np.zeros((l_f_phase, self.l[m].size)) for m in xrange(self.M-1)]
                    self.B_hists = [ np.zeros((l_f_phase, self.l[m].size)) for m in xrange(self.M)]
                    self.C_hists = [ np.zeros((l_f_phase, self.l[m].size)) for m in xrange(self.M)]

                # do forward & target phases
                self.f_phase(self.x, None, n, training=True)

                if record_training_error:
                    sel_num = np.argmax(np.mean(self.l[-1].average_C_f.reshape(-1, self.n_neurons_per_category), axis=-1))

                    # get the target number from testing example data
                    target_num = np.dot(np.arange(10), self.t)

                    # increment correct classification counter if they match
                    if sel_num == target_num:
                        num_correct += 1

                self.t_phase(self.x, self.t.repeat(self.n_neurons_per_category, axis=0), n)

                if record_loss:
                    self.losses[k*n_training_examples + n] = self.loss

                if record_training_labels:
                    self.training_labels[k*n_training_examples + n] = np.dot(np.arange(10), self.t)

                if record_eigvals:
                    # get max eigenvalues for jacobians
                    # U = np.dot(np.mean(np.array(self.J_betas), axis=0), np.mean(np.array(self.J_gammas), axis=0)) # product of mean of last 100 Jacobians
                    U = np.mean(np.array([ np.dot(self.J_betas[i], self.J_gammas[i]) for i in range(len(self.J_betas)) ]), axis=0) # mean of product of last 100 Jacobians

                    p = np.dot((I - U).T, I - U)
                    if record_matrices:
                        self.jacobian_prod_matrices[k*n_training_examples + n] = U
                    self.max_jacobian_eigvals[k*n_training_examples + n] = np.amax(np.linalg.eigvals(p))

                    # get max eigenvalues for weights
                    U = np.dot(k_D*self.W[-1], self.Y[-2])
                    p = np.dot((I - U).T, I - U)
                    if not continuing:
                        if record_matrices:
                            self.weight_prod_matrices[k*n_training_examples + n + 1] = U
                        self.max_weight_eigvals[k*n_training_examples + n + 1] = np.amax(np.linalg.eigvals(p))
                    else:
                        if record_matrices:
                            self.weight_prod_matrices[k*n_training_examples + n] = U
                        self.max_weight_eigvals[k*n_training_examples + n] = np.amax(np.linalg.eigvals(p))

                    if plot_eigvals and k == 0 and n == 0:
                        # draw initial plots
                        if record_matrices:
                            A = self.jacobian_prod_matrices[0]
                            im_plot = ax1.imshow(A, interpolation='nearest', vmin=0, vmax=1)
                            fig.colorbar(im_plot, ax=ax1)
                        if record_loss:
                            loss_plot, = ax2.plot(np.arange(1), self.losses[0])
                        max_jacobian_plot, = ax3.plot(np.arange(1), self.max_jacobian_eigvals[0], '.')
                        fig.canvas.draw()
                        fig.canvas.flush_events()

                if record_backprop_angle and not use_backprop:
                    # get backprop angle
                    if self.M > 1:
                        bp_angle = np.arccos(np.sum(self.l[0].delta_b_bp * self.l[0].delta_b_full) / (np.linalg.norm(self.l[0].delta_b_bp)*np.linalg.norm(self.l[0].delta_b_full.T)))*180.0/np.pi
                        self.bp_angles[k*n_training_examples + n] = bp_angle

                if plot_eigvals and record_eigvals and (n+1) % 100 == 0:
                    max_inds = np.argsort(self.max_jacobian_eigvals[k*n_training_examples + n -99:k*n_training_examples + n + 1])
                    max_ind = np.argmax(self.max_jacobian_eigvals[k*n_training_examples + n-99:k*n_training_examples + n + 1])
                    min_ind = np.argmin(self.max_jacobian_eigvals[k*n_training_examples + n-99:k*n_training_examples + n + 1])
                    n_small = np.sum(self.max_jacobian_eigvals[k*n_training_examples + n-99:k*n_training_examples + n + 1] < 1)

                    # update plots
                    if record_matrices:
                        A = np.mean(np.array([self.jacobian_prod_matrices[k*n_training_examples + n-99:k*n_training_examples + n + 1][i] for i in max_inds][:-10]), axis=0)
                        im_plot.set_data(A)

                    if record_loss:
                        loss_plot.set_xdata(np.arange(k*n_training_examples + n))
                        loss_plot.set_ydata(self.losses[:k*n_training_examples + n])
                        ax2.set_xlim(0, k*n_training_examples + n)
                        ax2.set_ylim(np.amin(self.losses[:k*n_training_examples + n]) - 1e-6, np.amax(self.losses[:k*n_training_examples + n]) + 1e-6)

                    max_jacobian_plot.set_xdata(np.arange(k*n_training_examples + n))
                    max_jacobian_plot.set_ydata(self.max_jacobian_eigvals[:k*n_training_examples + n])
                    ax3.set_xlim(0, k*n_training_examples + n)
                    ax3.set_ylim(np.amin(self.max_jacobian_eigvals[:k*n_training_examples + n]) - 1e-6, np.amax(self.max_jacobian_eigvals[:k*n_training_examples + n]) + 1e-6)

                    fig.canvas.draw()
                    fig.canvas.flush_events()

                if (n+1) % 1000 == 0:
                    if n != n_training_examples - 1:
                        # we're partway through an epoch; do a quick weight test
                        test_err = self.test_weights(n_test=n_quick_test)

                        sys.stdout.write("\x1b[2K\rEpoch {0}, example {1}/{2}. QE: {3:05.2f}%. ".format(self.current_epoch + 1, n+1, n_training_examples, test_err))

                        if not continuing:
                            self.quick_test_errs[(k+1)*int(n_training_examples/1000)] = test_err
                        else:
                            self.quick_test_errs[(k+1)*int(n_training_examples/1000) - 1] = test_err

                        if save_simulation:
                            with open(os.path.join(self.simulation_path, "quick_test_errors.txt"), 'a') as test_err_file:
                                line = "%.10f" % test_err
                                print(line, file=test_err_file)
                    else:
                        # we've reached the end of an epoch; do a full weight test
                        test_err = self.test_weights(n_test=n_full_test)

                        sys.stdout.write("\x1b[2K\rFE: {0:05.2f}%. ".format(test_err))

                        if not continuing:
                            self.full_test_errs[k+1] = test_err
                            self.quick_test_errs[(k+1)*int(n_training_examples/1000)] = test_err
                        else:
                            self.full_test_errs[k] = test_err
                            self.quick_test_errs[(k+1)*int(n_training_examples/1000) - 1] = test_err

                        if save_simulation:
                            with open(os.path.join(self.simulation_path, "full_test_errors.txt"), 'a') as test_err_file:
                                line = "%.10f" % test_err
                                print(line, file=test_err_file)

                        if record_training_error:
                            # calculate percent training error for this epoch
                            err_rate = (1.0 - float(num_correct)/n_training_examples)*100.0
                            self.training_errors[k] = err_rate

                            print("TE: {0:05.2f}%. ".format(err_rate), end="")

                            num_correct = 0

                        # save recording arrays
                        if save_simulation:
                            print("Saving...", end="")
                            if not continuing:
                                # we are running a new simulation
                                quick_test_errs = self.quick_test_errs[:(k+1)*int(n_training_examples/1000)+1]
                                full_test_errs  = self.full_test_errs[:k+2]

                                if record_backprop_angle and not use_backprop:
                                    bp_angles = self.bp_angles[:(k+1)*n_training_examples]

                                if record_loss:
                                    losses = self.losses[:(k+1)*n_training_examples]

                                if record_training_labels:
                                    training_labels = self.training_labels[:(k+1)*n_training_examples]

                                if record_plateau_times:
                                    plateau_times_full = [ self.plateau_times_full[m][:(k+1)*2*n_training_examples] for m in range(self.M) ]

                                if record_training_error:
                                    training_errors = self.training_errors[:k+1]

                                if record_eigvals:
                                    max_jacobian_eigvals   = self.max_jacobian_eigvals[:(k+1)*n_training_examples]
                                    max_weight_eigvals     = self.max_weight_eigvals[:(k+1)*n_training_examples+1]
                                    if record_matrices:
                                        jacobian_prod_matrices = self.jacobian_prod_matrices[:(k+1)*n_training_examples]
                                        weight_prod_matrices   = self.weight_prod_matrices[:(k+1)*n_training_examples+1]
                            else:
                                # this is a continuation of a previously-started simulation; append current recording vectors to previous ones
                                quick_test_errs = np.concatenate([self.prev_quick_test_errs, self.quick_test_errs[:(k+1)*int(n_training_examples/1000)]], axis=0)
                                if n == n_training_examples - 1:
                                    full_test_errs = np.concatenate([self.prev_full_test_errs, self.full_test_errs[:k+1]], axis=0)

                                if record_backprop_angle and not use_backprop:
                                    bp_angles = np.concatenate([self.prev_bp_angles, self.bp_angles[:(k+1)*n_training_examples]], axis=0)

                                if record_loss:
                                    losses = np.concatenate([self.prev_losses, self.losses[:(k+1)*n_training_examples]], axis=0)

                                if record_training_labels:
                                    training_labels = np.concatenate([self.prev_training_labels, self.training_labels[:(k+1)*n_training_examples]], axis=0)

                                if record_plateau_times:
                                    plateau_times_full = [ np.concatenate([self.prev_plateau_times_full[m], self.plateau_times_full[m][:(k+1)*2*n_training_examples]]) for m in range(self.M) ]

                                if record_training_error:
                                    training_errors = np.concatenate([self.prev_training_errors, self.training_errors[:k+1]], axis=0)

                                if record_eigvals:
                                    max_jacobian_eigvals   = np.concatenate([self.prev_max_jacobian_eigvals, self.max_jacobian_eigvals[:(k+1)*n_training_examples]], axis=0)
                                    max_weight_eigvals     = np.concatenate([self.prev_max_weight_eigvals, self.max_weight_eigvals[:(k+1)*n_training_examples]], axis=0)
                                    if record_matrices:
                                        jacobian_prod_matrices = np.concatenate([self.prev_jacobian_prod_matrices, self.jacobian_prod_matrices[:(k+1)*n_training_examples]], axis=0)
                                        weight_prod_matrices   = np.concatenate([self.prev_weight_prod_matrices, self.weight_prod_matrices[:(k+1)*n_training_examples]], axis=0)

                            # save quick test error
                            np.save(os.path.join(self.simulation_path, "quick_test_errors.npy"), quick_test_errs)

                            if n == n_training_examples - 1:
                                # save test error
                                np.save(os.path.join(self.simulation_path, "full_test_errors.npy"), full_test_errs)

                                # save weights
                                self.save_weights(self.simulation_path, prefix="epoch_{}_".format(self.current_epoch + 1))

                            if record_backprop_angle and not use_backprop:
                                if self.M > 1:
                                    # save backprop angles
                                    np.save(os.path.join(self.simulation_path, "bp_angles.npy"), bp_angles)

                            if record_loss:
                                np.save(os.path.join(self.simulation_path, "final_layer_loss.npy"), losses)

                            if record_training_labels:
                                np.save(os.path.join(self.simulation_path, "training_labels.npy"), training_labels)

                            if record_plateau_times:
                                for m in range(self.M):
                                    np.save(os.path.join(self.simulation_path, "plateau_times_{}.npy".format(m)), self.plateau_times_full[m])

                            if record_training_error:
                                np.save(os.path.join(self.simulation_path, "training_errors.npy"), training_errors)

                            if record_eigvals:
                                # save eigenvalues
                                np.save(os.path.join(self.simulation_path, "max_jacobian_eigvals.npy"), max_jacobian_eigvals)
                                np.save(os.path.join(self.simulation_path, "max_weight_eigvals.npy"), max_weight_eigvals)
                                if record_matrices:
                                    np.save(os.path.join(self.simulation_path, "jacobian_prod_matrices.npy"), jacobian_prod_matrices)
                                    np.save(os.path.join(self.simulation_path, "weight_prod_matrices.npy"), weight_prod_matrices)

                            print("done. ", end="")

                    if record_eigvals:
                        # print the minimum max eigenvalue of (I - J_g*J_f).T * (I - J_g*J_f) from the last 1000 examples
                        print("Min max Jacobian eigval: {:.4f}. ".format(np.amin(self.max_jacobian_eigvals[max(0, k*n_training_examples + n - 999):k*n_training_examples + n + 1])), end="")

                        # print the number of max eigenvalues of (I - J_g*J_f).T * (I - J_g*J_f) from the last 1000 examples that were smaller than 1
                        print("# max eigvals < 1: {}. ".format(np.sum(self.max_jacobian_eigvals[max(0, k*n_training_examples + n - 999):k*n_training_examples + n + 1] < 1)), end="")

                    # get end time & reset start time
                    end_time = time.time()
                    time_elapsed = end_time - start_time
                    print("T: {0:.3f}s.\n".format(time_elapsed))
                    start_time = None

            # update latest epoch counter
            self.current_epoch += 1

        # record end time of training
        if save_simulation:
            with open(os.path.join(self.simulation_path, 'simulation.txt'), 'a') as simulation_file:
                sim_end_time = datetime.datetime.now()
                print("-----------------------------", file=simulation_file)
                print("End time: {}".format(sim_end_time), file=simulation_file)

    def test_weights(self, n_test=n_quick_test):
        '''
        Test the network's current weights on the test set. The network's layers are copied
        and restored to their previous state after testing.

        Arguments:
            n_test (int) : The number of test examples to use.
        '''

        global l_f_phase, integration_time

        # save old length of forward phase
        old_l_f_phase = l_f_phase

        # set new length of forward phase
        l_f_phase = l_f_phase_test

        # save old integration time
        old_integration_time = integration_time

        # set new integration time
        integration_time = integration_time_test

        old_x_hist = self.x_hist

        # initialize count of correct classifications
        num_correct = 0

        # shuffle testing data
        self.x_test, self.t_test = shuffle_arrays(self.x_test, self.t_test)

        digits = np.arange(10)

        # create new integration recording variables
        for m in xrange(self.M):
            self.l[m].create_integration_vars()

        for n in xrange(n_test):
            # clear all layer variables
            for m in xrange(self.M):
                self.l[m].clear_vars()

            # clear input spike history
            self.x_hist *= 0

            # get testing example data
            self.x = lambda_max*self.x_test[:, n][:, np.newaxis]
            self.t = self.t_test[:, n][:, np.newaxis]

            # do a forward phase & get the unit with maximum average somatic potential
            self.f_phase(self.x, self.t.repeat(self.n_neurons_per_category, axis=0), None, training=False)
            sel_num = np.argmax(np.mean(self.l[-1].average_C_f.reshape(-1, self.n_neurons_per_category), axis=-1))

            # get the target number from testing example data
            target_num = np.dot(digits, self.t)

            # increment correct classification counter if they match
            if sel_num == target_num:
                num_correct += 1

            # print every 100 testing examples
            if (n + 1) % 100  == 0:
                sys.stdout.write("\x1b[2K\rTesting example {0}/{1}. E: {2:05.2f}%.".format(n+1, n_test, (1.0 - float(num_correct)/(n+1))*100.0))
                sys.stdout.flush()

        # calculate percent error
        err_rate = (1.0 - float(num_correct)/n_test)*100.0

        if old_x_hist is not None:
            self.x_hist = old_x_hist

        integration_time = old_integration_time

        l_f_phase = old_l_f_phase

        # create new integration recording variables
        for m in xrange(self.M):
            self.l[m].create_integration_vars()

        # clear all layer variables
        for m in xrange(self.M):
            self.l[m].clear_vars()

        if n_test > 100:
            sys.stdout.write("\x1b[2K\r")
            sys.stdout.flush()

        return err_rate

    def save_weights(self, path, prefix=""):
        '''
        Save the network's current weights to .npy files.

        Arguments:
            path (string)   : The path of the folder in which to save the network's weights.
            prefix (string) : A prefix to append to the filenames of the saved weights.
        '''

        for m in xrange(self.M):
            np.save(os.path.join(path, prefix + "W_{}.npy".format(m)), self.W[m])
            np.save(os.path.join(path, prefix + "b_{}.npy".format(m)), self.b[m])
            if m != self.M - 1:
                np.save(os.path.join(path, prefix + "Y_{}.npy".format(m)), self.Y[m])
                if use_feedback_bias:
                    np.save(os.path.join(path, prefix + "c_{}.npy".format(m)), self.c[m])

    def load_weights(self, path, prefix=""):
        '''
        Load weights from .npy files and set them to the network's weights.

        Arguments:
            path (string)   : The path of the folder from which to load the weights.
            prefix (string) : Prefix appended to the filenames of the saved weights.
        '''

        print("Loading weights from \"{}\" with prefix \"{}\".".format(path, prefix))
        print("--------------------------------")

        for m in xrange(self.M):
            self.W[m] = np.load(os.path.join(path, prefix + "W_{}.npy".format(m)))
            self.b[m] = np.load(os.path.join(path, prefix + "b_{}.npy".format(m)))
            if m != self.M - 1:
                self.Y[m] = np.load(os.path.join(path, prefix + "Y_{}.npy".format(m)))
                if use_feedback_bias:
                    self.c[m] = np.load(os.path.join(path, prefix + "c_{}.npy".format(m)))

        # print network weights
        self.print_weights()

        print("--------------------------------")

# ---------------------------------------------------------------
"""                     Layer classes                         """
# ---------------------------------------------------------------

class Layer:
    def __init__(self, net, m):
        '''
        Initialize the layer.

        Arguments:
            net (Network) : The network that the layer belongs to.
            m (int)       : The layer number, eg. m = 0 for the first layer.
        '''

        self.net  = net
        self.m    = m
        self.size = self.net.n[m]

    def spike(self):
        '''
        Generate Poisson spikes based on the firing rates of the neurons.
        '''

        self.S_hist = np.concatenate([self.S_hist[:, 1:], np.random.poisson(self.lambda_C)], axis=-1)

class hiddenLayer(Layer):
    def __init__(self, net, m, f_input_size, b_input_size):
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

        self.f_input_size = f_input_size
        self.b_input_size = b_input_size

        self.A             = np.zeros((self.size, 1))
        self.B             = np.zeros((self.size, 1))
        self.C             = np.zeros((self.size, 1))
        self.lambda_C      = np.zeros((self.size, 1))

        self.S_hist        = np.zeros((self.size, mem), dtype=np.int8)

        self.E       = np.zeros((self.size, 1))
        self.delta_W = np.zeros(self.net.W[self.m].shape)
        self.delta_Y = np.zeros(self.net.Y[self.m].shape)
        self.delta_b = np.zeros((self.size, 1))

        self.average_C_f        = np.zeros((self.size, 1))
        self.average_C_t        = np.zeros((self.size, 1))
        self.average_A_f        = np.zeros((self.size, 1))
        self.average_A_t        = np.zeros((self.size, 1))
        self.average_lambda_C_f = np.zeros((self.size, 1))
        self.average_PSP_B_f    = np.zeros((self.f_input_size, 1))
        if update_feedback_weights:
            self.average_PSP_A_f = np.zeros((self.b_input_size, 1))

        self.alpha_f            = np.zeros((self.size, 1))
        self.alpha_t            = np.zeros((self.size, 1))

        # set integration counter
        self.integration_counter = 0

        # create integration variables
        self.create_integration_vars()

    def create_integration_vars(self):
        self.A_hist        = np.zeros((self.size, integration_time))
        self.PSP_A_hist    = np.zeros((self.b_input_size, integration_time))
        self.PSP_B_hist    = np.zeros((self.f_input_size, integration_time))
        self.C_hist        = np.zeros((self.size, integration_time))
        self.lambda_C_hist = np.zeros((self.size, integration_time))

    def clear_vars(self):
        '''
        Clear all layer variables.
        '''

        self.A             *= 0
        self.B             *= 0
        self.C             *= 0
        self.lambda_C      *= 0

        self.S_hist        *= 0
        self.A_hist        *= 0
        self.PSP_A_hist    *= 0
        self.PSP_B_hist    *= 0
        self.C_hist        *= 0
        self.lambda_C_hist *= 0

        self.E       *= 0
        self.delta_W *= 0
        self.delta_Y *= 0
        self.delta_b *= 0

        self.average_C_f        *= 0
        self.average_C_t        *= 0
        self.average_A_f        *= 0
        self.average_A_t        *= 0
        self.average_lambda_C_f *= 0
        self.average_PSP_B_f    *= 0
        if update_feedback_weights:
            self.average_PSP_A_f *= 0

        self.alpha_f            *= 0
        self.alpha_t            *= 0

        self.integration_counter = 0

    def update_W(self):
        '''
        Update feedforward weights.
        '''

        if not use_backprop:
            self.E = (self.alpha_t - self.alpha_f)*-k_B*lambda_max*deriv_sigma(self.average_C_f)

            if record_backprop_angle and not use_backprop and calc_E_bp:
                self.E_bp = (np.dot(self.net.W[self.m+1].T, self.net.l[self.m+1].E_bp)*k_B*lambda_max*deriv_sigma(self.average_C_f))
        else:
            self.E_bp = (np.dot(self.net.W[self.m+1].T, self.net.l[self.m+1].E_bp)*k_B*lambda_max*deriv_sigma(self.average_C_f))
            self.E    = self.E_bp

        if record_backprop_angle and (not use_backprop) and calc_E_bp:
            self.delta_b_bp = self.E_bp

        self.delta_W        = np.dot(self.E, self.average_PSP_B_f.T)
        self.net.W[self.m] += -self.net.f_etas[self.m]*P_hidden*self.delta_W

        self.delta_b        = self.E
        self.net.b[self.m] += -self.net.f_etas[self.m]*P_hidden*self.delta_b

    def update_Y(self):
        '''
        Update feedback weights.
        '''

        E_inv = (lambda_max*sigma(self.average_C_f) - self.alpha_f)*-deriv_sigma(self.average_A_f)

        self.delta_Y        = np.dot(E_inv, self.average_PSP_A_f.T)
        self.net.Y[self.m] += -self.net.b_etas[self.m]*self.delta_Y

        if use_feedback_bias:
            self.delta_c        = E_inv
            self.net.c[self.m] += -self.net.b_etas[self.m]*self.delta_c

    def update_A(self, b_input):
        '''
        Update apical potentials.

        Arguments:
            b_input (ndarray) : Feedback input.
        '''

        if use_spiking_feedback:
            self.PSP_A = np.dot(b_input, kappas)
        else:
            self.PSP_A = b_input

        self.PSP_A_hist[:, self.integration_counter % integration_time] = self.PSP_A[:, 0]

        if use_feedback_bias:
            self.A = np.dot(self.net.Y[self.m], self.PSP_A) + self.net.c[self.m]
        else:
            self.A = np.dot(self.net.Y[self.m], self.PSP_A)
        self.A_hist[:, self.integration_counter % integration_time] = self.A[:, 0]

    def update_B(self, f_input):
        '''
        Update basal potentials.

        Arguments:
            f_input (ndarray) : Feedforward input.
        '''

        if use_spiking_feedforward:
            self.PSP_B = np.dot(f_input, kappas)
        else:
            self.PSP_B = f_input

        self.PSP_B_hist[:, self.integration_counter % integration_time] = self.PSP_B[:, 0]

        self.B = np.dot(self.net.W[self.m], self.PSP_B) + self.net.b[self.m]

    def update_C(self):
        '''
        Update somatic potentials & calculate firing rates.
        '''

        if use_conductances:
            if use_apical_conductance:
                self.C_dot = -g_L*self.C + g_B*(self.B - self.C) + g_A*(self.A - self.C)
            else:
                self.C_dot = -g_L*self.C + g_B*(self.B - self.C)
            self.C += self.C_dot*dt
        else:
            self.C = k_B*self.B

        self.C_hist[:, self.integration_counter % integration_time] = self.C[:, 0]

        self.lambda_C = lambda_max*sigma(self.C)
        self.lambda_C_hist[:, self.integration_counter % integration_time] = self.lambda_C[:, 0]

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

        self.integration_counter = (self.integration_counter + 1) % integration_time

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

        self.integration_counter = (self.integration_counter + 1) % integration_time

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

            if update_feedback_weights:
                self.average_PSP_A_f = np.mean(self.PSP_A_hist, axis=-1)[:, np.newaxis]
        elif phase == "target":
            self.average_C_t        = np.mean(self.C_hist, axis=-1)[:, np.newaxis]
            self.average_lambda_C_t = np.mean(self.lambda_C_hist, axis=-1)[:, np.newaxis]

            if update_feedback_weights:
                self.average_PSP_A_t = np.mean(self.PSP_A_hist, axis=-1)[:, np.newaxis]

"""
NOTE: In the paper, we denote the output layer's somatic & dendritic potentials
      as U and V. Here, we use C & B purely in order to simplify the code.
"""
class finalLayer(Layer):
    def __init__(self, net, m, f_input_size):
        '''
        Initialize the final layer.

        Arguments:
            net (Network)      : The network that the layer belongs to.
            m (int)            : The layer number, ie. m = M - 1 where M is the total number of layers.
            f_input_size (int) : The size of feedforward input. This is the same as the
                                 the number of units in the previous layer.
        '''

        Layer.__init__(self, net, m)

        self.f_input_size = f_input_size

        self.B             = np.zeros((self.size, 1))
        self.I             = np.zeros((self.size, 1))
        self.C             = np.zeros((self.size, 1))
        self.lambda_C      = np.zeros((self.size, 1))

        self.S_hist        = np.zeros((self.size, mem), dtype=np.int8)

        self.E       = np.zeros((self.size, 1))
        self.delta_W = np.zeros(self.net.W[self.m].shape)
        self.delta_b = np.zeros((self.size, 1))

        self.average_C_f        = np.zeros((self.size, 1))
        self.average_C_t        = np.zeros((self.size, 1))
        self.average_lambda_C_f = np.zeros((self.size, 1))
        self.average_lambda_C_t = np.zeros((self.size, 1))
        self.average_PSP_B_f    = np.zeros((self.f_input_size, 1))

        # set integration counter
        self.integration_counter = 0

        # create integration variables
        self.create_integration_vars()

    def create_integration_vars(self):
        self.PSP_B_hist    = np.zeros((self.f_input_size, integration_time))
        self.C_hist        = np.zeros((self.size, integration_time))
        self.lambda_C_hist = np.zeros((self.size, integration_time))

    def clear_vars(self):
        '''
        Clear all layer variables.
        '''

        self.B             *= 0
        self.I             *= 0
        self.C             *= 0
        self.lambda_C      *= 0

        self.S_hist        *= 0
        self.PSP_B_hist    *= 0
        self.C_hist        *= 0
        self.lambda_C_hist *= 0

        self.E       *= 0
        self.delta_W *= 0
        self.delta_b *= 0

        self.average_C_f        *= 0
        self.average_C_t        *= 0
        self.average_lambda_C_f *= 0
        self.average_lambda_C_t *= 0
        self.average_PSP_B_f    *= 0

        self.integration_counter = 0

    def update_W(self):
        '''
        Update feedforward weights.
        '''

        self.E = (self.average_lambda_C_t - lambda_max*sigma(self.average_C_f))*-k_D*lambda_max*deriv_sigma(self.average_C_f)

        if use_backprop or (record_backprop_angle and calc_E_bp):
            self.E_bp = (self.average_lambda_C_t - lambda_max*sigma(self.average_C_f))*-k_D*lambda_max*deriv_sigma(self.average_C_f)

        self.delta_W        = np.dot(self.E, self.average_PSP_B_f.T)
        self.net.W[self.m] += -self.net.f_etas[self.m]*P_final*self.delta_W

        self.delta_b        = self.E
        self.net.b[self.m] += -self.net.f_etas[self.m]*P_final*self.delta_b

    def update_B(self, f_input):
        '''
        Update basal potentials.

        Arguments:
            f_input (ndarray) : Feedforward input.
        '''

        if use_spiking_feedforward:
            self.PSP_B = np.dot(f_input, kappas)
        else:
            self.PSP_B = f_input

        self.PSP_B_hist[:, self.integration_counter % integration_time] = self.PSP_B[:, 0]

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
            if use_conductances:
                self.I = g_E*(E_E - self.C) + g_I*(E_I - self.C)
            else:
                self.k_D2 = g_D/(g_L + g_D + g_E + g_I)
                self.k_E  = g_E/(g_L + g_D + g_E + g_I)
                self.k_I  = g_I/(g_L + g_D + g_E + g_I)

    def update_C(self, phase):
        '''
        Update somatic potentials & calculate firing rates.

        Arguments:
            phase (string) : Current phase of the network, "forward" or "target".
        '''

        if use_conductances:
            if phase == "forward":
                self.C_dot = -g_L*self.C + g_D*(self.B - self.C)
            elif phase == "target":
                self.C_dot = -g_L*self.C + g_D*(self.B - self.C) + self.I
            self.C += self.C_dot*dt
        else:
            if phase == "forward":
                self.C = k_D*self.B
            elif phase == "target":
                self.C = self.k_D2*self.B + self.k_E*E_E + self.k_I*E_I

        self.C_hist[:, self.integration_counter % integration_time] = self.C[:, 0]

        self.lambda_C = lambda_max*sigma(self.C)
        self.lambda_C_hist[:, self.integration_counter % integration_time] = self.lambda_C[:, 0]

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

        self.integration_counter = (self.integration_counter + 1) % integration_time

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

        self.integration_counter = (self.integration_counter + 1) % integration_time

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
