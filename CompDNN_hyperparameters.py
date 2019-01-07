n_full_test  = 10000 # number of examples to use for full tests  (every epoch)
n_quick_test = 100   # number of examples to use for quick tests (every 1000 examples)

nonspiking_mode         = True  # whether to run in non-spiking mode (real-valued outputs)
use_rand_phase_lengths  = True  # use random phase lengths (chosen from Wald distribution)
use_rand_plateau_times  = False # randomly sample the time of each neuron's apical plateau potential
use_conductances        = True  # use conductances between dendrites and soma
use_broadcast           = True  # use broadcast (ie. feedback to all layers comes from output layer)
use_spiking_feedback    = True  # use spiking feedback
use_spiking_feedforward = True  # use spiking feedforward input

use_symmetric_weights   = False # enforce symmetric weights
noisy_symmetric_weights = False # add noise to symmetric weights

use_sparse_feedback     = False # use sparse feedback weights
update_feedback_weights = False # update feedback weights
use_backprop            = False # use error backpropagation
use_apical_conductance  = False # use attenuated conductance from apical dendrite to soma
use_weight_optimization = True  # attempt to optimize initial weights
use_feedback_bias       = False # use biases in feedback paths
initial_test            = False # whether to do an initial test on the test set prior to training

record_backprop_angle   = False # record angle b/w hidden layer error signals and backprop-generated error signals
record_loss             = True  # record final layer loss during training
record_training_error   = True  # record training error during training
record_training_labels  = True  # record labels of images that were shown during training
record_phase_times      = False # record phase transition times across training
record_plateau_times    = False # record plateau potential times for each neuron across training
record_voltages         = False # record voltages of neurons during training (huge arrays for long simulations!)

# --- Jacobian testing --- #
record_eigvals          = False # record maximum eigenvalues for Jacobians
record_matrices         = False # record Jacobian product & weight product matrices (huge arrays for long simulations!)
plot_eigvals            = False # dynamically plot maximum eigenvalues for Jacobians

default_simulations_folder = 'Simulations/' # folder in which to save simulations (edit accordingly)
weight_cmap                = 'bone'         # color map to use for weight plotting

dt  = 1.0        # time step (ms)
mem = int(10/dt) # spike memory (time steps) - used to limit PSP integration of past spikes (for performance)

l_f_phase      = int(50/dt)  # length of forward phase (time steps)
l_t_phase      = int(50/dt)  # length of target phase (time steps)
l_f_phase_test = int(250/dt) # length of forward phase for tests (time steps)

integration_time      = l_f_phase - int(30/dt)      # time steps of integration of neuronal variables used for plasticity
integration_time_test = l_f_phase_test - int(30/dt) # time steps of integration of neuronal variables during testing

# 2nd set of hyperparameters
lambda_max = 0.2*dt # maximum spike rate (spikes per time step)

# kernel parameters
tau_s = 3.0  # synaptic time constant
tau_L = 10.0 # leak time constant

# conductance parameters
g_B = 0.6                                   # basal conductance
g_A = 0.05 if use_apical_conductance else 0 # apical conductance
g_L = 1.0/tau_L                             # leak conductance
g_D = g_B                                   # dendritic conductance in output layer

E_E = 8  # excitation reversal potential
E_I = -8 # inhibition reversal potential

# steady state constants
k_B = g_B/(g_L + g_B + g_A)
k_D = g_D/(g_L + g_D)
k_I = 1.0/(g_L + g_D)

# weight update constants
P_hidden = 20.0/lambda_max      # hidden layer error signal scaling factor
P_final  = 20.0/(lambda_max**2) # final layer error signal scaling factor

if nonspiking_mode:
    # set parameters for non-spiking mode
    use_rand_phase_lengths  = False
    use_rand_plateau_times  = False
    use_conductances        = False
    use_spiking_feedforward = False
    use_spiking_feedback    = False
    record_phase_times      = False
    record_plateau_times    = False
    record_voltages         = False

    l_f_phase             = 2
    l_t_phase             = 2
    l_f_phase_test        = 2
    integration_time      = 1
    integration_time_test = 1
    mem                   = 1

if use_rand_phase_lengths:
    # set minimum phase lengths
    min_l_f_phase = l_f_phase
    min_l_t_phase = l_t_phase
