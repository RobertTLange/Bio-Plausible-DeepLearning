import argparse
from utils.helpers import *
# Guergiev et al (2017) - Segregated Compartments DNN Learning
# Import Bayesian Optimization Module
from utils.bayesian_opt import BO_NN

# Import Network Architectures
from models.DNN import eval_dnn
from models.CNN import eval_cnn
from models.CompDNN import eval_comp_dnn

# Create all necessary directory if non-existent
global data_dir
data_dir = os.getcwd() +"/data"

if not os.path.exists(data_dir):
    os.makedirs(data_dir)
    print("Created New Data Directory")

# Create Log Directory or remove tensorboard log files in log dir
log_dir = os.getcwd() + "/logs"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
    print("Created New Log Directory")
else:
    filelist = [ f for f in os.listdir(log_dir) if f.startswith("events")]
    for f in filelist:
        os.remove(os.path.join(log_dir, f))
    print("Deleted Old TF/TensorBoard Log Files in Existing Log Directory")


download_data()

# Define Search Hyperspace for Bayesian Optimization on DNN architectures
hyper_space_dnn = {'batch_size': (10, 500),
                   'learning_rate': (0.0001, 0.05),
                   'num_layers': (1, 6),
                   'h_l_1': (30, 500),
                   'h_l_2': (30, 500),
                   'h_l_3': (30, 500),
                   'h_l_4': (30, 500),
                   'h_l_5': (30, 500),
                   'h_l_6': (30, 500)}

# Define Search Hyperspace for Bayesian Optimization on Compartmental DNN architectures
hyper_space_comp_dnn = {"use_sparse_feedback": (0, 1),
                        "use_conductances": (0, 1),
                        "use_broadcast": (0, 1),
                        "use_spiking_feedback": (0, 1),
                        "use_spiking_feedforward": (0, 1),
                        "use_symmetric_weights": (0, 1),
                        "noisy_symmetric_weights": (0, 1),
                        "update_feedback_weights": (0, 1),
                        "use_apical_conductance": (0, 1),
                        "use_weight_optimization": (0, 1),
                        "use_feedback_bias": (0, 1),

                        "dt" : (0.1, 1.0),
                        "l_f_phase": (2, 50),
                        "l_t_phase": (2, 50),
                        "l_f_phase_test": (50, 100),

                        "lambda_max": (0.2, 0.5),
                        "tau_s": (1, 5),
                        "tau_L": (7, 13),
                        "g_B": (0.3, 0.9),
                        "g_A": (0.02, 0.1),
                        "E_E": (5, 12),
                        "E_I": (-12, -5),

                        "f_etas": (0.01, 0.5),
                        "b_etas": (0.01, 0.5),
                        'num_layers': (1, 6),

                        'h_l_1': (30, 500),
                        'h_l_2': (30, 500),
                        'h_l_3': (30, 500),
                        'h_l_4': (30, 500),
                        'h_l_5': (30, 500),
                        'h_l_6': (30, 500)}

# Run Bayesian Optimization (UCB-Acquisition Fct) on DNN
hyper_space_cnn = {'batch_size': (10, 500),
                   'learning_rate': (0.0001, 0.05),
                   'num_layers': (1, 5),
                   'ch_1': (3, 64),
                   'ch_2': (3, 64),
                   'ch_3': (3, 64),
                   'ch_4': (3, 64),
                   'ch_5': (3, 64),
                   'k_1': (2, 10),
                   'k_2': (2, 10),
                   'k_3': (2, 10),
                   'k_4': (2, 10),
                   'k_5': (2, 10),
                   'stride': (1, 3),
                   'padding': (1, 3)}

bo_iters = 50
num_epochs = 10
k_fold = 3

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--type', action="store",
                        default="comp_dnn", type=str,
                        help='Model for which to run Bayesian Optimization')
    parser.add_argument('-d', '--dataset', action="store",
                        default="mnist", type=str,
                        help='Dataset on which to Bayesian Optimization')
    
    args = parser.parse_args()
    model_type = args.type
    dataset = args.dataset

    if model_type == "comp_dnn":
        model = eval_comp_dnn
        hyper_space = hyper_space_comp_dnn
    elif model_type == "dnn":
        model = eval_dnn
        hyper_space = hyper_space_dnn
    elif model_type == "cnn":
        model = eval_cnn
        hyper_space = hyper_space_cnn
    else:
        raise "Provide a valid Model!"

    # Run Bayesian Optimization (UCB-Acquisition Fct) for dataset
    opt_log = BO_NN(bo_iters, model, model_type, dataset,
                    hyper_space,
                    num_epochs, k_fold, logging=True, verbose=True)
