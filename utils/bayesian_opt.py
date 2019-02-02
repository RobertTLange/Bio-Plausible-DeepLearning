import os
import time
import json
import numpy as np

# Import Bayesian Optimization Module
from bayes_opt import BayesianOptimization, UtilityFunction
from bayes_opt.util import load_logs
from bayes_opt.event import Events
from bayes_opt.observer import JSONLogger

# Import Network Architectures
from models.DNN import eval_dnn
from models.CNN import eval_cnn
from models.CompDNN import eval_comp_dnn

from utils.helpers import update_tensor_dim
# Dont print depreciation warning
import warnings
warnings.filterwarnings("ignore")


def BO_NN(num_evals, eval_func, func_type, dataset, hyper_space,
          num_epochs, k_fold, logging, verbose):

    optimizer = BayesianOptimization(
        f=eval_func,
        pbounds=hyper_space,
        verbose=2,
        random_state=1,
    )

    log_fname = "./logs/bo_logs_" + func_type + "_" + dataset + ".json"
    temp_fname = "./logs/bo_logs_" + func_type + "_" + dataset + "_session.json"

    # Try to merge logs if previous BO opt fct was interrupted
    merge_json_logs(log_fname, temp_fname)
    prev_iters = 0
    if os.path.isfile(log_fname):
        prev_iters = get_iter_log(log_fname)
        load_logs(optimizer, logs=[log_fname])
        print("Loaded previously existing Log with {} BO iterations.".format(prev_iters))

    if logging:
        logger = JSONLogger(path=temp_fname)
        print("Start Logging to {}".format(log_fname))
        optimizer.subscribe(Events.OPTMIZATION_STEP, logger)

    utility = UtilityFunction(kind="ucb", kappa=2.5, xi=0.0)

    # Define printing template for verbose
    template = "BO iter {:>2} | cv-acc: {:.4f} | best-acc: {:.4f} | time: {:.2f}"

    for _ in range(num_evals):
        tic = time.time()
        next_point = optimizer.suggest(utility)
        next_point = check_next_point(next_point, func_type)

        if func_type == "cnn":
            if dataset != "cifar10":
                dim_in = 28
            else:
                dim_in = 32 
            while invalid_kernel_size(next_point, dim_in):
                # Sample random point if BO suggestion fails!
                next_point = sample_random_point(hyper_space)
                next_point = check_next_point(next_point, func_type)
        if func_type != "comp_dnn":
            # Add additional inputs to list - remove from dict after fct call
            next_point["num_epochs"] = num_epochs
            next_point["k_fold"] = k_fold
            next_point["dataset"] = dataset
            target = eval_func(**next_point)
            del next_point["num_epochs"]
            del next_point["k_fold"]
            del next_point["dataset"]
        else:
            with open(func_type + "_" + dataset + '_params_temp.json', 'w') as fp:
                json.dump(next_point, fp)
            target = eval_func(dataset, func_type + "_" + dataset + '_params_temp.json',
                               num_epochs, k_fold)

        optimizer.register(params=next_point, target=target)
        time_t = time.time() - tic

        if verbose:
            print(template.format(prev_iters + _ + 1, target,
                                  optimizer.max['target'], time_t))

    # Finally merge both logs into single log
    merge_json_logs(log_fname, temp_fname)
    return optimizer


def invalid_kernel_size(next_point, input_size):
    # Checks if kernel sizes is actually computable with cnn setup and specific
    # input dimensionality
    k_sizes = [next_point["k_1"], next_point["k_2"],
               next_point["k_3"], next_point["k_4"],
               next_point["k_5"]][:next_point["num_layers"]]

    padding = next_point["padding"]
    stride = next_point["stride"]
    W_in = input_size
    for i in range(len(k_sizes)):
        W_in = update_tensor_dim(W_in, k_sizes[i], padding, stride)
        W_in = update_tensor_dim(W_in, 2, 0, 2)

        if W_in <= 0 or not W_in.is_integer():
            return True
    return False


def check_next_point(next_point, func_type):
    # Assert/Enforce that params have correct type (dicrete/continuous) for BO
    if func_type != "comp_dnn":
        for key in next_point.keys():
            if key != "learning_rate":
                next_point[key] = int(round(next_point[key]))
    else:
        round_keys = ["use_sparse_feedback", "use_conductances",
                      "use_broadcast", "use_spiking_feedback",
                      "use_spiking_feedforward", "use_symmetric_weights",
                      "noisy_symmetric_weights", "update_feedback_weights",
                      "use_apical_conductance", "use_weight_optimization",
                      "use_feedback_bias", "l_f_phase", "l_t_phase",
                      "l_f_phase_test", "integration_time",
                      "integration_time_test", "num_layers",
                      "h_l_1", "h_l_2", "h_l_3", "h_l_4", "h_l_5", "h_l_6"]

        for key in next_point.keys():
            if key in round_keys:
                next_point[key] = int(round(next_point[key]))
    return next_point


def merge_json_logs(fname1, fname2):
    try:
        with open(fname1, "a") as outfile:
            with open(fname2) as infile:
                while True:
                    try:
                        iteration = next(infile)
                    except StopIteration:
                        break
                    iteration = json.loads(iteration)
                    outfile.write(json.dumps(iteration) + "\n")

        os.remove(fname2)
        print("Merged JSONs - Total its: {}".format(get_iter_log(fname1)))
        print("Removed temporary log file.")
    except:
        return


def get_iter_log(fname):
    counter = 0
    with open(fname) as outfile:
        while True:
            try:
                iteration = next(outfile)
                counter += 1
            except StopIteration:
                break
    return counter


def sample_random_point(hyper_space):
    random_point = {}
    for var, var_range in hyper_space.items():
        random_point[var] = np.random.uniform(var_range[0], var_range[1], 1)[0]
    return random_point


if __name__ == "__main__":
    a = "/logs/bo_logs_dnn_mnist.json"
    b = "/logs/bo_logs_dnn_mnist_session.json"

    merge_json_logs(a, b)
