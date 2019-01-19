import os
import time
import json

# Import Bayesian Optimization Module
from bayes_opt import BayesianOptimization, UtilityFunction
from bayes_opt.util import load_logs
from bayes_opt.event import Events
from bayes_opt.observer import JSONLogger

# Import Network Architectures
from models.DNN import eval_dnn
from models.CNN import eval_cnn

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
        next_point = check_next_point(next_point)

        if func_type == "cnn":
            while invalid_kernel_size(next_point, 28):
                next_point = optimizer.suggest(utility)
                next_point = check_next_point(next_point)

        # Add additional inputs to list - remove again from dict after fct call
        next_point["num_epochs"] = num_epochs
        next_point["k_fold"] = k_fold
        next_point["dataset"] = dataset
        target = eval_func(**next_point)
        del next_point["num_epochs"]
        del next_point["k_fold"]
        del next_point["dataset"]

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
        if W_in <= 0:
            return True
    return False


def check_next_point(next_point):
    # Assert/Enforce that params have correct type (dicrete/continuous) for BO
    for key in next_point.keys():
        if key != "learning_rate":
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


if __name__ == "__main__":
    a = "/logs/bo_logs_dnn_mnist.json"
    b = "/logs/bo_logs_dnn_mnist_session.json"

    merge_json_logs(a, b)
