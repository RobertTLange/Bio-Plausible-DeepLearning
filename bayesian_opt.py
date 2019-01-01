import time
# Import Bayesian Optimization Module
from bayes_opt import BayesianOptimization, UtilityFunction
from bayes_opt.observer import JSONLogger
from bayes_opt.event import Events
from bayes_opt.util import load_logs

# Import Network Architectures
from DNN import eval_dnn
from CNN import eval_cnn, update_tensor_dim

# Dont print depreciation warning
import warnings
warnings.filterwarnings("ignore")

def BO_NN(num_evals, eval_func, func_type, hyper_space, verbose,
          reload_log_fname=None):

    logger = JSONLogger(path="./logs/bo_logs_"
                        + time.strftime("%Y%m%d_%H%M%S") + ".json")

    optimizer = BayesianOptimization(
        f=eval_func,
        pbounds=hyper_space,
        verbose=2,
        random_state=1,
    )

    if reload_log_fname is not None:
        logger = load_logs(optimizer, logs=["/logs/" + reload_log_fname])

    optimizer.subscribe(Events.OPTMIZATION_STEP, logger)

    utility = UtilityFunction(kind="ucb", kappa=2.5, xi=0.0)

    # Define printing template for verbose
    template = "BO iter {:>2} | cv-acc: {:.4f} | best-acc: {:.4f} | time: {:.2f}"

    for _ in range(num_evals):
        tic = time.time()
        next_point = optimizer.suggest(utility)
        next_point = check_next_point(next_point)

        if func_type == "cnn":
            print(next_point, invalid_kernel_size(next_point, 28))
            while invalid_kernel_size(next_point, 28):
                next_point = optimizer.suggest(utility)
                print(next_point)

        target = eval_func(**next_point)
        optimizer.register(params=next_point, target=target)
        time_t = time.time() - tic

        if verbose:
            print(template.format(_ + 1, target, optimizer.max['target'], time_t))
    return optimizer

def update_tensor_dim(W_in, k_size, padding, stride):
    return (W_in - k_size + 2*padding)/stride + 1

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
        if W_in <= 0:
            return True
    return False

def check_next_point(next_point):
    # Assert/Enforce that params have correct type (dicrete/continuous) for BO
    for key in next_point.keys():
        if key != "learning_rate":
            next_point[key] = int(round(next_point[key]))
    return next_point
