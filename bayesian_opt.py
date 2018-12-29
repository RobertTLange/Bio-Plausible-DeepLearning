# Import Bayesian Optimization Module
from bayes_opt import BayesianOptimization, UtilityFunction

# Import Network Architectures
from DNN import eval_dnn

# Dont print depreciation warning
import warnings
warnings.filterwarnings("ignore")

def BO_NN(num_evals, eval_func, hyper_space):
    optimizer = BayesianOptimization(
        f=eval_func,
        pbounds=hyper_space,
        verbose=2,
        random_state=1,
    )

    utility = UtilityFunction(kind="ucb", kappa=2.5, xi=0.0)

    for _ in range(num_evals):
        next_point = optimizer.suggest(utility)
        next_point = check_next_point_dnn(next_point)
        target = eval_dnn(**next_point)
        optimizer.register(params=next_point, target=target)
        print(target, next_point)

    return(optimizer.max)


def check_next_point_dnn(next_point):
    # Assert/Enforce that params have correct type (dicrete/continuous) for BO
    for key in next_point.keys():
        if key != "learning_rate":
            next_point[key] = int(round(next_point[key]))
    return next_point


if __name__ == "__main__":
    BO_DNN(num_evals=2)
