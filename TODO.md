# Biological Plausible Deep Learning
## Author: Robert Tjarko Lange | December 2018

## DONE:

* [x] PyTorch MLP/CNN baseline for MNIST
* [x] Create remote repo
* [x] Generalize network architecture to variable inputs
* [x] Write update_logger, process_logger function
* [x] Plot learning curves - output from logger
* [x] Add Xavier init for networks
* [x] Rewrite architecture and simplify code
* [x] Tried running in colab
* [x] Set up bayesian optimization pipeline - BayesianOptimization
    * [x] implement cross-validation with torch data/skorch
    * [x] one fct taking in hyperparams, return objective
    * [x] write fct that transforms cont variables to discrete
    * [x] check how to add folds/add input to eval_nn, BO pipeline
 	* [x] Generalize BO pipeline to CNN
    * [x] Write fct that checks if BO CNN proposal is valid (kernel/in/out)
    * [x] Add logging to BO pipeline
* [x] get_data - Different datasets - FashionMNIST, CIFAR 10
* [x] Add plotting of all 3 datasets
* [x] Get models running on all three datasets
* [x] Get Guergiev Code running/understand
* [x] Evaluate the model more frequently - not only once per epoch
* [x] Record weight changes
* [x] Work on weight visualization/changes in weights!
* [x] Work on error propagation comparison/delta W (||W_t - W_t-1||/||W_t||)
* [x] Run BO for 10 Epochs and 50 evaluations/BO iterations for all 3 datasets
* [x] Get best/worst performance, standard dev - plot as bar chart across approaches DNN/CNN/Guergiev
* [x] Restructure for python 3? +: cleaner folder structure, -: New env setup :(
* [x] Check what is wrong with reload of BO logs
* [x] Make weight logging not store whole weights but only stats!
* [x] Change weight change plot for all three datasets
* [ ] Restructure Guergiev code and integrate into current pipeline
    * [x] Delete random stuff
    * [x] Write a (weight)-logger module for Guergiev
    * [x] Write a get_test_error function for Guergiev
    * [x] Change plots for all three model types
    * [x] Check what is wrong with difference computations - weights not properly updates?
    * [x] Write an eval_comp_dnn function

## TODO - CODING:

* [ ] Optimize the code - run faster time it!
* [ ] Add a BO pipeline for guergiev
    * [ ] Load params from file
    * [ ] Update check params function
* [ ] Run CNN and plot performance
* [ ] Run BO pipeline for CNNs - figure out memory usage
* [ ] Add comments! - Look up pep8 standard for fcts/classes
* [ ] Finalize plots

## TODO - REPORT/PRESENTATION:

* [ ] Read papers/Add first notes of papers
    * [x] Lillicrap et al (2016)
    * [x] Guergiev et al (2017)
    * [x] Bartunov et al (2018)
    * [x] Sacramento et al (2018)
* [ ] Add first skeleton of report/sections - max 10 pages
    * [x] Backprop/Notation
    * [ ] Literature Notes
* [x] Overview figure - Solution approaches
* [ ] Overview of hyperparameter spaces
