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

## TODO - CODING:

* [ ] Check what is wrong with reload of BO logs
* [ ] Make weight logging not store whole weights but only stats!
* [ ] Restructure Guergiev code and integrate into current pipeline
* [ ] Optimize the code - run faster time it!
* [ ] Run BO pipeline for CNNs - figure out memory usage
* [ ] Add a BO pipeline for guergiev
* [ ] Add comments! - Look up pep8 standard for fcts/classes



## TODO - REPORT:

* [ ] Read papers/Add first notes of papers
    * [x] Lillicrap et al (2016)
    * [ ] Guergiev et al (2017)
    * [x] Bartunov et al (2018)
    * [x] Sacramento et al (2018)
    * [ ] Larkum (2013)
    * [ ] Whittington, Bogacz (2017)
* [ ] Add first skeleton of report/sections - max 10 pages
    * [x] Backprop/Notation
    * [ ] Literature Notes
* [ ] Overview figure - Problems with backprop
* [x] Overview figure - Solution approaches
