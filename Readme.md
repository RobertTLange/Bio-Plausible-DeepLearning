# Biological Plausible Deep Learning
## Author: Robert Tjarko Lange | December 2018

This project analyzes different learning rules in deep layered structures. More specifically, we explore alternatives to backpropagation (aka the chain rule). Weight transport (access to all weights at every layer of the backward pass) renders backpropagation biologically implausible. Recent alternatives explore local learning rules and draw inspiration from the compartmental design of pyramidal neurons.

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


## TODO - CODING:
* [ ] get_data - Different datasets - FashionMNIST, CIFAR 10
* [ ] Get Guergiev Code running/understand
* [ ] Restructure Guergiev code and integrate into current pipeline
* [ ] Add comments! - Look up pep8 standard for fcts/classes
* [ ] Work on weight visualization
* [ ] Work on error propagation comparison/delta W


## TODO - REPORT:
* [ ] Read papers/Add first notes of papers
    * [x] Lillicrap et al (2016)
    * [ ] Guergiev et al (2017)
    * [x] Bartunov et al (2018)
    * [ ] Sacramento et al (2018)
* [ ] Add first skeleton of report/sections

## Repository Structure
```
Bio-Plausible-DeepLearning
+- workspace.ipynb: Main workspace notebook - Execute for replication
```

## How to use this code
1. Clone the repo.
```
git clone https://github.com/RobertTLange/Bio-Plausible-DeepLearning && cd Bio-Plausible-DeepLearning
```
2. Create a virtual environment (optional but recommended).
```
virtualenv -p python BPDL
```
Activate the env (the following command works on Linux, other operating systems might differ):
```
source BPDL/bin/activate
```
3. Install all dependencies:
```
pip install -r requirements.txt
```
4. Run the main notebook:
```
jupyter notebook workspace.ipynb
```
