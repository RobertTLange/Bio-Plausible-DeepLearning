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

## TODO - CODING:
* [ ] Set up bayesian optimization pipeline - BayesianOptimization
    * [x] implement cross-validation with torch data/skorch
    * [x] one fct taking in hyperparams, return objective
    * [x] write fct that transforms cont variables to discrete
    * [ ] check how to add folds/add input to eval_nn, BO pipeline
 	* [x] Generalize BO pipeline to CNN
* [ ] Get Guergiev Code running/understand
* [ ] get_data - Different datasets - FashionMNIST, CIFAR 10

## TODO - REPORT:
* [ ] Read papers
* [ ] Add first notes of papers
* [ ] Add comments! - Look up pep8 standard for fcts/classes

## Structure of Report:

    1. General Intro/Motivation/Structure Outline
    2. Backpropagation and Compartmental Learning Rules
        a. Base Theory Introduction
        b. Problems with bio plausibility
        c. Pyramidal neurons/Larkum
        d. Model Guergiev 2017
    3. Comparison across different datasets
        a. MNIST
        b. Fashion-MNIST
        c. CIFAR10
    4. Hyperparameter Robustness/Model Selection Analysis
