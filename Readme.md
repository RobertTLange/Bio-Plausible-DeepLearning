# Biological Plausible Deep Learning
## Author: Robert Tjarko Lange | December 2018

This project analyzes different learning rules in deep layered structures. More specifically, we explore alternatives to backpropagation (aka the chain rule). Weight transport (access to all weights at every layer of the backward pass) renders backpropagation biologically implausible. Recent alternatives explore local learning rules and draw inspiration from the compartmental design of pyramidal neurons.

## DONE:

* [x] PyTorch MLP/CNN baseline for MNIST
* [x] Create remote repo
* [x] Generalize network architecture to variable inputs

## TODO:

* [ ] Write update_logger function
* [ ] Plot learning curves - output from logger
* [ ] Extend to different datasets - CIFAR 10
* [ ] Try paperspace/remote gpu testing
* [ ] Read papers
* [ ] Add first notes of papers


## Structure of Report:

    1. General Intro/Motivation/Structure Outline
    2. Backpropagation and Compartmental Learning Rules
        a. Base Theory Introduction
        b. Problems with bio plausibility
        c. Pyramidal neurons/Larkum
        d. Model Guergiev 2017
    3. Comparison across different datasets
