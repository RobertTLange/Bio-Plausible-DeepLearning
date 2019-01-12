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

## TODO - CODING:

* [ ] Restructure for python 3? +: cleaner folder structure, -: New env setup :(
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

## Repository Structure
```
Bio-Plausible-DeepLearning
+- workspace.ipynb: Main workspace notebook - Execute for replication
```

## (Basic) How to use this code
1. Clone the repo.
```
git clone https://github.com/RobertTLange/Bio-Plausible-DeepLearning
cd Bio-Plausible-DeepLearning
```
2. Create a virtual environment (optional but recommended).
```
virtualenv -p python BioDL
```
Activate the env (the following command works on Linux, other operating systems might differ):
```
source BioDL/bin/activate
```
3. Install all dependencies:
```
pip install -r requirements.txt
```
4. Run the main notebook:
```
jupyter notebook workspace_*.ipynb
```


## (Advanced) Jupyter Env on AWS EC2 Instance Setup

During the course of this project I trained many models. Running the Bayesian Optimization (BO) pipeline takes a while. More specifically, we run 50 iterations of the BO pipeline

1. Clone repo, Create/Activate the environment and install dependencies
```
git clone https://github.com/RobertTLange/Bio-Plausible-DeepLearning
cd Bio-Plausible-DeepLearning
conda create --name BioDL python=2.7 --no-default-packages
source activate BioDL
pip install -r requirements.txt --quiet
```
2. Add ipykernel to listed env kernels, Launch notebook silent and open port (start a screen session in between!)
```
python -m ipykernel install --user --name BioDL --display-name "Python2 (BioDL)"
jupyter notebook --no-browser --port=8080
```
3. In new terminal window on local machine rewire port and listen
```
ssh -i keyname.pem -N -f -L localhost:2411:localhost:8080 user@MACHINE_IP_ADDRESS
```
4. In Browser open localhost port and start working on the notebook of choice. If required copy paste the token/set a password. Afterwards run notebook of choice
```
localhost:2411
```
5. After computations are done either git add, comnmit, push to remote repo or copy files back.
```
scp -i keyname.pem -r user@MACHINE_IP_ADDRESS:Bio-Plausible-DeepLearning .
```

## Jupyter Env Cleanup
conda env remove -n BioDL
jupyter kernelspec uninstall BioDL
