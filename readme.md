# Code for NeurIPS Submission

©2020 the authors, all rights reserved. For Review at NeurIPS only. Do not distribute.

This repository contains the code to reproduce the experiments in the paper. It is a minimal excerpt of a larger repository under development by the authors.

# Experiment Results

The subdirectory `Experiments` contains the raw data of the experiments presented in the paper.
Some artefacts (e.g. plots) have been ommitted due to file size limitations.

the following are included
* full evaluation logs (after each iteration of every repitition)
* Trained model for each setting (of a single run)


# How to run the code

You may also run the code to generate experiment results yourself.

## Requirements
The main dependency of this repo is `pytorch 1.4`. The code is designed to run on a `cuda` GPU with 11GB of RAM (e.g. Nvidia GeForce 2080TI). 
The code should run on CPU-only systems with sufficient RAM but is expected to fail on systems with smaller GPUs.

A full list of requirements can be found in `requirements.txt`. Note that we use a `conda` environment, but some dependencies have to be installed via `pip` (see comments in requirements file.)

The experiments can be run by executing the script `scripts/run_NeurIPS_experiments.py`. It will run each setting 10 times (with seeds 0,1,...,9).