# Code for NeurIPS Submission

©2020 the authors, all rights reserved. For Review at NeurIPS only. Do not distribute.

This repository contains the code to fully reproduce the experiments in the paper and supplementary material.
It is a minimal, anonymized excerpt of a larger repository under development by the authors.

# Experiment Results

The subdirectory `Experiments` contains the raw artefacts of the experiments presented in the paper.
Some artefacts (e.g. intermediate plots, tensorboard logs) have been ommitted due to file size limitations.
The `R` repository contains code to reproduce the tables and figures from these raw artefacts.

The following are included
* full evaluation logs in csv format (containing metrics for every iteration of each repitition) for each setting
* aggregate logs for each setting (containing only the last iteration of each repitition)
* Trained `pytorch` models for each setting (of the first run, i.e. with seed `0`). `model_0.pt` refers to the model of the local bidders, `model_2` (LLG) or `model_4` (LLLLGG) to that of the global bidders.


# How to run the code


You may also run the code to generate experiment results yourself.

The main implementation is contained in the `bnelearn` subdirectory, which constitutes a python package.

## Requirements
The main dependencies of this repo are `python=3.7`and `pytorch=1.4`. The code is designed to run on a `cuda` GPU with 11GB of RAM (e.g. Nvidia GeForce 2080TI). 
The code should run on CPU-only systems with sufficient RAM but is expected to fail on systems with smaller GPUs.

A full list of requirements can be found in `requirements.txt`. Note that we use a `conda` environment, but some dependencies have to be installed via `pip` (see comments in requirements file.)

## Running the code.

The experiments can be run by executing the script `scripts/run_NeurIPS_experiments.py`. The script will run a single setting  10 times (with seeds 0,1,...,9).
To adjust the settings, change the parameters in lines 27 (LLG or LLLLGG) and 30 (pricing rule).