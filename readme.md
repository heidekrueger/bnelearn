# Learning Equilibria in Bayesian Games


[![pipeline status](https://gitlab.lrz.de/heidekrueger/bnelearn/badges/master/pipeline.svg)](https://gitlab.lrz.de/heidekrueger/bnelearn/commits/master) | [![coverage report](https://gitlab.lrz.de/heidekrueger/bnelearn/badges/master/coverage.svg)](https://gitlab.lrz.de/heidekrueger/bnelearn/commits/master)

This repository contains a framework for finding Bayes Nash Equilibria through learning with Neural Networks.

# Current Status

### What works

Running experiments for n-player Matrix and single-item one-shot auction Games with either
* "Neural Self Play" using ES strategy learning.:
  * Players are kept track of in an 'environment', each player's strategy is updated using an 'optimizer'.
  * Players can have distinct strategy models (that are updated with distinct optimizers) or share a strategy model in symmetric settings.
* Fictitious Play, Stochastic Fictitious Play, Mixed Fictitious Play



# Installation

The framework conceptually consists of the following
* A python package `bnelearn` in the `./bnelearn` subdirectory.
* Jupyter notebooks in the `./notebooks` directory that trigger experiments and
  log results to subdirectories (or other places) in tensorboard format.
* The `R` subdirectory contains scripts for parsing tensorboard logs into R dataframes, in order to create pretty figures.

To use the software, the following is required:
* A local copy of the software (via `git clone`)
* A python environment that contains all `bnelearn` requirements, especially pytorch.
    * Tested with python 3.7, pytorch 1.0.1, CUDA 10, cuDNN 7.1
* A running installation of jupyter with access to a Python-kernel of the above environment.
* A running installation of tensorboard.
    * Tested with python 3.6, tensorflow-gpu 1.14, CUDA 10, cuDNN 7.3.1
    * No gpu required here, so cpu tensorflow is absolutely sufficient.

The easies way to achieve the above is to create _two_ separate python environments, one for running the bnelearn code and jupyter, and another one for running tensorboard.
Separating the bnelearn/pytorch and tensorboard/tensorflow environments is desirable, because different release schedules
of pytorch and TensorFlow regularly lead to incompatible dependencies of the latest versions of both frameworks.
_Update 12.08.2019: Starting in TF 1.14, there's a standalone Tensorboard installation that can just be installed in the bnelearn pytorch env. However, a full-blown tensorflow installation is
required for the R parsing to work. Improve documentation on this._
Right now, the following two conda envs are installed for all users:
* `bnelearn` for everyrthing required to run the experiments. (especially pytorch+tensorboard, see requirements.txt)
 `/opt/anaconda/anaconda3/envs/bnelearn/bin/python`
* `r-tensorflow` with full tensorflow, for R-interoperability.  `/opt/anaconda/anaconda3/envs/r-tensorflow/bin/python`

## Creating the environments using conda

### Install conda

As all workloads will usually be run in specialized conda environments, installing `miniconda` is sufficient.
https://docs.conda.io/en/latest/miniconda.html
On windows: Install in user mode, do NOT choose 'install for all users'/admin mode as this will inevitably lead to permission problems later on.

Once conda is running, update to latest version

`conda update conda`

### Create a conda env for bnelearn

#### Create a conda environment named `bnelearn` (or name of your choice)

This environment will contain all required dependencies to run the experiment code, i.e.
numpy, matplotlib, jupyter, pytorch and tensorboard.
Start by creating the environment:

`conda create -n bnelearn python=3.7`

#### Activate the environment
Windows: `activate bnelearn`, Linux/OSX: `source activate bnelearn`

#### Install bnelearn-requirements from conda

`conda install numpy matplotlib jupyter jupyterlab`

#### Install pytorch

Using conda from the pytorch-channel on Windows:
`conda install pytorch torchvision cudatoolkit=10.0 -c pytorch`
or equivalent command for your system (https://pytorch.org/get-started/locally/)

#### Install tensorboard
Tensorboard is part of the `tensorflow` family and since TF 1.14 is available as a standalone app that's compatible with other DL frameworks - like pytorch.
Currently (25.06.2019) tensorboard 1.14 contains a bug that prevents the dashboard from updating when using with pytorch.
As a workaround, we'll temporarily install the tensorboard nightly build using `pip`. The easiest way to install it is

* Install `pip` inside the bnelearn environment (with activated environment as above)
`conda install pip`

* Using the bnelearn-environment's pip, install tensorboard

`pip install tb-nightly` (replace by `conda install tensorboard` once 1.15 is in the conda channels)

### Create another environment for tensorflow

* If necessary, deactivate the bnelearn env above
`deactivate` (on Linux/OSX: `source deactivate` and/or `conda deactivate`)


# Running the software

* Navigate to your local `bnelearn` folder (in the following: `.`)
* Activate the `bnelearn` conda-env: `activate bnelearn`
* Start a jupyter server using `jupyter lab`
* A browser window with jupyter should open automatically. If it doesn't you can find it at http://localhost:8888/lab.
* In jupyter lab, browse to the notebooks directory and run any of the notebooks there.


# Experiment logging 
## On the fly logging
Results of notebook experiments are written to a subdirectory as specified in each notebook. To view the results
or monitor training process, start a tensorboard instance:
* Navigate to the `./notebooks/` directory.
* In another terminal window, activate the `bnelearn` conda env as well: `activate bnelearn`.
* Start a tensorboard instance, pointing at the relevant subdirectory for your experiment (tensorboard can simultaneously display multiple runs of the same experiment.) I.e. if you're interested in fpsb experiments and your directory structure is

```
    ./
    |
    *---* notebooks/
        |
        *---* fpsb/
            |
            *--- run1/
            *--- run2/
```
This folder structure should be used for work in progress. 

then start tensorboard using

`tensorboard --logdir fpsb`

The tensorboard server is then accessible at http://localhost:6006

## Persistent Experiments
Persistent Experiments (for papers etc) should be logged (or copied) to a subdirectory
of
`/srv/bnelearn/`, which has been made globally read-writable.
When creating subdirectories, please set their permissions to 777.

# Remote development on GPU-Server

Since the June 2019 release, Visual Studio Code supports remote development via SSH which should be the most comfortable way to work with this package. Come and ask me (Stefan) how to set it up.
