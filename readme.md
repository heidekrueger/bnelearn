# Learning Equilibria in Bayesian Games


[![pipeline status](https://gitlab.lrz.de/heidekrueger/bnelearn/badges/master/pipeline.svg)](https://gitlab.lrz.de/heidekrueger/bnelearn/commits/master) | [![coverage report](https://gitlab.lrz.de/heidekrueger/bnelearn/badges/master/coverage.svg)](https://gitlab.lrz.de/heidekrueger/bnelearn/commits/master)

This repository contains a framework for finding Bayes Nash Equilibria through learning with Neural Networks.

# Current Status

### What works

* Running experiments for 2-player Matrix and Auction Games with either
    * 'Static environments':
    Players are named/numbered and fixed, each player has their own strategy model
    that is updated in-place in each iteration
    * 'Dynamic environments' for symmetric games:
    Players share one common model, that gets updated in each iteration based on playing against previous and current model in the environment.
    Updated model gets added into the environment, pushing out the oldest model currently in it.
* n-player Matrix games should work but haven't been tested yet

### What doesn't work yet
* n-player symmetric auctions with 'dynamic environment' (code generating opponent bid profile is incomplete)



# Installation

The framework conceptually consists of the following
* A python package `bnelearn` in the `./bnelearn` subdirectory.
* Jupyter notebooks in the `./notebooks` directory that trigger experiments and
  log results to subdirectories in tensorboard format.

To use the software, the following is required:
* A local copy of the software (via `git clone`)
* A python environment that contains all `bnelearn` requirements, especially pytorch.
    * Tested with python 3.7, pytorch 1.0.1, CUDA 10, cuDNN 7.1
* A running installation of jupyter with access to a Python-kernel of the above environment.
* A running installation of tensorboard.
    * Tested with python 3.6, tensorflow-gpu 1.13.1, CUDA 10, cuDNN 7.3.1
    * No gpu required here, so cpu tensorflow is absolutely sufficient.

The easies way to achieve the above is to create _two_ separate python environments, one for running the bnelearn code and jupyter, and another one for running tensorboard.
Separating the bnelearn/pytorch and tensorboard/tensorflow environments is desirable, because different release schedules
of pytorch and TensorFlow regularly lead to incompatible dependencies of the latest versions of both frameworks.

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
numpy, matplotlib, jupyter, pytorch and tensorboardX.
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

#### Install tensorboardX
tensorboardX is a wrapper package for pytorch, chainer and other frameworks to write output that can
be vizualized using TF's tensorboard function. `tensorboardX`is not available in the conda channels, but can only be
installed from source or using `pip`. The easiest way to install it is

* Install `pip` inside the bnelearn environment (with activated environment as above)
`conda install pip`

* Using the bnelearn-environment's pip, install tensorboardX

`pip install tensorboardX`

### Create another environment for tensorflow

* If necessary, deactivate the bnelearn env above
`deactivate` (on Linux/OSX: `source deactivate`)

* Create a new conda environment `tf` with the latest stable tensorflow in the conda channels.

To install CPU-only TF, use
`conda create -n tf tensorflow`
or, if your machine has a CUDA-compatible Nvidia GPU and you want GPU features (not used for bnelearn-vizualisation!)
`conda create -n tf tensorflow-gpu`

# Running the software

* Navigate to your local `bnelearn` folder (in the following: `.`)
* Activate the `bnelearn` conda-env: `activate bnelearn`
* Start a jupyter server using `jupyter lab`
* A browser window with jupyter should open automatically. If it doesn't you can find it at http://localhost:8888/lab.
* In jupyter lab, browse to the notebooks directory and run any of the notebooks there.

Results of notebook experiments are written to a subdirectory as specified in each notebook. To view the results
or monitor training process, start a tensorboard instance:
* Navigate to the `./notebooks/` directory.
* In another terminal window, activate the `tf` conda env: `activate tf`.
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

then start tensorboard using

`tensorboard --logdir fpsb`

The tensorboard server is then accessible at http://localhost:6006
