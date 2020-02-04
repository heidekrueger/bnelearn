# Installation

The framework conceptually consists of the following
* A python package `bnelearn` in the `./bnelearn` subdirectory.
* Some User-code scripts in `./scripts`
* Jupyter notebooks in the `./notebooks` directory that trigger experiments and
  log results to subdirectories (or other places) in tensorboard format.
* The `R` subdirectory contains scripts for parsing tensorboard logs into R dataframes, in order to create pretty figures. Some of this functionality uses r-tensorflow, see source code for installation instructions.

To use the bnelearn software, the following is required:
* A local copy of the software (via `git clone`)
* A python environment that contains all `bnelearn` requirements, especially pytorch.
    * Tested with python 3.7, pytorch 1.2.0, CUDA 10, cuDNN 7.6.2
* A running installation of jupyter with access to a Python-kernel of the above environment.
* For tensorboard vizualizations, a running version of tensorboard
    * Tested with tb-nightly 1.15.0a20190624. (tensorboard stable 1.14 is incompatible with some of the logging we use.)

The easiest way to achieve the above is to create a conda environment. If you want to use the latest tensorboard features, you may want to create separate environments,
separating the bnelearn/pytorch and tensorboard/tensorflow because different release schedules
of pytorch and TensorFlow regularly lead to incompatible dependencies of the latest versions of both frameworks.

There's a standalone Tensorboard installation that can just be installed in the bnelearn pytorch env. However, a full-blown tensorflow installation is
required for the R parsing to work. _Improve documentation on this._

On srvbichler14, the following two conda envs are installed for all users:
* `bnelearn` for everyrthing required to run the experiments. (especially pytorch+tensorboard, see requirements.txt)
 `/opt/anaconda/anaconda3/envs/bnelearn/bin/python`
* `r-tensorflow` with full tensorflow, for R-interoperability.  `/opt/anaconda/anaconda3/envs/r-tensorflow/bin/python`

## Creating the environments using conda

### Install conda

As all workloads will usually be run in specialized conda environments, installing `miniconda` is sufficient.
https://docs.conda.io/en/latest/miniconda.html
On windows: Install in user mode, **do NOT** choose 'install for all users'/admin mode as this will inevitably lead to permission problems later on.

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
Currently (25.08.2019) the latest stable version tensorboard 1.14 but this contains a bug that prevents the dashboard from updating when using with pytorch.
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
Results of notebook experiments are written to a subdirectory as specified in each notebook. (Similar for experiments in `scripts`. To view the results
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

`tensorboard --logdir fpsb [--port 6006]`

The standard port is 6006, but each user should use their own port to enable different sessions.
The tensorboard server is then accessible at http://localhost:6006

## Persistent Experiments
Persistent Experiments (for papers etc) should be logged (or copied) to a subdirectory
of
`/srv/bnelearn/`, which has been made globally read-writable.
When creating subdirectories, please set their permissions to 777.

# Remote development on GPU-Server

Since the June 2019 release, Visual Studio Code supports remote development via SSH which should be the most comfortable way to work with this package. Come and ask me (Stefan) how to set it up.

# Non-Python Requirements

* `jq` used for git-filter to clean up .ipynb outputs at git staging.
