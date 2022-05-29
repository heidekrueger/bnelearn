
==========================================
Installation and Reproduction Instructions
==========================================

These are minimal instructions to install dependencies and run the scripts to reproduce the results presented in our paper.


Prerequesites
=============

Current parameter settings assume that you are running on a Unix system with a cuda-enabled Nvidia GPU with at least 11GB GPU-RAM, and that the root directory (containing this readme file) is located at ~/bnelearn. On a system without a GPU (but at least 11GB of available RAM), everything should work, but you will experience significantly longer runtimes. On systems with less GPU-RAM, standard configurations may fail, in that case try adjusting the batch_size and eval_batch_size parameters in the run-scripts (and possibly the tests). Everything should likewise run on Windows, but you may need to change the output directories manually.


Installation
============

Install Python (tested on 3.8) and pip, we recommend using a separate environment using conda or virtualenv. If you have a CUDA-enabled GPU, follow the pytorch (tested on 1.8.1) installation instructions for your system at https://pytorch.org/get-started/locally/ (for a CPU-only installation, you can skip directly to the next step.) Install the remaining dependencies via pip install -r requirements.txt. Run the included tests via pytest. If installation was successfull, all tests should pass on GPU-systems. (On cpu-only systems, some tests will be skipped.)

1. (Recommended but optional) Install the requirements via ``pip install -r requirements.txt``. This will install `all` requirements, including GPU-enabled pytorch, external solvers required for some combinatorial auctions and development tools. Note that ``requirements.txt`` will pull the latest stable torch version with cuda that is available at the time of writing (July 2021). You may want to manually install the latest version available for your system, see https://pytorch.org/get-started/locally/ for details.
2. Install the bnelearn package via ``pip install -e .``.
3. Test your installation via ``pytest``. If all tests pass, everything was successfull. You may also see ``SKIP``s or ``XFAIL``s: In this case, the, installation seems to work, but the tests have determined that you are missing some optional requirements for advanced features, or that your system does not support cuda. In this case, not all features may be available to you.

The framework conceptually consists of the following

* A python package ``bnelearn`` in the ``./bnelearn`` subdirectory.
* Some User-code scripts in ``./scripts``.
* Jupyter notebooks in the ``./notebooks`` directory that trigger experiments and log results to subdirectories (or other places) in tensorboard format.
* The ``R`` subdirectory contains scripts for parsing tensorboard logs into R dataframes, in order to create pretty figures. Some of this functionality uses r-tensorflow, see source code for installation instructions.


Creating the environments using conda
-------------------------------------

Install conda
~~~~~~~~~~~~~

As all workloads will usually be run in specialized conda environments, installing ``miniconda`` is sufficient. https://docs.conda.io/en/latest/miniconda.html
On Windows: Install in user mode, **do NOT** choose 'install for all users'/admin mode as this will inevitably lead to permission problems later on. Once conda is running, update to latest version ``conda update conda``.


**Create a conda environment for bnelearn**

Create a conda environment named ``bnelearn`` (or name of your choice).

This environment will contain all required dependencies to run the experiment code, i.e. numpy, matplotlib, jupyter, pytorch and tensorboard.
Start by creating the environment:

.. code-block:: bash

    conda create -n bnelearn python=3.7


**Activate the environment**

Windows: ``activate bnelearn``, Linux/OSX: ``source activate bnelearn``


Install bnelearn-requirements from conda:

.. code-block:: bash

    conda install numpy matplotlib jupyter jupyterlab

Install pytorch: Using conda from the pytorch-channel on Windows:

.. code-block:: bash

    conda install pytorch torchvision cudatoolkit=10.0 -c pytorch

or equivalent command for your system (https://pytorch.org/get-started/locally/)

Install tensorboard: Tensorboard is part of the ``tensorflow`` family and since TF 1.14 is available as a standalone app that's compatible with other DL frameworks - like pytorch.

* Install ``pip`` inside the bnelearn environment (with activated environment as above) ``conda install pip``.
* Using the bnelearn-environment's pip, install tensorboard with ``pip install tb-nightly`` (replace by ``conda install tensorboard`` once 1.15 is in the conda channels).

Create another environment for tensorflow: If necessary, deactivate the bnelearn env above ``deactivate`` (on Linux/OSX: ``source deactivate`` and/or ``conda deactivate``).


Running the software
====================

* Navigate to your local ``bnelearn`` folder (in the following: ``.``).
* Activate the ``bnelearn`` conda-env: ``activate bnelearn``.
* Start a jupyter server using ``jupyter lab``.
* A browser window with jupyter should open automatically. If it doesn't you can find it at http://localhost:8888/lab.
* In jupyter lab, browse to the notebooks directory and run any of the notebooks there.


Experiment logging 
==================

**On the fly logging:** Results of notebook experiments are written to a subdirectory as specified in each notebook. (Similar for experiments in ``scripts``. To view the results or monitor training process, start a tensorboard instance:

* Navigate to the ``./notebooks/`` directory.
* In another terminal window, activate the ``bnelearn`` conda env as well: ``activate bnelearn``.
* Start a tensorboard instance, pointing at the relevant subdirectory for your experiment (tensorboard can simultaneously display multiple runs of the same experiment.) I.e. if you're interested in fpsb experiments and your directory structure is

    .. code-block:: txt

        ./
        |
        *---* notebooks/
            |
            *---* fpsb/
                |
                *--- run1/
                *--- run2/

This folder structure should be used for work in progress.  Then start tensorboard using

.. code-block:: bash

    tensorboard --logdir fpsb [--port 6006]

The standard port is 6006, but each user should use their own port to enable different sessions. The tensorboard server is then accessible at http://localhost:6006.

