==========================================
Installation and Reproduction Instructions
==========================================

These are minimal instructions to install dependencies and run the scripts to reproduce the results presented in our paper.


Prerequesites
=============

Current parameter settings assume that you are running on a Unix system with a cuda-enabled Nvidia GPU with at least 11GB GPU-RAM, and that the root directory (containing this readme file) is located at ~/bnelearn. On a system without a GPU (but at least 11GB of available RAM), everything should work, but you will experience significantly longer runtimes. On systems with less GPU-RAM, standard configurations may fail, in that case try adjusting the batch_size and eval_batch_size parameters in the run-scripts (and possibly the tests). Everything should likewise run on Windows, but you may need to change the output directories manually.


Installation
============

``bnelearn`` mainly runs on top of ``pytorch`` and python. The repository was extensively tested with python 3.9 and pytorch 1.10 on a Ubuntu system. 
The package should be cross-platform compatible, but we can make no guarantees. (A limited feature set has been tested on Windows 11.)


Installation instructions for Ubuntu using conda:

1. Install Python (tested on 3.9) and pip, we recommend using a separate environment using conda or virtualenv. Assuming you have a running installation of conda, make a new environment: ```conda create -n bnelearn-test python=3.9 pip``` Then activate your environment via ```conda activate bnelearn``` 


2. Optional -- you can skip directly to step 3, but this will install the package with a reduced feature set (e.g. no GPU support, no commercial solvers, no development extras.)  Make sure you are using the correct version of pip ()`which pip` should point to your new conda environment!) and install the remaining requirements via ``pip install -r requirements.txt``. This will install `all` requirements, including GPU-enabled pytorch, external solvers required for some combinatorial auctions and development tools. Note that ``requirements.txt`` will pull the latest stable torch version with cuda that is available at the time of writing (July 2021). You may want to manually install the latest version available for your system, see https://pytorch.org/get-started/locally/ for details.

3. Install the bnelearn package via ``pip install -e .``.

4. Test your installation via ``pytest``. If all tests pass, everything was successfull. You may also see ``SKIP``s or ``XFAIL``s: In this case, the, installation seems to work, but the tests have determined that you are missing some optional requirements for advanced features, or that your system does not support cuda. In this case, not all features may be available to you.

The framework conceptually consists of the following

* A python package ``bnelearn`` in the ``./bnelearn`` subdirectory.
* Some User-code scripts in ``./scripts``.
* Jupyter notebooks in the ``./notebooks`` directory that trigger experiments and log results to subdirectories (or other places) in tensorboard format.
* The ``R`` subdirectory contains scripts for parsing tensorboard logs into R dataframes, in order to create pretty figures. Some of this functionality uses r-tensorflow, see source code for installation instructions.



Running the software
====================

* Navigate to your local ``bnelearn`` folder (in the following: ``.``).
* Activate the ``bnelearn`` conda-env: ``conda activate bnelearn``.
* Execute one of the scripts in the `scripts` directory, or run a `jupyter lab` instance to run of the notebooks in the `notebooks` directory.


Experiment logging 
==================

**On the fly logging:** Results of script and notebook experiments are written to a subdirectory as specified in each script or notebook. To view the results or monitor training process, start a tensorboard instance:

* Navigate to your experiment output directory.
* In another terminal window, activate the ``bnelearn`` conda env as well: ``activate bnelearn``.
* Start a tensorboard instance, pointing at the relevant subdirectory for your experiment (tensorboard can simultaneously display multiple runs of the same experiment.) I.e. if you're interested in fpsb experiments and your directory structure is

    .. code-block:: bash

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

