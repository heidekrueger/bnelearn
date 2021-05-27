# bnelearn: A Framework for Equilibrium Learning in Sealed-Bid Auctions
Authors: Stefan Heidekrüger, Paul Sutterer, Nils Kohring, Martin Bichler

Currently, this repository contains minimal code to reproduce the experiments in our forthcoming paper: "Learning Equilibria in Symmetric Auction Games using
Artificial Neural Networks".

Over the next months, we will release the entire framework, so stay tuned.

## Suggested Citation
If you find `bnelearn` helpful and use it in your work, please consider using the following citation:

```@misc{Heidekrueger2021,
  author = {Heidekr\"uger, Stefan and Kohring, Nils and Sutterer, Paul and Bichler, Martin},
  title = {{bnelearn}: A Framework for Equilibrium Learning in Sealed-Bid Auctions},
  year = {2021},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/heidekrueger/bnelearn}}
}
```

We'd also be happy to hear from you, so please reach out with any questions.

## Installation and Reproduction Instructions

These are minimal instructions to install dependencies and run the scripts to reproduce the results presented in our paper. 

### Prerequesites 

Current parameter settings assume that you are running on a Unix system with a cuda-enabled Nvidia GPU with at least 11GB GPU-RAM, and that the root directory (containing this readme file) is located at `~/bnelearn`. 
On a system without a GPU (but at least 11GB of available RAM), everything should work, but you will experience significantly longer runtimes.
On systems with less GPU-RAM, standard configurations may fail, in that case try adjusting the `batch_size` and `eval_batch_size` parameters in the run-scripts (and possibly the tests).
Everything should likewise run on Windows, but you may need to change the output directories manually.

### Installation

1. Install Python (tested on 3.8) and pip, we recommend using a separate environment using `conda` or `virtualenv`.
1. If you have a CUDA-enabled GPU, follow the pytorch (tested on 1.8.1) installation instructions for your system at https://pytorch.org/get-started/locally/ (for a CPU-only installation, you can skip directly to the next step.)
1. Install the remaining dependencies via `pip install -r requirements.txt`.
1. Run the included tests via `pytest`. If installation was successfull, all tests should pass on GPU-systems. (On cpu-only systems, some tests will be skipped.)


### Running the experiments

@Nils: what's the easiest way? we should ideally provide scripts that run everything behind the tables and figures?

You can then run the experiments via `python ./scripts/run_inderdependent.py` (from the root directory).

Logs will be written into the `experiments` subdirectory. While the experiments are running, you can examine the training progress via `tensorboard --logdir experiments`.