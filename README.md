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

You can then run the experiments underlying the tables and figures in the main paper via `python run_experiments.py`. The standard configuration of the file reruns the experiments behind Figure 3, i.e. one run each for all combinations of the correlation strength and the risk parameter. To run the other experiments reported in the paper, make the following changes in lines 17 to 34:

* For the experiments underlying Figure 2, set `n_runs = 10`, `gammas = [0.5]`, `payment_rules = ['nearest_vcg']`.
* For the experiments underlying Table 1, set `n_runs = 10`, `risks=[1.0]`, `gammas = [0.5]`, `payment_rules = ['vcg', 'nearest_bid', 'nearest_zero', 'nearest_vcg', 'first_price']`

Logs will be written into the `experiments` subdirectory. While the experiments are running, you can examine the training progress via `tensorboard --logdir experiments`.