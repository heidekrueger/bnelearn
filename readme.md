# Learning Equilibria in Bayesian Games


[![pipeline status](https://gitlab.lrz.de/heidekrueger/bnelearn/badges/master/pipeline.svg)](https://gitlab.lrz.de/heidekrueger/bnelearn/commits/master) | [![coverage report](https://gitlab.lrz.de/heidekrueger/bnelearn/badges/master/coverage.svg)](https://gitlab.lrz.de/heidekrueger/bnelearn/commits/master)

This repository contains a framework for finding Bayes Nash Equilibria through learning with Neural Networks.

# Current Status

### What's implemented

Running experiments for n-player Matrix and sealed-bid auction Games with either

* Fictitious Play, Stochastic Fictitious Play, Mixed Fictitious Play in matrix games
* "Neural Pseudogradient Ascent" in a wide array of Auction games:
  * single-item auctions with first-, second- and third-price rules, with known-bne support for a wide range of settings.
  * Local-Global combinatorial auctions, in particular LLG and LLLLGG
    * for LLG we support bne for independent and correlated local bidders for several core-selecting payment rules
  * split-award and mineral-rights auctions



## Installation and Running the software
See [Installation](installation.md)

## Before your first commits
Please read [Contributing](contributing.md) carefully and follow the set-up steps described there.

### git lfs
On a new machine, please make sure you have git-lfs installed and configured for this repository. (See [contributing.md](contributing.md) for details.)