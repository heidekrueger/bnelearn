```diff
- +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
- +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
```
# AAMAS 2020
**_This is a snapshot of this project as it was used for the AAMAS 2020 submission. (15th November 2019) Please see `conda-env-export.yaml` for full list of dependencies including package versions. (created using `conda env export`)_**

* `/experiments` contains a tar archive that includes skripts used to run the experiments as well as all artefacts.
* 2 player asymmetric setting was run from the respective notebook with incrementing seeds set manually, all other experiments run via script (see above).

```diff
- +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
- +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
```




# Learning Equilibria in Bayesian Games


[![pipeline status](https://gitlab.lrz.de/heidekrueger/bnelearn/badges/master/pipeline.svg)](https://gitlab.lrz.de/heidekrueger/bnelearn/commits/master) | [![coverage report](https://gitlab.lrz.de/heidekrueger/bnelearn/badges/master/coverage.svg)](https://gitlab.lrz.de/heidekrueger/bnelearn/commits/master)

This repository contains a framework for finding Bayes Nash Equilibria through learning with Neural Networks.

# Current Status

### What's implemented

Running experiments for n-player Matrix and single-item one-shot auction Games with either
* "Neural Self Play" using ES strategy learning.:
  * Players are kept track of in an 'environment', each player's strategy is updated using an 'optimizer'.
  * Players can have distinct strategy models (that are updated with distinct optimizers) or share a strategy model in symmetric settings.
* Fictitious Play, Stochastic Fictitious Play, Mixed Fictitious Play

### What's next
* Combinatorial Auctions
* Sequential Games
* Algorithmic changes

## Installation and Running the software
See [Installation](installation.md)

### git filters
When developing on a new machine, run 
```git config --local include.path ../.gitconfig```
in your repository root once.
This ensures the filters in `.gitconfig` will be applied to commits (e.g. cleaning up notebook output)

### git lfs
On a new machine, please make sure you have git-lfs installed and configured for this repository. (See [contributing.md](contributing.md) for details.