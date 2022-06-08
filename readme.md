![bnelearn-logo](docs/bnelearn-logo.png)

# A Framework for Equilibrium Learning in Sealed-Bid Auctions

bnelearn is a framework for equilibrium learning in sealed-bid auctions and other markets that can be modeled as Bayesian Games. The documentation can be found at [Documentation](https://bnelearn.readthedocs.io/en/latest/#). Please bear in mind that the documentation is still a work in progress and may not be complete or up-to-date in all respects. We're working on improving it.

**Maintainers**: Stefan Heidekrüger ([@heidekrueger](https://github.com/heidekrueger)), Nils Kohring ([@kohring](https://github.com/kohring)), Markus Ewert ([@Markus-Ewert](https://github.com/Markus-Ewert)).

**Original Authors**: Stefan Heidekrüger, Paul Sutterer ([@PaulR-S](https://github.com/PaulR-S)), Nils Kohring, Martin Bichler.

**Further Contributors**: Gleb Kilichenko ([@kilichenko](https://github.com/kilichenko)), Carina Fröhlich, Anne Christopher ([@annechris13](https://github.com/annechris13)), Iheb Belgacem ([@belgacemi](https://github.com/belgacemi)).


## Suggested Citation
If you find `bnelearn` helpful and use it in your work, please consider using the following citation:

```
@misc{Heidekrueger2021,
  author = {Heidekr\"uger, Stefan and Kohring, Nils and Sutterer, Paul and Bichler, Martin},
  title = {{bnelearn}: A Framework for Equilibrium Learning in Sealed-Bid Auctions},
  year = {2021},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/heidekrueger/bnelearn}}
}
```


## What's Implemented?
Running experiments for $n$-player matrix and sealed-bid auction games with either

### Auctions and Other Games
* Single-item, multi-unit, LLG combinatorial auction, LLLLGG combinatorial auction.
* Priors and correlations: Uniform and normal priors that are either independent or Bernoulli or constant weight dependent. (Independent private values (IPV) and non-PV, e.g., common values.)
* Utility functions: Quasi-linear utility, either risk neutral, risk averse, or risk seeking.
* For combinatorial auctions: custom batched, cuda-enabled QP solver for quadratic auction rules + gurobi/cvxpy integration for arbitrary auctions stated as a MIP.
* Single-item auctions with first-, second-, and third-price rules, with known-BNE support for a wide range of settings.
* Local-Global combinatorial auctions, in particular LLG and LLLLGG
    * For LLG we support bne for independent and correlated local bidders for several core-selecting payment rules
* Split-award and mineral-rights auctions
* Tullock contest and crowd sourcing contest


### Algorithms
* Fictitious play, stochastic fictitious play, mixed fictitious play in matrix games.
* Neural self-play with directly computed policy gradients from [(Heinrich and Silver, 2016)](https://arxiv.org/abs/1603.01121), which is called ``PGLearner``.
* Neural pseudogradient ascent (NPGA), called ``ESPGLearner``, from [(Bichler et al., 2021)](https://www.nature.com/articles/s42256-021-00365-4).
* Particle swarm optimization (PSO), called ``PSOLearner``, from [(Kohring et al., 2022)](http://aaai-rlg.mlanctot.info/papers/AAAI22-RLG_paper_8.pdf).



## Where to Start?
* You can find the installation instructions at [Installation](docs/usage/installation).
* A quickstart guide is provided at [Quickstart](docs/usage/quickstart).
* Background information can be found under [Background](docs/usage/background).



## Contribute: Before Your First Commit

(If you see this on GitHub and want to contribute, please get in touch with one of the maintainers. The development version of this repository is hosted on a private GitLab server at TUM, we'll be happy to grant access on an individual basis.)

Please read [Contributing](contributing.md) carefully and follow the set-up steps described there.
**Git LFS**: On a new machine, please make sure you have git-lfs installed and configured for this repository. (See [contributing.md](contributing.md) for details.)
