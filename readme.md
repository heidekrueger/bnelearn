![bnelearn-logo](docs/bnelearn-logo.png)

# A Framework for Equilibrium Learning in Sealed-Bid Auctions

[![pipeline status](https://gitlab.lrz.de/heidekrueger/bnelearn/badges/master/pipeline.svg)](https://gitlab.lrz.de/heidekrueger/bnelearn/commits/master) | [![coverage report](https://gitlab.lrz.de/heidekrueger/bnelearn/badges/master/coverage.svg)](https://gitlab.lrz.de/heidekrueger/bnelearn/commits/master)

**Maintainers**: Stefan Heidekrüger ([@heidekrueger](https://github.com/heidekrueger)), Nils Kohring ([@kohring](https://github.com/kohring)), Markus Ewert ([@Markus-Ewert](https://github.com/Markus-Ewert))

**Original Authors**: Stefan Heidekrüger, Paul Sutterer, Nils Kohring, Martin Bichler
**Contributors**: Markus Ewert, Gleb Kilichenko ([@kilichenko](https://github.com/kilichenko)), Carina Fröhlich

Currently, this repository contains minimal code to reproduce the experiments in our paper: "Learning Equilibria in Symmetric Auction Games using Artificial Neural Networks", published in _Nature Machine Intelligence_ [Link](https://www.nature.com/articles/s42256-021-00365-4).

TODO: Update this

#### Overview of What's Implemented

Running experiments for $n$-player Matrix and sealed-bid auction games with either

* Fictitious Play, Stochastic Fictitious Play, Mixed Fictitious Play in matrix games
* "Neural Pseudogradient Ascent" in a wide array of Auction games:
  * single-item auctions with first-, second- and third-price rules, with known-bne support for a wide range of settings.
  * Local-Global combinatorial auctions, in particular LLG and LLLLGG
    * for LLG we support bne for independent and correlated local bidders for several core-selecting payment rules
  * split-award and mineral-rights auctions
  
**TODO:** Update.

#### Table of Contents
1. [Installation](#Installation)
2. [Background](#Background)
3. [The bnelearn package](#package)
4. [Contribute](#Contribute)
5. [Citation](#Citation)

## 1. Installation and Running the Software <a name="Installation"></a>
See [Installation](installation.md).

**TODO:** With what's on Github.

**TODO:** Add example of customization.



## 4. Contribute: Before Your First Commit <a name="Contribute"></a>
Please read [Contributing](contributing.md) carefully and follow the set-up steps described there.

#### Git LFS
On a new machine, please make sure you have git-lfs installed and configured for this repository. (See [contributing.md](contributing.md) for details.)


## 5. Suggested Citation <a name="Citation"></a>
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
