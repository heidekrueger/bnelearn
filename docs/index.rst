.. bnelearn documentation master file

====
Home
====

Welcome to `bnelearn`'s documentation! bnelearn is a framework for equilibrium learning in sealed-bid auctions.

* You can find the installation instructions at :doc:`/usage/installation`.
* A quickstart guide is provided at :doc:`/usage/quickstart`.
* Background information can be found under :doc:`/usage/background`.


.. toctree::
   :caption: Contents
   :maxdepth: 2

   self
   usage/installation
   usage/quickstart
   usage/background
   modules


Features
========

In this section, we will present the `bnelearn` package with its most essential features and the auction games and learning algorithms it contains.

Structure
---------
* I(PV) and non-PV (e.g. common values), with arbitrary priors/correlation profiles, utilities, valuation/observation/bid dimensionalities.
* modular organization allows for easy construction of new markets (e.g. bid languages / ...) from existing or custom building blocks
* extensive metrics (learning-related: estimates of "equilibrium quality", utilities over time, market analysis: efficiency, revenue, individual payoffs) and built-in plotting capacities.
* wide range of predefined settings and building blocks. (Learning rules: NPGA, PSO, Bosshards(???), Auctions (...), Priors/Correlations (...), utility functions, ...)


* fully vectorized, cuda enabled, massive parallelism
* Variance Reduction easy-to-use built-in / Quasirandom sampling (not yet implemented, but should be easy)
* for combinatorial auctions: custom batched, cuda-enabled QP solver for quadratic auction rules + gurobi/cvxpy integration for arbitrary auctions stated as a MIP.


Predefined Auction Settings
---------------------------

A diverse set of auction games is implemented in the framework.


Predefined Learners
-------------------

Algorithms for trying to iteratively learn equilibria implement the base class `Learner` in the framework. Two noteworthy algorithms that are contained are (i.) neural self-play with directly computed policy gradients from \cite{heinrich2016deep}, which is called `PGLearner`, and (ii.) neural pseudogradient ascent, `ESPGLearner`, from \cite{bichler2021LearningEquilibriaSymmetric}

**TODO:** Add PSO.


Limitations and Alternatives
============================

Same dimensionality for all players. Current implementations and learners use deterministic/pure continuous actions.


**Other exxisting multi-agent Learning Packages:** Other multi-agent learning frameworks, such as OpenSpiel \citep{lanctotEtAl2019OpenSpiel} that is a collection of games and algorithms for reinforcement learning and planning or PettingZoo \citep{terry2020pettingzoo} that is a multi-agent extension of the famous OpenAI Gym framework \citep{OpenAIGym}, mainly focus on zero-sum games and on games with discrete action spaces. Crucially, they neither allow an efficient evaluation of running a large batch of games in parallel.



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
