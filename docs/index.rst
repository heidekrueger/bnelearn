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
* I(PV) and non-PV (e.g., common values), with arbitrary priors/correlation profiles, utilities, valuation/observation/bid dimensionalities.
* Modular organization allows for easy construction of new markets (e.g., bid languages, etc.) from existing or custom building blocks.
* Extensive metrics (learning-related: estimates of "equilibrium quality", utilities over time, market analysis: efficiency, revenue, individual payoffs) and built-in plotting capacities.
* Wide range of predefined settings and building blocks:
    * Learning rules: Policy Gradients, NPGA, PSO.
    * Auctions: Single-item, multi-unit, LLG combinatorial auction, LLLLGG combinatorial auction.
    * Priors/correlations: Uniform and normal priors that are either independent or Bernoulli or constant weight dependent.
    * Utility functions: Quasi-linear utility, either risk neutral, risk averse, or risk seeking.
* Fully vectorized, cuda enabled, massive parallelism
* For combinatorial auctions: custom batched, cuda-enabled QP solver for quadratic auction rules + gurobi/cvxpy integration for arbitrary auctions stated as a MIP.


Predefined Auction Settings
---------------------------

A diverse set of auction games is implemented in the framework.


Predefined Learners
-------------------

Algorithms for trying to iteratively learn equilibria implement the base class ``Learner`` in the framework. Two noteworthy algorithms that are contained are

* Neural self-play with directly computed policy gradients from `(Heinrich and Silver, 2016) <https://arxiv.org/abs/1603.01121>`_, which is called ``PGLearner``,
* Neural pseudogradient ascent (NPGA), called ``ESPGLearner``, from `(Bichler et al., 2021) <https://www.nature.com/articles/s42256-021-00365-4>`_,
* Particle swarm optimization (PSO), called ``PSOLearner``, from `(Kohring et al., 2022) <http://aaai-rlg.mlanctot.info/papers/AAAI22-RLG_paper_8.pdf>`_.


Limitations and Alternatives
============================

Same dimensionality for all players. Current implementations and learners use deterministic/pure continuous actions.

Variance reduction or Quasirandom sampling is not yet implemented but should be easy.


**Other existing multi-agent Learning Packages:** Other multi-agent learning frameworks, such as `OpenSpiel <https://github.com/deepmind/open_spiel>`_ that is a collection of games and algorithms for reinforcement learning and planning or `PettingZoo <https://github.com/Farama-Foundation/PettingZoo>`_ that is a multi-agent extension of the famous `OpenAI Gym <https://github.com/openai/gym>`_ framework, mainly focus on zero-sum games and on games with discrete action spaces. Crucially, they neither allow an efficient evaluation of running a large batch of games in parallel.



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
