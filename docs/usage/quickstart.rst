==========
Quickstart
==========

Let us take a look at how to learn in one of the predefined auction games with one of the predefined algorithms.


Running the Experiments
=======================

The commands here can be used to reproduce the results published in `(Bichler et al., 2021) <https://www.nature.com/articles/s42256-021-00365-4>`_. All references refer to the numbering in that paper.

You can then run the experiments underlying the tables and figures in the main paper via ``python run_experiments_npga.py`` (see `run_experiments_npga.py <https://github.com/heidekrueger/bnelearn/blob/main/run_experiments_npga.py>`_). The standard configuration of the file reruns the experiments behind Figure 3, i.e., one run each for all combinations of the correlation strength and the risk parameter. To run the other experiments reported in the paper, make the following changes in lines 17 to 34:

* For the experiments underlying Figure 2, set ``n_runs = 10``, ``gammas = [0.5]``, ``payment_rules = ['nearest_vcg']``.
* For the experiments underlying Table 1, set ``n_runs = 10``, ``risks = [1.0]``, ``gammas = [0.5]``, ``payment_rules = ['vcg', 'nearest_bid', 'nearest_zero', 'nearest_vcg', 'first_price']``.

To run the other experiments reported in the Supplementary Information, make the following changes:

* For the experiments underlying Table S.1, set ``n_runs = 10``, ``risks = [1.0]`` or ``[0.5]``, ``gammas = [0.0]``, ``payment_rules = ['first_price']``, ``corr_models = ['independent']``, ``experiment_type = 'single_item_uniform_symmetric'`` or ``'single_item_gaussian_symmetric'`` and additionally supply the corresponding number of bidders via the ``n_players`` parameter to the ``set_setting()`` call.
* For the experiments underlying Figure S.1, set ``n_runs = 10``, ``risks = [1.0]``, ``gammas = [0.0]``, ``payment_rules = ['first_price']``, ``corr_models = ['independent']``, ``experiment_type = 'single_item_gaussian_symmetric`` and additionally supply the number of bidders via ``n_players = 10`` within the ``set_setting()`` call.
* For the experiments underlying Table S.2, set ``n_runs = 10``, ``risks = [1.0]``, ``gammas = [None]`` (fallback to default), ``payment_rules = ['first_price']`` for the Affiliated values setting and ``'second_price'`` for the Common value settings, ``corr_models = [None]`` (fallback to default), ``experiment_type = 'affiliated_observations'`` or ``'mineral_rights'`` and additionally supply the corresponding number of bidders via the ``n_players`` parameter to the ``set_setting()`` call.
* For the experiments underlying Table S.3, set ``n_runs = 10``, ``risks = [1.0]``, ``gammas = [0.0]``, ``payment_rules = ['first_price', 'uniform', 'vcg']``, ``corr_models = ['independent']``, ``experiment_type = 'multiunit'`` and additionally supply the corresponding number of bidders via the ``n_players`` parameter and the number of units via the ``n_units`` parameter to the ``set_setting()`` call.
* For the experiments underlying Figure S.2, make the corresponding changes to the previous experiments.
* For the experiments underlying Figure S.3, make the corresponding changes to the previous experiments and set ``corr_models = ['additive']``.
* For the experiments underlying Figure S.2, make the corresponding changes to the previous experiments.

Logs will be written into the ``experiments`` subdirectory. While the experiments are running, you can examine the training progress via ``tensorboard --logdir experiments``.


Running Experiments for Learning with PSO
=========================================

You can then run the experiments for particle swarm optimization (from `Kohring et al., 2022 <http://aaai-rlg.mlanctot.info/papers/AAAI22-RLG_paper_8.pdf>`_) via ``python run_experiments_pso.py`` (see `run_experiments_pso.py <https://github.com/heidekrueger/bnelearn/blob/main/run_experiments_pso.py>`_).


Running Experiments for Learning in Contests
============================================

You can then run the experiments for particle swarm optimization via ``python run_experiments_contests.py`` (see `run_experiments_contests.py <https://github.com/heidekrueger/bnelearn/blob/main/run_experiments_contestss.py>`_).
