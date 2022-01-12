"""
Runs predefined experiments with individual parameters
fire.Fire() asks you to decide for one of the experiments defined above
by writing its name and define the required (and optional) parameters
e.g.:
    experiment.py single_item_uniform_symmetric 1 20 [2,3] 'first_price'

alternatively instead of fire.Fire() use, e.g.:
    single_item_uniform_symmetric(1,20,[2,3],'first_price')

"""
import os
import sys

import torch
from itertools import product

# put bnelearn imports after this.
# pylint: disable=wrong-import-position
sys.path.append(os.path.realpath('.'))
sys.path.append(os.path.join(os.path.expanduser('~'), 'bnelearn'))

from bnelearn.experiment.configuration_manager import ConfigurationManager  # pylint: disable=import-error
from bnelearn.strategy import NeuralNetStrategy


if __name__ == '__main__':

    # Path is user-specific
    log_root_dir = os.path.join(
        os.path.expanduser('~'), 'bnelearn', 'experiments', 'asymmetric-2022',
        )

    sigma = .1
    specific_gpu = 5


    # 1. Individual experiments
    if False:
        # Common parameters #######################################################
        n_epochs = 2000
        n_runs = 10

        # Single-item asymmetric experiments ######################################
        experiment_types = [
            'single_item_asymmetric_uniform_overlapping',
            'single_item_asymmetric_uniform_disjunct',
            'single_item_asymmetric_beta',
            ]

        for experiment_type in experiment_types:
            experiment_config, experiment_class = \
                ConfigurationManager(
                    experiment_type=experiment_type,
                    n_runs=n_runs, n_epochs=n_epochs,
                    ) \
                .set_learning(
                    learner_hyperparams={
                        'population_size': 64,
                        'sigma': sigma,
                        'scale_sigma_by_model_size': True
                        },
                    ) \
                .set_logging(
                    util_loss_batch_size=2**12,
                    util_loss_grid_size=2**10,
                    util_loss_frequency=n_epochs,
                    plot_frequency=500,
                    log_root_dir=log_root_dir,
                    best_response=True,
                    cache_eval_actions=True,
                    save_models=True,
                    ) \
                .set_hardware(
                    specific_gpu=specific_gpu,
                ) \
                .get_config()
            experiment = experiment_class(experiment_config)
            experiment.run()
            torch.cuda.empty_cache()


        # Asymmetric LLG experiment ###############################################
        experiment_config, experiment_class = \
            ConfigurationManager(
                experiment_type='llg_full',
                n_runs=n_runs, n_epochs=n_epochs,
                ) \
            .set_setting(
                payment_rule='mrcs_favored',
                ) \
            .set_learning(
                batch_size=2**17,
                   learner_hyperparams={
                       'population_size': 64,
                       'sigma': sigma,
                       'scale_sigma_by_model_size': True
                       },
                ) \
            .set_logging(
                eval_batch_size=2**17,
                util_loss_batch_size=2**11,
                util_loss_grid_size=2**10,
                util_loss_frequency=n_epochs,
                best_response=True,
                plot_frequency=500,
                cache_eval_actions=True,
                log_root_dir=log_root_dir,
                save_models=True,
                ) \
            .set_hardware(
                specific_gpu=specific_gpu
                ) \
            .get_config()
        experiment = experiment_class(experiment_config)
        experiment.run()
        torch.cuda.empty_cache()


        # Split-award auction #####################################################
        experiment_config, experiment_class = \
            ConfigurationManager(
                experiment_type='splitaward',
                n_runs=n_runs, n_epochs=n_epochs
                ) \
            .set_learning(
                learner_hyperparams={
                    'population_size': 64,
                    'sigma': sigma,
                    'scale_sigma_by_model_size': True
                    },
                ) \
            .set_logging(
                util_loss_batch_size=2**12,
                util_loss_grid_size=2**10,
                util_loss_frequency=n_epochs,
                best_response=True,
                plot_frequency=500,
                cache_eval_actions=True,
                log_root_dir=log_root_dir,
                save_models=True,
                ) \
            .set_hardware(
                specific_gpu=specific_gpu
                ) \
            .get_config()
        experiment = experiment_class(experiment_config)
        experiment.run()
        torch.cuda.empty_cache()


    if True:
        ### LLLLGG combinatorial experiment ###
        n_runs = [10, 2]
        n_epochss = [5000, 1000]
        payment_rules = ['first_price', 'nearest_vcg']
        population_sizes = [64, 32]
        batch_sizes = [2**18, 2**14]
        util_loss_batch_sizes = [2**12, 2**7]
        util_loss_grid_sizes = [2**10, 2**8]
        
        for n_run, n_epochs, payment_rule, population_size, batch_size, \
            util_loss_batch_size, util_loss_grid_size in zip(n_runs, \
            n_epochss, payment_rules, population_sizes, batch_sizes, \
            util_loss_batch_sizes, util_loss_grid_sizes):
            experiment_config, experiment_class = \
                ConfigurationManager(
                    experiment_type='llllgg',
                    n_runs=n_run, n_epochs=n_epochs
                    ) \
                .set_setting(
                    core_solver='mpc',
                    payment_rule=payment_rule
                    ) \
                .set_learning(
                    batch_size=batch_size,
                    pretrain_iters=500,
                    learner_hyperparams={
                        'population_size': population_size,
                        'sigma': sigma,
                        'scale_sigma_by_model_size': True
                        },
                    ) \
                .set_logging(
                    log_root_dir=log_root_dir,
                    util_loss_batch_size=util_loss_batch_size,
                    util_loss_grid_size=util_loss_grid_size,
                    util_loss_frequency=n_epochs,
                    plot_frequency=500,
                    cache_eval_actions=True,
                    save_models=True,
                    ) \
                .set_hardware(
                    specific_gpu=specific_gpu
                    ) \
                .get_config()
            experiment = experiment_class(experiment_config)
            experiment.run()
            torch.cuda.empty_cache()


    # 2. Scalability experiment
    if False:
        n_epochs = 500
        sigma = .1
        # population_sizes = [16, 32, 64]
        # batch_size = 2**18
        # for population_size in population_sizes:
        batch_sizes = [2**6, 2**10, 2**18]
        population_size = 64
        for batch_size in batch_sizes:
            experiment_config, experiment_class = \
                ConfigurationManager(
                    experiment_type='llllgg',
                    n_runs=3, n_epochs=n_epochs
                    ) \
                .set_setting(
                    core_solver='mpc',
                    payment_rule='first_price'
                    ) \
                .set_learning(
                    batch_size=batch_size,
                    pretrain_iters=500,
                    # optimizer_hyperparams={'lr': 1e-3},
                    learner_hyperparams={
                        'population_size': population_size,
                        'sigma': sigma,
                        'scale_sigma_by_model_size': True
                        },
                    ) \
                .set_logging(
                    # eval_batch_size=2**14,
                    log_root_dir=log_root_dir,
                    util_loss_batch_size=2**12,
                    util_loss_grid_size=2**10,
                    util_loss_frequency=50,
                    plot_frequency=n_epochs,
                    cache_eval_actions=True,
                    save_tb_events_to_csv_detailed=True,
                    save_models=True,
                    ) \
                .set_hardware(
                    specific_gpu=7
                    ) \
                .get_config()
            experiment = experiment_class(experiment_config)
            experiment.run()
            torch.cuda.empty_cache()