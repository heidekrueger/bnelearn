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
        os.path.expanduser('~'), 'bnelearn', 'experiments', 'asymmmetric-final-results'
        )

    # Common parameters #######################################################
    n_epochs = 2000
    n_runs = 10
    sigma = .1
    specific_gpu = 7

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
                util_loss_grid_size=2**12,
                util_loss_frequency=n_epochs,
                plot_frequency=500,
                log_root_dir=log_root_dir,
                best_response=True,
                cache_eval_actions=True,
                log_metrics = {
                    'opt': True,
                    'util_loss': True,
                    'epsilon': True,
                    },
                save_models=True
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
            util_loss_grid_size=2**11,
            util_loss_frequency=n_epochs,
            best_response=True,
            plot_frequency=500,
            cache_eval_actions=True,
            log_root_dir=log_root_dir,
            save_models=True,
            log_metrics = {
                'opt': True,
                'util_loss': True,
                'epsilon': True,
                },
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
            util_loss_grid_size=2**12,
            util_loss_frequency=n_epochs,
            best_response=True,
            plot_frequency=500,
            cache_eval_actions=True,
            log_root_dir=log_root_dir,
            log_metrics = {
                'opt': True,
                'util_loss': True,
                'epsilon': True,
                },
            ) \
        .set_hardware(
            specific_gpu=specific_gpu
            ) \
        .get_config()
    experiment = experiment_class(experiment_config)
    experiment.run()
    torch.cuda.empty_cache()


    ### LLLLGG combinatorial experiment ###
    experiment_config, experiment_class = \
        ConfigurationManager(
            experiment_type='llllgg',
            n_runs=5, n_epochs=10000
            ) \
        .set_setting(
            core_solver='mpc',
            payment_rule='nearest_vcg'
            ) \
        .set_learning(
            batch_size=2**8,
            learner_hyperparams={
                'population_size': 32,
                'sigma': sigma,
                'scale_sigma_by_model_size': True
                },
            ) \
        .set_logging(
            eval_batch_size=2**8,
            log_root_dir=log_root_dir,
            util_loss_batch_size=2**4,
            util_loss_grid_size=2**6,
            util_loss_frequency=n_epochs,
            plot_frequency=500,
            cache_eval_actions=True,
            log_metrics = {
                'util_loss': True,
                'epsilon': True,
                },
            ) \
        .set_hardware(
            specific_gpu=specific_gpu
            ) \
        .get_config()
    experiment = experiment_class(experiment_config)
    experiment.run()
    torch.cuda.empty_cache()


    # Combinatoriral auction with item bidding ################################
    # # DEPRECATED
    # log_root_dir = os.path.join(
    #     os.path.expanduser('~'), 'bnelearn', 'experiments', 'asymmmetric', 'caib'
    #     )

    # # n_collections = [3, 2, 1]
    # n_items = [2]  # [3, 2, 1]
    # n_players = [2]  #[3, 2]

    # # TODO Nils: debug sampler for one_player_with_unit_demand
    # # for n_collection in n_collections:
    # for n_item in n_items:
    #     for n_player in n_players:
    #         experiment_config, experiment_class = \
    #             ConfigurationManager(
    #                 experiment_type='caib', n_runs=5,
    #                 n_epochs=n_epochs
    #                 ) \
    #             .set_setting(
    #                 n_players=n_player,
    #                 n_items=n_item,
    #                 exp_type='XOS',
    #                 exp_params={
    #                     'n_collections': n_collection,
    #                     'one_player_with_unit_demand': True
    #                     },
    #                 ) \
    #             .set_logging(
    #                 log_root_dir=log_root_dir,
    #                 util_loss_batch_size=2**11,
    #                 util_loss_grid_size=2**10,
    #                 util_loss_frequency=20,
    #                 best_response=True,
    #                 plot_frequency=100,
    #                 log_metrics = {
    #                     'opt': True,
    #                     'util_loss': True,
    #                     'epsilon': True,
    #                     },
    #                 ) \
    #             .set_hardware(
    #                 specific_gpu=specific_gpu
    #                 ) \
    #             .get_config()
    #         experiment = experiment_class(experiment_config)
    #         experiment.run()
    #         torch.cuda.empty_cache()
