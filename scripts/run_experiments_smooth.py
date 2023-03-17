"""
Script for running experiments involving the smoothed game versions in Kohring et al. (2023).
"""
import os
import sys
import traceback
import numpy as np
import torch
from torch import nn

sys.path.append(os.path.realpath('.'))
sys.path.append(os.path.join(os.path.expanduser('~'), 'bnelearn'))

# pylint: disable=wrong-import-position
from bnelearn.experiment.configuration_manager import ConfigurationManager


if __name__ == '__main__':

    # Experiments comparing NPGA and Smooth for single-item
    if True:
        log_root_dir = os.path.join(
            os.path.expanduser('~'), 'bnelearn', 'experiments', 'smooth'
        )

        payment_rules = ["first_price", "second_price"]
        # experiment_types = ["mineral_rights", "affiliated_observations"]
        n_items_list = [1, 2, 4]

        # User parameters
        specific_gpu = 1
        n_runs = 10
        n_epochs = 2000
        pretrain_iters = 50

        batch_size = 2**18
        eval_batch_size = 2**22
        util_loss_own_batch_size = 2**10
        util_loss_opponent_batch_size = 2**20
        util_loss_grid_size = 2**10
        eval_frequency = n_epochs

        learners = dict(
            ReinforceLearner=dict(
                smoothing_temperature=None,
                mixed_strategy="normal",
                ),
            PGLearner=dict(
                smoothing_temperature=0.01,
                mixed_strategy=None
                ),
            ESPGLearner=dict(
                smoothing_temperature=None,
                mixed_strategy=None,
                )
        )
        for learner, learner_params in learners.items():
            for payment_rule in payment_rules:
                for n_items in n_items_list:
                    experiment_config, experiment_class = \
                        ConfigurationManager(
                            experiment_type="single_item_uniform_symmetric",
                            n_runs=n_runs,
                            n_epochs=n_epochs,
                            ) \
                        .set_setting(
                            payment_rule=payment_rule,
                            n_items=n_items,
                            # n_players=2
                            ) \
                        .set_learning(
                            learner_type=learner,
                            smoothing_temperature=learner_params["smoothing_temperature"],
                            model_sharing=True,
                            batch_size=batch_size,
                            pretrain_iters=pretrain_iters,
                            mixed_strategy=learner_params["mixed_strategy"]
                            ) \
                        .set_logging(
                            log_root_dir=log_root_dir,
                            eval_frequency=eval_frequency,
                            util_loss_batch_size=util_loss_own_batch_size,
                            util_loss_opponent_batch_size=util_loss_opponent_batch_size,
                            util_loss_grid_size=util_loss_grid_size,
                            save_tb_events_to_csv_detailed=True,
                            eval_batch_size=eval_batch_size,
                            best_response=True,
                            cache_eval_actions=True,
                            log_metrics={
                                'opt': True,
                                'util_loss': True,
                                'gradient_variance': False
                                }
                            ) \
                        .set_hardware(specific_gpu=specific_gpu) \
                        .get_config()
                    experiment = experiment_class(experiment_config)
                    experiment.run()
                    torch.cuda.empty_cache()

    # Experiment analyzing temperature
    if False:
        log_root_dir = os.path.join(
            os.path.expanduser('~'), 'bnelearn', 'experiments', 'smooth', 'temperature'
        )
        log_root_dir = os.path.join(log_root_dir, 'players')
        # log_root_dir = os.path.join(log_root_dir, 'items')
        # log_root_dir = os.path.join(log_root_dir, 'batch_size')

        # User parameters
        specific_gpu = 1
        n_runs = 5
        n_epochs = 2000
        pretrain_iters = 50

        eval_batch_size = 2**22
        util_loss_batch_size = 2**3
        util_loss_grid_size = 2**3
        eval_frequency = n_epochs

        n_players_list = [2, 3, 4, 5]
        n_items_list = [1]
        batch_size_list = [2**18]  # [2**10, 2**14, 2**18, 2**22]
        smoothing_temperatures = np.linspace(0.00005, 0.08, 10)

        for n_players in n_players_list:
            for n_items in n_items_list:
                for batch_size in batch_size_list:
                    for smoothing_temperature in smoothing_temperatures:
                        experiment_config, experiment_class = \
                            ConfigurationManager(
                                experiment_type='single_item_uniform_symmetric',
                                n_runs=n_runs,
                                n_epochs=n_epochs,
                            ) \
                            .set_setting(
                                payment_rule='first_price',
                                n_players=n_players,
                                n_items=n_items,
                                ) \
                            .set_learning(
                                learner_type='PGLearner',
                                smoothing_temperature=smoothing_temperature,
                                model_sharing=True,
                                batch_size=batch_size,
                                pretrain_iters=pretrain_iters,
                                ) \
                            .set_logging(
                                log_root_dir=log_root_dir,
                                eval_frequency=eval_frequency,
                                util_loss_batch_size=util_loss_batch_size,
                                util_loss_grid_size=util_loss_grid_size,
                                eval_batch_size=eval_batch_size,
                                best_response=True,
                                cache_eval_actions=True,
                                ) \
                            .set_hardware(specific_gpu=specific_gpu) \
                            .get_config()
                        experiment = experiment_class(experiment_config)
                        experiment.run()
                        torch.cuda.empty_cache()

    # Experiments comparing NPGA and Smooth for single-item
    if False:
        log_root_dir = os.path.join(
            os.path.expanduser('~'), 'bnelearn', 'experiments', 'smooth-new', 'variance'
        )

        # User parameters
        specific_gpu = 1
        n_runs = 5
        n_epochs = 500
        pretrain_iters = 50

        batch_size = 2**16
        eval_batch_size = 2**22
        util_loss_batch_size = 2**7
        util_loss_grid_size = 2**10
        eval_frequency = n_epochs
        log_gradient_variance = True

        learners = dict(
            ReinforceLearner=dict(
                learner_type="ReinforceLearner",
                smoothing_temperature=None,
                mixed_strategy="normal",
                learner_hyperparams=dict()
                ),
            PGLearner_01=dict(
                learner_type="PGLearner",
                smoothing_temperature=0.01,
                mixed_strategy=None,
                learner_hyperparams=dict()
                ),
            PGLearner_005=dict(
                learner_type="PGLearner",
                smoothing_temperature=0.005,
                mixed_strategy=None,
                learner_hyperparams=dict()
                ),
            ESPGLearner_64=dict(
                learner_type="ESPGLearner",
                smoothing_temperature=None,
                mixed_strategy=None,
                learner_hyperparams={
                    'population_size': 64, 'sigma': 1.,
                    'scale_sigma_by_model_size': True
                    },
                ),
            ESPGLearner_128=dict(
                learner_type="ESPGLearner",
                smoothing_temperature=None,
                mixed_strategy=None,
                learner_hyperparams={
                    'population_size': 128, 'sigma': 1.,
                    'scale_sigma_by_model_size': True
                    },
                )
        )
        for _, learner_params in learners.items():
            experiment_config, experiment_class = \
                ConfigurationManager(
                    experiment_type="single_item_uniform_symmetric",
                    n_runs=n_runs,
                    n_epochs=n_epochs,
                    ) \
                .set_setting(
                    payment_rule="first_price",
                    n_items=1,
                    ) \
                .set_learning(
                    learner_type=learner_params["learner_type"],
                    learner_hyperparams=learner_params["learner_hyperparams"],
                    smoothing_temperature=learner_params["smoothing_temperature"],
                    model_sharing=True,
                    batch_size=batch_size,
                    pretrain_iters=pretrain_iters,
                    mixed_strategy=learner_params["mixed_strategy"]
                    ) \
                .set_logging(
                    log_root_dir=log_root_dir,
                    eval_frequency=eval_frequency,
                    util_loss_batch_size=util_loss_batch_size,
                    util_loss_grid_size=util_loss_grid_size,
                    save_tb_events_to_csv_detailed=True,
                    eval_batch_size=eval_batch_size,
                    best_response=True,
                    cache_eval_actions=True,
                    log_metrics={
                        'opt': True,
                        'util_loss': True,
                        'gradient_variance': log_gradient_variance
                        }
                    ) \
                .set_hardware(specific_gpu=specific_gpu) \
                .get_config()
            experiment = experiment_class(experiment_config)
            experiment.run()
            torch.cuda.empty_cache()
