"""
Script for running experiments involving the smoothed game versions.
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

    # Experiments comparing NPGA and Smooth for single-item and LLG
    if True:
        log_root_dir = os.path.join(
            os.path.expanduser('~'), 'bnelearn', 'experiments', 'smooth', 'llg'
        )
 
        # User parameters
        specific_gpu = 2
        n_runs = 10
        n_epochs = 2000

        # model_sharing = True
        pretrain_iters = 50

        batch_size = 2**18
        eval_batch_size = 2**22
        util_loss_batch_size = 2**10
        util_loss_grid_size = 2**12
        util_loss_frequency = n_epochs

        for smoothing_temperature, learner in zip([0.02, None], ['PGLearner', 'ESPGLearner']):
        # for smoothing_temperature, learner in zip([0.01], ['PGLearner']):
        # for smoothing_temperature, learner in zip([None], ['ESPGLearner']):
            experiment_config, experiment_class = \
                ConfigurationManager(
                    # experiment_type='single_item_uniform_symmetric',
                    experiment_type='llg',
                    n_runs=n_runs,
                    n_epochs=n_epochs,
                ) \
                .set_setting(
                    # payment_rule='first_price',
                    # payment_rule='second_price',
                    # payment_rule='nearest_zero',
                    payment_rule='vcg',
                    # n_players=2
                    ) \
                .set_learning(
                    learner_type=learner,
                    smoothing_temperature=smoothing_temperature,
                    model_sharing=True,
                    batch_size=batch_size,
                    pretrain_iters=pretrain_iters,
                    ) \
                .set_logging(
                    log_root_dir=log_root_dir,
                    util_loss_frequency=util_loss_frequency,
                    util_loss_batch_size=util_loss_batch_size,
                    util_loss_grid_size=util_loss_grid_size,
                    save_tb_events_to_csv_detailed=True,
                    eval_batch_size=eval_batch_size,
                    best_response=True,
                    cache_eval_actions=True,
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
        # log_root_dir = os.path.join(log_root_dir, 'players')
        # log_root_dir = os.path.join(log_root_dir, 'items')
        log_root_dir = os.path.join(log_root_dir, 'batch_size')

        # User parameters
        specific_gpu = 1
        n_runs = 5
        n_epochs = 1000

        # model_sharing = True
        pretrain_iters = 10

        eval_batch_size = 2**22
        util_loss_batch_size = 2**3
        util_loss_grid_size = 2**3
        util_loss_frequency = n_epochs

        n_players_list = [2]  # [2, 3, 4, 5]
        n_items_list = [1]
        batch_size_list = [2**8, 2**12, 2**18, 2**23]
        smoothing_temperatures = np.linspace(0.001, 0.1, 10)

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
                                util_loss_frequency=util_loss_frequency,
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
