"""
Script for running experiments involving the smoothed game versions.
"""
import os
import sys
import traceback
import torch
from torch import nn

sys.path.append(os.path.realpath('.'))
sys.path.append(os.path.join(os.path.expanduser('~'), 'bnelearn'))

# pylint: disable=wrong-import-position
from bnelearn.experiment.configuration_manager import ConfigurationManager


if __name__ == '__main__':

    torch.set_printoptions(precision=4, sci_mode=False)

    # NOTE: n_players changes optimal smoothing factor.
    # User parameters
    specific_gpu = 1
    n_runs = 1
    n_epochs = 500

    # model_sharing = True
    pretrain_iters = 10

    batch_size = 2**16
    eval_batch_size = 2**22
    util_loss_frequency = n_epochs
    util_loss_batch_size = 2**8
    util_loss_grid_size = 2**9

    log_root_dir = os.path.join(
        os.path.expanduser('~'), 'bnelearn', 'experiments', 'debug-smooth'
    )

    for smooth_market, learner in zip([True, False], ['PGLearner', 'ESPGLearner']):
    # for smooth_market, learner in zip([True], ['PGLearner']):
    # for smooth_market, learner in zip([False], ['ESPGLearner']):
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
                smooth_market=smooth_market,
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
