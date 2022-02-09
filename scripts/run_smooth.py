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

    # User parameters
    specific_gpu = 1
    n_runs = 1
    n_epochs = 1000

    # model_sharing = True
    pretrain_iters = 50

    batch_size = 2**17
    eval_batch_size = 2**22
    util_loss_frequency = 50
    util_loss_batch_size = 2**10
    util_loss_grid_size = 2**10

    # Run LLG nearest-vcg for different risks / correlations ################
    log_root_dir = os.path.join(
        os.path.expanduser('~'), 'bnelearn', 'experiments', 'debug-smooth'
    )

    experiment_config, experiment_class = \
        ConfigurationManager(
            experiment_type='single_item_uniform_symmetric',
            n_runs=n_runs,
            n_epochs=n_epochs) \
            .set_setting(
                payment_rule='first_price',
                ) \
            .set_learning(
                learner_type='PGLearner',
                smooth_market=True,
                model_sharing=True,
                batch_size=batch_size,
                pretrain_iters=pretrain_iters,
                ) \
            .set_logging(
                log_root_dir=log_root_dir,
                util_loss_frequency=util_loss_frequency,
                util_loss_batch_size=util_loss_batch_size,
                util_loss_grid_size=util_loss_grid_size,
                eval_batch_size=eval_batch_size) \
            .set_hardware(specific_gpu=specific_gpu) \
            .get_config()
    experiment = experiment_class(experiment_config)
    experiment.run()
    torch.cuda.empty_cache()
