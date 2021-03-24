"""Runs predefined experiments with opponent awareness."""
import os
import sys
import numpy as np

import torch
from torch.utils.tensorboard import SummaryWriter

# put bnelearn imports after this.
# pylint: disable=wrong-import-position
sys.path.append(os.path.realpath('.'))
sys.path.append(os.path.join(os.path.expanduser('~'), 'bnelearn'))

from bnelearn.util import logging
from bnelearn.experiment.configuration_manager import ConfigurationManager  # pylint: disable=import-error


if __name__ == '__main__':
    r"""
    TODO
    * integrate `learners` switich for `.set_learning()` as seen below, s.t.
      user can provide string of which learner should be used.
      `Experiment._setup_learners()` should be the place for the actual selection.

    """
    # ['ESPGLearner', 'LOLALearner', 'SOSLearner', 'LOLA_ESPGLearner', 'SOS_ESPGLearner']
    learners = ['ESPGLearner']
    log_root_dir = os.path.join(os.path.expanduser('~'), 'bnelearn',
                                'experiments', 'opponent-awareness')

    # Params
    n_epochs = 200
    batch_size = 2**10
    util_loss_batch_size = 2

    for learner in learners:
        experiment_config, experiment_class = ConfigurationManager(
            experiment_type='cycle_game',
            n_runs=1,
            n_epochs=n_epochs,
        ) \
        .set_setting(
            continuous=True,
        ) \
        .set_learning(
            batch_size=batch_size,
            pretrain_iters=0,
            learner_type=learner,
            learner_hyperparams={
                'population_size': 32,
                'sigma': 1.,
                'scale_sigma_by_model_size': True},
            # hidden_nodes=[],
            # hidden_activations=[],
            non_negative_actions=False,
            use_bias=False,
            optimizer_type='SGD',
            model_sharing=False,
        ) \
        .set_hardware(
            specific_gpu=6,
        ) \
        .set_logging(
            log_root_dir=log_root_dir,
            util_loss_batch_size=util_loss_batch_size
        ) \
        .get_config()
        experiment = experiment_class(experiment_config)

        try:
            experiment.run()
            print("---")
            print(f"{learner} reached a L2 of [{experiment._cur_epoch_log_params['L_2']}.")
            print("---")

        except KeyboardInterrupt:
            print('\nKeyboardInterrupt: released memory after interruption')
            torch.cuda.empty_cache()
