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
    log_root_dir = os.path.join(os.path.expanduser('~'), 'bnelearn',
                                'experiments', 'opponent-awareness-specific-v1')


    # Params ------------------------------------------------------------------
    # ['ESPGLearner', 'LOLALearner', 'SOSLearner', 'LOLA_ESPGLearner', 'SOS_ESPGLearner']
    learners = ['ESPGLearner', 'LOLALearner', 'SOSLearner', 'LOLA_ESPGLearner', 'SOS_ESPGLearner']
    n_runs = 3
    game_versions = [1]
    radii = [1.]
    n_epochs = 1000
    batch_size = 2**12
    eval_batch_size = 2**17
    specific_gpu = 6
    # -------------------------------------------------------------------------

    for learner in learners:
        for radius in radii:
            for game_version in game_versions:
                log_dir = os.path.join(log_root_dir, learner)
                experiment_config, experiment_class = ConfigurationManager(
                    experiment_type='cycle_game',
                    n_runs=n_runs,
                    n_epochs=n_epochs,
                ) \
                .set_setting(
                    continuous=False,
                    radius=radius,
                    game_version=game_version,
                ) \
                .set_learning(
                    batch_size=batch_size,
                    pretrain_iters=0,
                    learner_type=learner,
                    learner_hyperparams={
                        'population_size': 32,
                        'sigma': 1.,
                        'scale_sigma_by_model_size': True},
                    non_negative_actions=False,
                    use_bias=True,
                    optimizer_type='SGD',
                    model_sharing=False,
                ) \
                .set_hardware(
                    specific_gpu=specific_gpu,
                ) \
                .set_logging(
                    log_root_dir=log_dir,
                    util_loss_batch_size=2,
                    util_loss_grid_size=2,
                    util_loss_frequency=1e6,
                    eval_batch_size=eval_batch_size,
                    save_tb_events_to_csv_detailed=True,
                ) \
                .get_config()
                experiment = experiment_class(experiment_config)

                try:
                    experiment.run()

                except KeyboardInterrupt:
                    print('\nKeyboardInterrupt: released memory after interruption')

                finally:
                    torch.cuda.empty_cache()
