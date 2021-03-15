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
    learners = ['ESPGLearner']  # ['ESPGLearner', 'LOLALearner', 'SOSLearner']
    log_root_dir = os.path.join(os.path.expanduser('~'), 'bnelearn',
                                'experiments', 'opponent-awareness')

    for learner in learners:
        experiment_config, experiment_class = ConfigurationManager(
            experiment_type='cycle_game',
            n_runs=1,
            n_epochs=100,
        ) \
        .set_setting(
        ) \
        .set_learning(
            learner_type=learner,
            # hidden_nodes=[],
            # hidden_activations=[],
            non_negative_actions=False,
            use_bias=False,
            optimizer_type='Adam',
            model_sharing=False,
        ) \
        .set_hardware(
            specific_gpu=6,
        ) \
        .set_logging(
            log_root_dir=log_root_dir,
        ) \
        .get_config()
        experiment = experiment_class(experiment_config)

        try:
            experiment.run()
        except KeyboardInterrupt:
            print('\nKeyboardInterrupt: released memory after interruption')
            torch.cuda.empty_cache()
