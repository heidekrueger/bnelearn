"""Runs predefined experiments with opponent awareness."""
import os
import sys
import numpy as np

import torch
from torch.utils.tensorboard import SummaryWriter

sys.path.append(os.path.realpath('.'))
sys.path.append(os.path.join(os.path.expanduser('~'), 'bnelearn'))

from bnelearn.util import logging
from bnelearn.experiment.configuration_manager import ConfigurationManager


if __name__ == '__main__':
    """
    TODO
    * integrate `learners` switich for `.set_learning()` as seen below, s.t.
      user can provide string of which learner should be used.
      `Experiment._setup_learners()` should be the place for the actual selection.
    """
    learners = ['ESPGLearner', 'LOLALearner', 'SOSLearner']

    log_root_dir = os.path.join(os.path.expanduser('~'), 'bnelearn',
                                'experiments', 'opponent-awareness')

    for learner in learners:
        experiment_config, experiment_class = ConfigurationManager(
            experiment_type='single_item_gaussian_symmetric',
            n_runs=1,
            n_epochs=5,
        ) \
        .set_setting() \
        .set_learning(
            learner=learner,  # TODO
            model_sharing=True,
        ) \
        .set_hardware() \
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
