"""Testing correctness of the BNE utilities database."""

import pytest
import os
import sys



from bnelearn.experiment.configuration_manager import ConfigurationManager
from bnelearn.util.logging import access_bne_utility_database


def test_bne_utility_database():
    """Testing correctness of the BNE utilities database."""
    specific_gpu = 0

    experiment_config, experiment_class = ConfigurationManager(experiment_type='llg', n_runs=1, n_epochs=1)\
        .set_hardware(specific_gpu=specific_gpu)\
        .set_learning(pretrain_iters=0, batch_size=2).set_logging(enable_logging=False) \
        .get_config()

    experiment = experiment_class(experiment_config)
    experiment.logging.eval_batch_size = 2
    experiment.logging.util_loss_batch_size = 2
    experiment.logging.util_loss_grid_size = 2
    experiment._setup_eval_environment() #pylint: disable=protected-access
    access_bne_utility_database(experiment, experiment.bne_utilities_new_sample)
