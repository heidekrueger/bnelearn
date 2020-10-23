"""Testing correctness of the BNE utilities database."""

import pytest
import os
import sys



from bnelearn.experiment.configuration_manager import ConfigurationManager
from bnelearn.util.logging import access_bne_utility_database


def test_bne_utility_database():
    """Testing correctness of the BNE utilities database."""
    specific_gpu = 0

    experiment_config, experiment_class = ConfigurationManager(
        experiment_type='llg').get_config(
            specific_gpu=specific_gpu,
            pretrain_iters=0,
            batch_size=2,
            enable_logging=False,
            eval_batch_size=2,
            util_loss_batch_size=2,
            util_loss_grid_size=2
        )

    _ = experiment_class(experiment_config)
