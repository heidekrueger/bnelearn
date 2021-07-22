"""Testing correctness of the BNE utilities database."""

import torch
import pandas as pd
import pkg_resources

from bnelearn.experiment.configuration_manager import ConfigurationManager
import bnelearn.util.logging as logging_utils

cuda = torch.cuda.is_available()


def test_bne_utility_database():
    """Testing correctness of the BNE utilities database."""

    BATCH_SIZE = 2**18

    # Create fake experiment
    config, exp_class = ConfigurationManager(experiment_type='llg', n_runs=0, n_epochs=0) \
        .set_setting(
            payment_rule='vcg',
            gamma=.9876,
            risk=.01234
        ) \
        .set_logging(eval_batch_size=BATCH_SIZE) \
        .set_hardware(cuda=cuda) \
        .get_config()
    exp = exp_class(config)
    n_agents = len(exp.bne_env.agents)

    # Check retrival
    b_batch_size, db_bne_utility = logging_utils.read_bne_utility_database(exp)
    assert b_batch_size == BATCH_SIZE, 'saved wrong batch size'
    assert all([abs(a-b) < 1e-16 for a, b in zip(db_bne_utility, exp.bne_utilities)]), \
        'saved wrong utilites'

    # Clean-up: Delete test entry
    file_path = pkg_resources.resource_filename(__name__[:__name__.find('.')],
                                                'util/bne_database.csv')
    bne_database = pd.read_csv(file_path)
    bne_database.drop(bne_database.tail(n_agents).index, inplace=True)
    bne_database.to_csv(file_path, index=False)
