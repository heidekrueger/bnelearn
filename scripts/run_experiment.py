import os
import sys

import fire
import torch

sys.path.append(os.path.realpath('.'))
sys.path.append(os.path.join(os.path.expanduser('~'), 'bnelearn'))

from bnelearn.experiment.configuration_manager import ConfigurationManager
from bnelearn.util import logging



if __name__ == '__main__':
    '''
    Runs predefined experiments with individual parameters
    fire.Fire() asks you to decide for one of the experiments defined above
    by writing its name and define the required (and optional) parameters
    e.g.:
        experiment.py single_item_uniform_symmetric 1 20 [2,3] 'first_price'

    alternatively instead of fire.Fire() use, e.g.:
        single_item_uniform_symmetric(1,20,[2,3],'first_price')

    '''
    # running_configuration, logging_configuration, experiment_configuration, experiment_class = \
    #     fire.Fire()

    # Run from a file
    # experiment_config = logging.get_experiment_config_from_configurations_log()
    # experiment_class = ConfigurationManager.get_class_by_experiment_type(experiment_config.experiment_class)

    # Well, path is user-specific
    log_root_dir = os.path.join(os.path.expanduser('~'), 'bnelearn', 'experiments')

    experiment_config, experiment_class = ConfigurationManager(experiment_type='single_item_uniform_symmetric') \
        .get_config(save_tb_events_to_csv_detailed=True, log_root_dir=log_root_dir, n_runs=1, n_epochs=500)
    # experiment_config, experiment_class = ConfigurationManager(experiment_type='single_item_gaussian_symmetric') \
    #     .get_config(log_root_dir=log_root_dir)

    # All three next experiments get AssertionError: scalar should be 0D
    # experiment_config, experiment_class = \
    #     ConfigurationManager(experiment_type='single_item_asymmetric_uniform_overlapping') \
    #     .get_config(log_root_dir=log_root_dir)
    # experiment_config, experiment_class = \
    #     ConfigurationManager(experiment_type='single_item_asymmetric_uniform_disjunct') \
    #     .get_config(log_root_dir=log_root_dir)
    # experiment_config, experiment_class = ConfigurationManager(experiment_type='llg') \
    #     .get_config(log_root_dir=log_root_dir)

    # experiment_config, experiment_class = ConfigurationManager(experiment_type='llllgg') \
    #     .get_config(log_root_dir=log_root_dir)
    
    # RuntimeError: Sizes of tensors must match
    # experiment_config, experiment_class = ConfigurationManager(experiment_type='multiunit') \
    #     .get_config(log_root_dir=log_root_dir)
    # experiment_config, experiment_class = ConfigurationManager(experiment_type='splitaward')\
    #     .get_config(log_root_dir=log_root_dir)


    try:
        experiment_class(experiment_config).run()
    except KeyboardInterrupt:
        print('\nKeyboardInterrupt: released memory after interruption')
        torch.cuda.empty_cache()
