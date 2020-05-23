import os
import sys
from bnelearn.experiment.configuration_manager import ConfigurationManager
import fire
import torch

from bnelearn.util import logging

sys.path.append(os.path.realpath('.'))
sys.path.append(os.path.join(os.path.expanduser('~'), 'bnelearn'))

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

    experiment_config, experiment_class = ConfigurationManager(experiment_type='single_item_uniform_symmetric') \
        .get_config(cuda=False, save_tb_events_to_csv_detailed=True)
    # experiment_config, experiment_class = ConfigurationManager(experiment_type='single_item_gaussian_symmetric') \
    #     .get_config()
    # experiment_config, experiment_class = \
    #     ConfigurationManager(experiment_type='single_item_asymmetric_uniform_overlapping') \
    #     .get_config()
    # experiment_config, experiment_class = \
    #     ConfigurationManager(experiment_type='single_item_asymmetric_uniform_disjunct') \
    #     .get_config()
    # experiment_config, experiment_class = ConfigurationManager(experiment_type='llg') \
    #     .get_config()
    # experiment_config, experiment_class = ConfigurationManager(experiment_type='llllgg') \
    #     .get_config()
    # experiment_config, experiment_class = ConfigurationManager(experiment_type='multiunit') \
    #     .get_config()
    # experiment_config, experiment_class = ConfigurationManager(experiment_type='splitaward')\
    #     .get_config()


    # try:
    #     experiment_class(experiment_config).run()
    # except KeyboardInterrupt:
    #     print('\nKeyboardInterrupt: released memory after interruption')
    #     torch.cuda.empty_cache()

    experiment_dir = '/home/gleb/bnelearn/experiments/single_item/first_price/uniform/symmetric/risk_neutral/2p/2020-05-23 Sat 01.55'
    logging.run_experiment_from_configurations_log(experiment_dir)

