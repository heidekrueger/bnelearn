import os
import sys
import numpy as np

import torch
from torch.utils.tensorboard import SummaryWriter

# put bnelearn imports after this.
# pylint: disable=wrong-import-position
sys.path.append(os.path.realpath('.'))
sys.path.append(os.path.join(os.path.expanduser('~'), 'bnelearn'))

# from bnelearn.util import logging
from bnelearn.experiment.configuration_manager import ConfigurationManager  # pylint: disable=import-error

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
    # experiment_config, experiment_class = ConfigurationManager(experiment_type='single_item_uniform_symmetric', n_runs=1,
    #                                                            n_epochs=200) \
    #     .set_setting(risk=1.1)\
    #     .set_logging(log_root_dir=log_root_dir, save_tb_events_to_csv_detailed=True)\
    #     .set_learning(pretrain_iters=5) \
    #     .set_logging(eval_batch_size=2**4).get_config()

    # experiment_config, experiment_class = ConfigurationManager(experiment_type='single_item_gaussian_symmetric',n_runs=1, n_epochs=5) \
    #     .set_logging(log_root_dir=log_root_dir).get_config()

    # All three next experiments get AssertionError: scalar should be 0D
    # experiment_config, experiment_class = \
    #    ConfigurationManager(experiment_type='single_item_asymmetric_uniform_overlapping', n_runs=1, n_epochs=200) \
    #     .set_logging(log_root_dir=log_root_dir) \
    #     .get_config()
    # experiment_config, experiment_class = \
    #     ConfigurationManager(experiment_type='single_item_asymmetric_uniform_disjunct', n_runs=1, n_epochs=200) \
    #     .set_logging(log_root_dir=log_root_dir) \
    #     .get_config()
    experiment_config, experiment_class = ConfigurationManager(experiment_type='llg', n_runs=1, n_epochs=100) \
        .set_setting(gamma=0.5) \
        .set_logging(log_root_dir=log_root_dir, util_loss_batch_size=2 ** 7, util_loss_grid_size=2 ** 6,
                     util_loss_frequency=1).get_config()
    # experiment_config, experiment_class = ConfigurationManager(experiment_type='llllgg', n_runs=1, n_epochs=200) \
    #     .set_logging(log_root_dir=log_root_dir) \
    #     .get_config()
    # RuntimeError: Sizes of tensors must match
    # experiment_config, experiment_class = ConfigurationManager(experiment_type='multiunit',n_runs=1, n_epochs=200) \
    #     .set_logging(log_root_dir=log_root_dir) \
    #     .get_config()
    # experiment_config, experiment_class = ConfigurationManager(experiment_type='splitaward',n_runs=1, n_epochs=200)\
    #     .set_logging(log_root_dir=log_root_dir) \
    #     .get_config()
    # experiment_config, experiment_class = ConfigurationManager(experiment_type='multiunit', n_runs=1, n_epochs=2) \
    #     .set_logging(log_root_dir=log_root_dir, save_tb_events_to_csv_detailed=True)\
    #     .set_setting().set_learning().set_hardware() \
    #     .get_config()
    # experiment_config, experiment_class = \
    # ConfigurationManager(experiment_type='mineral_rights', n_runs=1, n_epochs=0)\
    #     .set_learning(pretrain_iters=3)\
    #     .set_logging(log_root_dir=log_root_dir)\
    #     .set_hardware(specific_gpu=7)\
    #     .get_config()
    # experiment_config, experiment_class = \
    #     ConfigurationManager(experiment_type='affiliated_observations', n_runs=1, n_epochs=1) \
    #     .set_learning(pretrain_iters=1) \
    #     .set_logging(log_root_dir=log_root_dir) \
    #     .set_hardware(specific_gpu=1) \
    #     .get_config()

    try:
        experiment = experiment_class(experiment_config)

        # Could only be done here and not inside Experiment itself while the checking depends on Experiment subclasses
        if ConfigurationManager.experiment_config_could_be_saved_properly(experiment_config):
            experiment.run()
        else:
            raise Exception('Unable to perform the correct serialization')
    except KeyboardInterrupt:
        print('\nKeyboardInterrupt: released memory after interruption')
        torch.cuda.empty_cache()

    # 10k epoch bug
    # log_dir = os.path.join('/home/gleb/Projects/bnelearn/experiments/test/subrun')
    # writer = SummaryWriter(log_dir)
    #
    # for n_iter in range(20000):
    #     writer.add_scalar('Loss/train', np.random.random(), n_iter)
    #     # writer.add_scalar('Loss/test', np.random.random(), n_iter)
    #
    # writer.close()
    #
    # logging.tabulate_tensorboard_logs('/home/gleb/Projects/bnelearn/experiments/', write_detailed=True)
