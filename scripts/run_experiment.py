import os
import sys
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from bnelearn.util import logging

sys.path.append(os.path.realpath('.'))
sys.path.append(os.path.join(os.path.expanduser('~'), 'bnelearn'))

# pylint: disable=wrong-import-position
from bnelearn.experiment.configuration_manager import ConfigurationManager

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
    log_root_dir = os.path.join(os.path.expanduser('~'), 'Projects/bnelearn', 'experiments')
    # experiment_config, experiment_class = ConfigurationManager(experiment_type='single_item_uniform_symmetric', n_runs=1,
    #                                                            n_epochs=200) \
    #     .set_setting(risk=1.1)\
    #     .set_logging(log_root_dir=log_root_dir, save_tb_events_to_csv_detailed=True)\
    #     .set_learning(pretrain_iters=5) \
    #     .set_running(n_runs=1, n_epochs=5).set_logging(eval_batch_size=2**4).get_config()

    # experiment_config, experiment_class = ConfigurationManager(experiment_type='single_item_gaussian_symmetric') \
    #     .set_logging(log_root_dir=log_root_dir).set_running(n_runs=1, n_epochs=5).get_config()

    # All three next experiments get AssertionError: scalar should be 0D
    # experiment_config, experiment_class = \
    #    ConfigurationManager(experiment_type='single_item_asymmetric_uniform_overlapping') \
    #     .set_logging(log_root_dir=log_root_dir) \
    #     .set_running(n_runs=1, n_epochs=200).get_config()
    # experiment_config, experiment_class = \
    #     ConfigurationManager(experiment_type='single_item_asymmetric_uniform_disjunct') \
    #     .set_logging(log_root_dir=log_root_dir) \
    #     .set_running(n_runs=1, n_epochs=200).get_config()
    # experiment_config, experiment_class = ConfigurationManager(experiment_type='llg')\
    #    .set_running(n_runs=1, n_epochs=100).set_logging(log_root_dir=log_root_dir).get_config()

    # experiment_config, experiment_class = ConfigurationManager(experiment_type='llllgg') \
    #     .set_logging(log_root_dir=log_root_dir) \
    #     .set_running(n_runs=1, n_epochs=200).get_config()
    # RuntimeError: Sizes of tensors must match
    # experiment_config, experiment_class = ConfigurationManager(experiment_type='multiunit') \
    #     .set_logging(log_root_dir=log_root_dir) \
    #     .set_running(n_runs=1, n_epochs=200).get_config()
    # experiment_config, experiment_class = ConfigurationManager(experiment_type='splitaward')\
    #     .set_logging(log_root_dir=log_root_dir) \
    #     .set_running(n_runs=1, n_epochs=200).get_config()
    # experiment_config, experiment_class = ConfigurationManager(experiment_type='multiunit', n_runs=1, n_epochs=2) \
    #     .set_running().set_logging(log_root_dir=log_root_dir, save_tb_events_to_csv_detailed=True)\
    #     .set_setting().set_learning().set_hardware() \
    #     .get_config()
    # experiment_config, experiment_class = \
    # ConfigurationManager(experiment_type='mineral_rights', n_runs=1, n_epochs=0)\
    #     .set_learning(pretrain_iters=3)\
    #     .set_logging(log_root_dir=log_root_dir)\
    #     .set_hardware(specific_gpu=7)\
    #     .get_config()
    pass
    experiment_config, experiment_class = \
        ConfigurationManager(experiment_type='affiliated_observations', n_runs=1, n_epochs=1) \
        .set_learning(pretrain_iters=1) \
        .set_logging(log_root_dir=log_root_dir) \
        .set_hardware(specific_gpu=1) \
        .get_config()

    try:
        experiment = experiment_class(experiment_config)

        experiment.logging.util_loss_batch_size = 2 ** 7
        experiment.logging.util_loss_grid_size = 2 ** 6
        experiment.logging.util_loss_frequency = 1

        # TODO: this is a short term fix - we can only determine whether BNE exists once experiment has been
        #  initialized. Medium Term -->  Set 'opt logging in experiment itself.
        if experiment.known_bne:
            experiment.logging.log_metrics = {
                'opt': True,
                'l2': True,
                'util_loss': True
            }

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



