import os
import sys

import fire
import torch

import torch.nn as nn

sys.path.append(os.path.realpath('.'))
sys.path.append(os.path.join(os.path.expanduser('~'), 'bnelearn'))

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

    #TODO, Paul: 1. specific gpu doesn't work. 2. separate qpth and mpc.
    # Well, path is user-specific
    log_root_dir = os.path.join(os.path.expanduser('~'), 'bnelearn', 'experiments')

    # Experiment
    experiment_type = 'llllgg'

    # Setting
    payment_rule = 'nearest_vcg'
    risk = 1.0
    n_players = 6

    # Learning
    batch_size = 2 ** 13

    # Hardware
    specific_gpu=4

    # Logging
    util_loss_batch_size = 2 ** 7
    util_loss_grid_size = 2 ** 8
    log_metrics = {'opt': False,#True
                 'l2': False,#True
                 'util_loss': True}

    #.with_correlation(gamma=0.0) \
    experiment_config, experiment_class = ConfigurationManager(experiment_type=experiment_type) \
        .get_config(log_root_dir=log_root_dir,
                    # Run
                    n_runs = 1,
                    n_epochs = 10,
                    #Setting
                    payment_rule=payment_rule,
                    risk=risk,
                    n_players=n_players,
                    core_solver="mpc",
                    # Learning
                    hidden_nodes = [10,10],
                    hidden_activations = [nn.SELU(),nn.SELU()],
                    learner_hyperparams = {'population_size': 2,
                                           'sigma': 1.,
                                           'scale_sigma_by_model_size': True},
                    optimizer_hyperparams = {'lr': 1e-3},
                    optimizer_type='adam',
                    pretrain_iters=500,
                    batch_size=batch_size,
                    model_sharing=True,
                    # Hardware
                    specific_gpu=specific_gpu,
                    # Logging
                    util_loss_batch_size=util_loss_batch_size,
                    util_loss_grid_size=util_loss_grid_size,
                    util_loss_frequency=100,
                    log_metrics=log_metrics,
                    eval_batch_size=2 ** 22,
                    enable_logging=True,
                    save_tb_events_to_csv_detailed=True,
                    save_tb_events_to_binary_detailed=True)

    try:
        experiment = experiment_class(experiment_config)

        # Could only be done here and not inside Experiment itself while the checking depends on Experiment subclasses
        if ConfigurationManager.experiment_config_could_be_serialized_properly(experiment_config):
            experiment.run()
        else:
            raise Exception('Unable to perform the correct serialization')
    except KeyboardInterrupt:
        print('\nKeyboardInterrupt: released memory after interruption')
        torch.cuda.empty_cache()
