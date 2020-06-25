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
    Runs predefined experiments with individual parameters for the Journal version
    '''

    # Well, path is user-specific
    log_root_dir = os.path.join(os.path.expanduser('~'), 'bnelearn', 'experiments')

    # Experiments [single_item_uniform_symmetric, single_item_gaussian_symmetric, single_item_asymmetric_uniform_overlapping, 
    # single_item_asymmetric_uniform_disjunct, llg, llllgg, multiunit, splitaward]
    experiment_type = 'llg'

    # Setting
    # Payments [first_price, nearest_vcg, nearest_zero, nearest_bid]
    payment_rule = 'first_price'
    risk = 1.0
    n_players = [3]#,3,5,10]

    # Learning
    batch_size = 2 ** 18

    # Hardware
    specific_gpu=4

    # Logging
    util_loss_batch_size = 2 ** 12
    util_loss_grid_size = 2 ** 13
    log_metrics = {'opt': True,#True
                 'l2': True,#True
                 'util_loss': True}
    for n_player in n_players:
        #.with_correlation(gamma=0.0) \
        experiment_config, experiment_class = ConfigurationManager(experiment_type=experiment_type) \
            .get_config(log_root_dir=log_root_dir,
                        # Run
                        n_runs = 10,
                        n_epochs = 5000,
                        #Setting
                        payment_rule=payment_rule,
                        risk=risk,
                        n_players=n_player,
                        core_solver="mpc",
                        # Learning
                        hidden_nodes = [10,10],
                        hidden_activations = [nn.SELU(),nn.SELU()],
                        learner_hyperparams = {'population_size': 64,
                                            'sigma': 1.,
                                            'scale_sigma_by_model_size': True},
                        optimizer_hyperparams = {'lr': 1e-3},
                        optimizer_type='adam',
                        pretrain_iters=500,
                        batch_size=batch_size,
                        #model_sharing=True,
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
