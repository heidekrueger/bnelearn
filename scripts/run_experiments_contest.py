"""
Script for defining and starting a contest experiment.
"""
import os
import sys

import torch

# put bnelearn imports after this.
# pylint: disable=wrong-import-position
sys.path.append(os.path.realpath('.'))

from bnelearn.experiment.configuration_manager import ConfigurationManager  

if __name__ == '__main__':

    # User parameters
    n_epochs = 3500
    n_runs = 1
    model_sharing = True
    pretrain_iters = 500
    batch_size = 2**18

    eval_batch_size = 2**18
    util_loss_frequency = 1000
    util_loss_batch_size = 2**6
    util_loss_grid_size = 2**6

    specific_gpu = 5
    log_root_dir = os.path.join(
        os.path.expanduser('~'), 'bnelearn', 'experiments', 'public-test',
    )

    # Contest Configurations
    contests = {
        'tullock': {
            'type': 'tullock_contest',
            'settings': {
                'impact_factor': 0.5,
            },
            'learning': {
                'use_valuation': True
            }
        },
        'all_pay': {
            'type': 'all_pay_uniform_symmetric',
            'settings': {
                'payment_rule': 'all_pay'
            },
            'learning': {}
        },
        'crowdsourcing': {
            'type': 'crowdsourcing',
            'settings': {
                'n_players': 3,
                'valuations': [0.8, 0.2, 0.0]
            },
            'learning': {
                'use_valuation': True
            }
        }
    }

    
    for _, contest in contests.items():
        
        experiment_config, experiment_class = \
            ConfigurationManager(
                experiment_type=contest['type'],
                n_runs=n_runs,
                n_epochs=n_epochs,
            ) \
            .set_setting(
                **contest['settings'],
            ) \
            .set_learning(
                batch_size=batch_size,
                **contest['learning']
            ) \
            .set_logging(
                log_root_dir=log_root_dir,
                util_loss_frequency=util_loss_frequency,
                util_loss_batch_size=util_loss_batch_size,
                util_loss_grid_size=util_loss_grid_size,
                eval_batch_size=eval_batch_size,
                log_metrics={
                    'opt': True,
                    'util_loss': True,
                },
            ) \
            .set_hardware(
                specific_gpu=specific_gpu,
            ) \
            .get_config()
        experiment = experiment_class(experiment_config)
        experiment.run()

        torch.cuda.empty_cache()
