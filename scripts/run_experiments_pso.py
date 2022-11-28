"""Script for reproducing the reported experiments in Kohring et. al (2022).
"""
import os
import sys
from itertools import product
import torch

sys.path.append(os.path.realpath('.'))

from bnelearn.experiment.configuration_manager import ConfigurationManager


if __name__ == '__main__':

    # User parameters
    n_epochs = 1000
    n_runs = 1
    model_sharing = False
    pretrain_iters = 500
    batch_size = 2**17

    eval_batch_size = 2**17
    util_loss_frequency = 1000
    util_loss_batch_size = 2**12
    util_loss_grid_size = 2**10

    specific_gpu = 0
    log_root_dir = os.path.join(
        os.path.expanduser('~'), 'bnelearn', 'experiments', 'public-test',
    )

    population_size = 64
    learners = [
        # {
        #     'learner_type': 'ESPGLearner',
        #     'pretrain_iters': 500,
        #     'learner_hyperparams': {
        #         'population_size': population_size,
        #         'sigma': 1.,
        #         'scale_sigma_by_model_size': True
        #     },
        # },
        {
            'learner_type': 'PSOLearner',
            'pretrain_iters': 0,
            'learner_hyperparams': {
                'swarm_size': population_size,
                'topology': 'von_neumann',
                'reevaluation_frequency': 10,
                'inertia_weight': .5,
                'cognition': .8,
                'social': .8,
            }
        }
    ]

    # Compare NPGA and PSO
    for learner in learners:
        experiment_config, experiment_class = \
            ConfigurationManager(
                experiment_type='single_item_uniform_symmetric',
                n_runs=n_runs,
                n_epochs=n_epochs,
            ) \
            .set_setting(
                payment_rule='first_price',
            ) \
            .set_learning(
                batch_size=batch_size,
                learner_type=learner['learner_type'],
                pretrain_iters=learner['pretrain_iters'],
                learner_hyperparams=learner['learner_hyperparams'],
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
