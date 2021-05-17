"""
Runs predefined experiments with individual parameters
fire.Fire() asks you to decide for one of the experiments defined above
by writing its name and define the required (and optional) parameters
e.g.:
    experiment.py single_item_uniform_symmetric 1 20 [2,3] 'first_price'

alternatively instead of fire.Fire() use, e.g.:
    single_item_uniform_symmetric(1,20,[2,3],'first_price')

"""
import os
import sys

import torch

# put bnelearn imports after this.
# pylint: disable=wrong-import-position
sys.path.append(os.path.realpath('.'))
sys.path.append(os.path.join(os.path.expanduser('~'), 'bnelearn'))

from bnelearn.experiment.configuration_manager import ConfigurationManager  # pylint: disable=import-error


if __name__ == '__main__':

    # running_configuration, logging_configuration, experiment_configuration, experiment_class = \
    #     fire.Fire()

    # Path is user-specific
    log_root_dir = os.path.join(
        os.path.expanduser('~'), 'bnelearn', 'experiments', 'pso_new-c')

    experiment_types = [
        'single_item_asymmetric_uniform_overlapping',
        'single_item_asymmetric_uniform_disjunct',
        'splitaward',
        'llg_full',
    ]

    for experiment_type in experiment_types:
    # Set up experiment
        experiment_config, experiment_class = \
            ConfigurationManager(
                experiment_type=experiment_type,
                n_runs=10,
                n_epochs=2000,
            ) \
            .set_setting(
            ) \
            .set_learning(
                # model_sharing=False,
                learner_type='PSOLearner',
                batch_size=2**17,
                pretrain_iters=0,
                learner_hyperparams={
                    'swarm_size': 64,
                    'topology': 'von_neumann',
                    'upper_bounds': 1,
                    'lower_bounds': -1,
                    'reevaluation_frequency': 10,
                    'inertia_weight': .5,
                    'cognition': .8,
                    'social': .8,
                    # 'pretrain_deviation': .2,
                }
                # pretrain_iters=500,
                # learner_hyperparams={
                #     'population_size': 64,
                #     'sigma': 1.,
                #     'scale_sigma_by_model_size': True,
                #     # 'regularize': {
                #     #     'inital_strength': inital_strength,
                #     #     'regularize_decay': regularize_decay,
                #     # },
                # }
            ) \
            .set_hardware(
                specific_gpu=5,
            ) \
            .set_logging(
                eval_batch_size=2**17,
                cache_eval_actions=True,
                util_loss_batch_size=2**11,
                util_loss_grid_size=2**10,
                util_loss_frequency=2000,
                best_response=False,
                log_root_dir=log_root_dir,
                save_tb_events_to_csv_detailed=True,
                stopping_criterion_frequency=1e8,
                save_models=True,
                plot_frequency=200,
            ) \
            .get_config()
        experiment = experiment_class(experiment_config)
        experiment.run()

        torch.cuda.empty_cache()
