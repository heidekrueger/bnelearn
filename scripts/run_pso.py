"""
Runs predefined experiments with individual parameters
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

    # Path is user-specific
    log_root_dir = os.path.join(
        os.path.expanduser('~'), 'bnelearn', 'experiments', 'pso-final')

    experiment_types = [
        'single_item_uniform_symmetric',
        'single_item_asymmetric_uniform_overlapping',
        'llg',
        'llg_full',
    ]
    learners = [
        {
            'learner_type': 'ESPGLearner',
            'pretrain_iters': 500,
            'learner_hyperparams': {
                'population_size': 64,
                'sigma': 1.,
                'scale_sigma_by_model_size': True
            },
        },
        {
            'learner_type': 'PSOLearner',
            'pretrain_iters': 0,
            'learner_hyperparams': {
                'swarm_size': 64,
                'topology': 'von_neumann',
                # 'upper_bounds': 1,
                # 'lower_bounds': -1,
                'reevaluation_frequency': 10,
                'inertia_weight': .5,
                'cognition': .8,
                'social': .8,
                # 'pretrain_deviation': .2,
            }
        }
    ]
    for experiment_type in experiment_types:
        for learner in learners:
            print(f'\n###\nRunning `{experiment_type}` with {learner}')
            log_root_dir_test = os.path.join(log_root_dir, learner['learner_type'])
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
                    batch_size=2**17,
                    learner_type=learner['learner_type'],
                    pretrain_iters=learner['pretrain_iters'],
                    learner_hyperparams=learner['learner_hyperparams'],
                ) \
                .set_hardware(
                    specific_gpu=1,
                ) \
                .set_logging(
                    eval_batch_size=2**17,
                    cache_eval_actions=True,
                    util_loss_batch_size=2**11,
                    util_loss_grid_size=2**10,
                    util_loss_frequency=2000,
                    best_response=True,
                    log_root_dir=log_root_dir_test,
                    save_tb_events_to_csv_detailed=True,
                    stopping_criterion_frequency=1e8,
                    save_models=True,
                    plot_frequency=2000,
                ) \
                .get_config()
            experiment = experiment_class(experiment_config)
            experiment.run()

            torch.cuda.empty_cache()
