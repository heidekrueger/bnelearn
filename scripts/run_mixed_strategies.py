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
        os.path.expanduser('~'), 'bnelearn', 'experiments', 'mixed-strategies')

    # Set up experiment
    experiment_config, experiment_class = \
        ConfigurationManager(
            experiment_type="single_item_uniform_symmetric",
            # experiment_type='single_item_asymmetric_uniform_overlapping',
            # experiment_type='single_item_asymmetric_uniform_disjunct',
            # experiment_type='llg_full',
            seeds=[69],
            n_runs=1,
            n_epochs=2000,
            ) \
        .set_setting(
            # payment_rule='mrcs_favored',
            # risk=1.1
            ) \
        .set_learning(
            # learner_type='PGLearner',
            batch_size=2**17,
            # pretrain_to_bne=0,
            pretrain_iters=500,
            learner_hyperparams={
                'population_size': 64,
                'sigma': .1,
                'scale_sigma_by_model_size': True,
                # 'regularization': {
                #     'inital_strength': .05,
                #     'regularize_decay': .999
                #     }
            },
            mixed_strategy='normal'
            ) \
        .set_hardware(
            specific_gpu=1,
            ) \
        .set_logging(
            eval_batch_size=2**17,
            cache_eval_actions=False,
            util_loss_batch_size=2**8,
            util_loss_grid_size=2**14,
            util_loss_frequency=100,
            best_response=True,
            log_root_dir=log_root_dir,
            save_tb_events_to_csv_detailed=True,
            plot_frequency=100,
            ) \
        .get_config()
    experiment = experiment_class(experiment_config)
    experiment.run()

    torch.cuda.empty_cache()
