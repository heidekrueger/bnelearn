import os
import sys
import torch

# put bnelearn imports after this.
# pylint: disable=wrong-import-position
sys.path.append(os.path.realpath('.'))
sys.path.append(os.path.join(os.path.expanduser('~'), 'bnelearn'))

from bnelearn.experiment.configuration_manager import ConfigurationManager


if __name__ == '__main__':

    log_root_dir=os.path.join(os.path.expanduser('~'), 'bnelearn',
                              'experiments', 'mixed-strategies')
    experiment_config, experiment_class = \
        ConfigurationManager(
            # experiment_type='splitaward',
            # experiment_type='single_item_asymmetric_uniform_overlapping',
            # experiment_type='single_item_asymmetric_uniform_disjunct',
            experiment_type='llg_full',
            n_runs=1,
            seeds=[0],
            n_epochs=2000
        ) \
        .set_hardware(
            specific_gpu=7
        ) \
        .set_setting() \
        .set_logging(
            log_root_dir=log_root_dir
        ) \
        .set_learning(
            learner_hyperparams={
                'population_size': 64,
                'sigma': 1.,
                'scale_sigma_by_model_size': True,
                # 'regularization':{
                #     'inital_strength': 0.2,
                #     'regularize_decay': 0.999
                # }
            },
            # optimizer_hyperparams={'lr': 1e-2},   
            pretrain_iters=500,
            batch_size=2**15,
            model_sharing=False,
            mixed_strategy='uniform'
        ) \
        .set_logging(
            eval_batch_size=2**15,
            util_loss_batch_size=2**8,
            util_loss_grid_size=2**8,
            best_response=True,
            plot_frequency=50,
            plot_points=500,
            cache_eval_actions=False,
        ) \
        .get_config()

    experiment = experiment_class(experiment_config)

    try:
        experiment.run()
    except KeyboardInterrupt:
        torch.cuda.empty_cache()
