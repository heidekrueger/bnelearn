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
                              'experiments', 'stochastic')
    experiment_config, experiment_class=ConfigurationManager(
        experiment_type='splitaward',
        # experiment_type='single_item_asymmetric_uniform_overlapping',
        # experiment_type='single_item_asymmetric_uniform_disjunct',
        n_runs=1,
        n_epochs=2000) \
        .set_hardware(
            specific_gpu=1) \
        .set_setting() \
        .set_logging(
            log_root_dir=log_root_dir) \
        .set_learning(
            optimizer_hyperparams={'lr': 1e-2},
            pretrain_iters=10,
            batch_size=2**18,
            model_sharing=True,
            mixed_strategy='normal'
            ) \
        .set_logging(
            eval_batch_size=2**22,
            util_loss_batch_size=2**8,
            util_loss_grid_size=2**7,
            plot_frequency=50,
            plot_points=500) \
        .get_config()
    experiment=experiment_class(experiment_config)

    try:
        experiment.run()
    except KeyboardInterrupt:
        torch.cuda.empty_cache()
