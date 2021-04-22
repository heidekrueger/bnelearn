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
        os.path.expanduser('~'), 'bnelearn', 'experiments')

    # Set up experiment
    experiment_config, experiment_class = \
        ConfigurationManager(
            # experiment_type='single_item_asymmetric_uniform_overlapping',
            # experiment_type='single_item_asymmetric_uniform_disjunct',
            experiment_type='llg_full',
            # seeds=[2],
            n_runs=10,
            n_epochs=2000,
            ) \
        .set_setting(
            payment_rule='mrcs_favored',
            ) \
        .set_learning(
            batch_size=2**17,
            pretrain_iters=500,
            ) \
        .set_hardware(
            specific_gpu=1,
            ) \
        .set_logging(
            eval_batch_size=2**20,
            cache_eval_actions=True,
            util_loss_batch_size=2**10,
            util_loss_grid_size=2**10,
            util_loss_frequency=1000,
            best_response=True,
            log_root_dir=log_root_dir,
            save_tb_events_to_csv_detailed=True,
            stopping_criterion_frequency=1e8,
            save_models=True,
            plot_frequency=100,
            ) \
        .get_config()
    experiment = experiment_class(experiment_config)
    experiment.run()

    torch.cuda.empty_cache()
