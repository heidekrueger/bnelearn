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
            experiment_type='llg_full',
            n_runs=1,
            n_epochs=500
            ) \
        .set_setting(
            # risk=1.1
            ) \
        .set_logging(
            util_loss_batch_size=2**10,
            util_loss_grid_size=2**8,
            util_loss_frequency=25,
            log_root_dir=log_root_dir,
            save_tb_events_to_csv_detailed=True,
            best_response=True
            ) \
        .set_learning(
            pretrain_iters=5
            ) \
        .set_hardware(
            specific_gpu=1
            ) \
        .set_logging() \
        .get_config()
    experiment = experiment_class(experiment_config)
    experiment.run()
   
    torch.cuda.empty_cache()
