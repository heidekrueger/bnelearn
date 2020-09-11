import os
import sys
import torch

sys.path.append(os.path.realpath('.'))
sys.path.append(os.path.join(os.path.expanduser('~'), 'bnelearn'))

#pylint: disable=wrong-import-position
from bnelearn.experiment.configuration_manager import ConfigurationManager 

if __name__ == '__main__':
    #pylint: disable=pointless-string-statement
    """
    Runs predefined experiments with interdependencies.

    TODO:
        - Create test for all settings and write data to tex-table
        - Create test for different gammas
    """

    # User Parameters
    log_root_dir = os.path.join(os.path.expanduser('~'), 'bnelearn', 'experiments')
    specific_gpu = 7
    n_runs = 2
    n_epochs = 2

    # LLG
    experiment_config, experiment_class = ConfigurationManager(experiment_type='llg') \
        .with_correlation(gamma=0.0) \
        .get_config(log_root_dir=log_root_dir, n_runs=n_runs, n_epochs=n_epochs, specific_gpu=specific_gpu)
                    # payment_rule='proxy')

    # Mineral rights
    experiment_config, experiment_class = ConfigurationManager(experiment_type='mineral_rights') \
        .get_config(log_root_dir=log_root_dir, n_runs=n_runs, n_epochs=n_epochs, specific_gpu=specific_gpu)

    # Affiliated observations
    experiment_config, experiment_class = ConfigurationManager(experiment_type='affiliated_observations') \
        .get_config(log_root_dir=log_root_dir, n_runs=n_runs, n_epochs=n_epochs, specific_gpu=specific_gpu)

    try:
        experiment = experiment_class(experiment_config)
        if experiment.known_bne:
            experiment.logging.log_metrics = {
                'opt': True,
                'l2': True,
                'util_loss': True
            }
        experiment.logging.util_loss_batch_size = 2**10
        experiment.logging.util_loss_grid_size = 2**10
        experiment.logging.util_loss_frequency = 200

        experiment.run()

    except KeyboardInterrupt:
        print('\nKeyboardInterrupt: released memory after interruption')
        torch.cuda.empty_cache()
