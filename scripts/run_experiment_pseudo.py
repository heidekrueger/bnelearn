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

    # path is user-specific
    log_root_dir = os.path.join(os.path.expanduser('~'), 'bnelearn', 'experiments', 'psuedo')

    # Run from a file
    # experiment_config = logging.get_experiment_config_from_configurations_log()
    # experiment_class = ConfigurationManager \
    #    .get_class_by_experiment_type(experiment_config.experiment_class)



    ### SINGLE ITEM EXPERIMENTS ###

    experiment_config, experiment_class = \
        ConfigurationManager(
            experiment_type='single_item_gaussian_symmetric',
            n_runs=3,
            n_epochs=10000
            ) \
        .set_setting(
            n_players = 5
            ) \
        .set_logging(
            eval_batch_size=2**22,
            util_loss_batch_size=2**10,
            util_loss_grid_size=2**10,
            util_loss_frequency=100,
            cache_eval_actions=True,
            log_root_dir=log_root_dir,
            save_tb_events_to_csv_detailed=True
            ) \
        .set_learning(
            model_sharing=True,
            sampling_method = "pseudorandom"
            ) \
        .set_hardware(
            specific_gpu=3
        ) \
        .get_config()

    experiment = experiment_class(experiment_config)
    experiment.run()
    torch.cuda.empty_cache()