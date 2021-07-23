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
import torch.nn as nn
import itertools

# put bnelearn imports after this.
# pylint: disable=wrong-import-position
sys.path.append(os.path.realpath('.'))
sys.path.append(os.path.join(os.path.expanduser('~'), 'bnelearn'))

from bnelearn.experiment.configuration_manager import ConfigurationManager  # pylint: disable=import-error


if __name__ == '__main__':


    # Parameter definition
    ## varying parameters
    experiment_types = ['single_item_symmetric_uniform_all_pay', 'single_item_asymmetric_uniform_all_pay']
    risks = [i/10 for i in range(1, 11)]
    players = [2, 4]
    varying_params = [experiment_types, risks, players]

    ## fixed parameters
    pretrain_iters = 500
    batch_size = 2**17
    eval_batch_size = 2**17
    util_loss_grid_size = 2 **10
    util_loss_batch_size = 2**12
    util_loss_frequency = 1000
    stopping_criterion_frequency = 100000
    n_epochs = 3500
    gpu = 2
    
    log_root_dir = os.path.join(os.path.expanduser('~'), 'bnelearn', 'all_pay', 'experiments')

    # Generate experiment configurations
    configs = list(itertools.product(*varying_params))

    # Run experiments
    for i, config in enumerate(configs):

        print("Executing experiment number {}".format(i))

        exp_type = config[0]
        risk = config[1]
        n_player = config[2]

        experiment_config, experiment_class = ConfigurationManager(experiment_type=exp_type, n_runs=1, n_epochs=n_epochs) \
        .set_setting(n_players=n_player, risk=risk) \
        .set_learning(pretrain_iters=pretrain_iters, batch_size=batch_size) \
        .set_logging(log_root_dir=log_root_dir, save_tb_events_to_csv_detailed=True, eval_batch_size=eval_batch_size, util_loss_grid_size=util_loss_grid_size, util_loss_batch_size=util_loss_batch_size, util_loss_frequency=util_loss_frequency, stopping_criterion_frequency=stopping_criterion_frequency) \
        .set_hardware(specific_gpu=gpu).get_config()

    
        ## Run experiment
        try:
            experiment = experiment_class(experiment_config)

            # Could only be done here and not inside Experiment itself while the checking depends on Experiment subclasses
            if ConfigurationManager.experiment_config_could_be_saved_properly(experiment_config):
                experiment.run()
            else:
                raise Exception('Unable to perform the correct serialization')

        except KeyboardInterrupt:
            print('\nKeyboardInterrupt: released memory after interruption')
            torch.cuda.empty_cache()
