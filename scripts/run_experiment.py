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

# put bnelearn imports after this.
# pylint: disable=wrong-import-position
sys.path.append(os.path.realpath('.'))
sys.path.append(os.path.join(os.path.expanduser('~'), 'bnelearn'))

from bnelearn.experiment.configuration_manager import ConfigurationManager  # pylint: disable=import-error


if __name__ == '__main__':

    # path is user-specific
    log_root_dir = os.path.join(os.path.expanduser('~'), 'bnelearn', 'experiments', 'debug')
    
    experiment_config, experiment_class = ConfigurationManager(experiment_type='single_item_symmetric_uniform_all_pay', n_runs=1, n_epochs=5000) \
        .set_setting(n_players=2, regret=[0.23153299,  0.18192447]) \
        .set_learning(pretrain_iters = 500, batch_size=2**18) \
        .set_logging(log_root_dir=log_root_dir, eval_batch_size=2**15, util_loss_grid_size=2**10, util_loss_batch_size=2**12, 
                     util_loss_frequency=1000, save_models=False) \
        .set_hardware(specific_gpu=6).get_config()

    experiment = experiment_class(experiment_config)
    experiment.run()
    torch.cuda.empty_cache()
