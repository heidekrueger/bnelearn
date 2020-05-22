import os
import sys

from bnelearn.util.logging import log_experiment_configurations

sys.path.append(os.path.realpath('.'))
sys.path.append(os.path.join(os.path.expanduser('~'), 'bnelearn'))

import fire
import torch

from bnelearn.experiment.configurations import (ModelConfiguration, LearningConfiguration,
                                                LoggingConfiguration, RunningConfiguration, GPUConfiguration)
from bnelearn.experiment import (GaussianSymmetricPriorSingleItemExperiment,
                                 TwoPlayerAsymmetricUniformPriorSingleItemExperiment,
                                 UniformSymmetricPriorSingleItemExperiment,
                                 LLGExperiment, LLLLGGExperiment, MultiUnitExperiment, SplitAwardExperiment)
from bnelearn.experiment.presets import (llg, llllgg, multiunit,
                                         single_item_asymmetric_uniform_disjunct,
                                         single_item_asymmetric_uniform_overlapping,
                                         single_item_gaussian_symmetric,
                                         single_item_uniform_symmetric, splitaward)

if __name__ == '__main__':
    '''
    Runs predefined experiments with individual parameters
    fire.Fire() asks you to decide for one of the experiments defined above
    by writing its name and define the required (and optional) parameters
    e.g.:
        experiment.py single_item_uniform_symmetric 1 20 [2,3] 'first_price'

    alternatively instead of fire.Fire() use, e.g.:
        single_item_uniform_symmetric(1,20,[2,3],'first_price')

    '''

    # General run configs (not class-specific)
    n_runs = 1
    n_epochs = 5
    n_players = 2
    model_sharing = True
    enable_logging = True
    cuda = False

    # running_configuration, logging_configuration, experiment_configuration, experiment_class = \
    #     fire.Fire()
    experiment_config, experiment_class = \
        single_item_uniform_symmetric(n_runs=n_runs, n_epochs=n_epochs, n_players=n_players,
                                      payment_rule='first_price', model_sharing=model_sharing,
                                      logging=enable_logging, cuda=cuda, save_tb_events_to_csv_detailed=True,
                                      save_tb_events_to_binary_detailed=True,
                                      stopping_criterion_rel_util_loss_diff=0.001,
                                      pretrain_iters=10, batch_size=2 ** 8)
    experiment_config, experiment_class = \
        single_item_gaussian_symmetric(1, 7, [2], 'second_price', logging=enable_logging, eval_batch_size=2 ** 8)
    experiment_config, experiment_class = \
        llg(2, 100, 'nearest_zero', specific_gpu=1, logging=enable_logging)
    experiment_config, experiment_class = \
        llllgg(2, 10, 'first_price', model_sharing=False, logging=enable_logging)
    experiment_config, experiment_class = \
        multiunit(n_runs=2, n_epochs=10, n_players=[2], n_units=2, payment_rule='first_price', logging=enable_logging)
    experiment_config, experiment_class = \
        multiunit(n_runs=2, n_epochs=100, n_players=[2], n_units=2, payment_rule='first_price', logging=enable_logging)
    experiment_config, experiment_class = \
        splitaward(1, 100, [2], logging=enable_logging)
    experiment_config, experiment_class = \
        single_item_asymmetric_uniform_overlapping(n_runs=1, n_epochs=500, logging=enable_logging)
    experiment_config, experiment_class = \
        single_item_asymmetric_uniform_disjunct(n_runs=1, n_epochs=500, logging=enable_logging)


    try:
        experiment_class(experiment_config).run()

    except KeyboardInterrupt:
        print('\nKeyboardInterrupt: released memory after interruption')
        torch.cuda.empty_cache()
