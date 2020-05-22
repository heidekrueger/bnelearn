import os
import sys

sys.path.append(os.path.realpath('.'))
sys.path.append(os.path.join(os.path.expanduser('~'), 'bnelearn'))

import fire
import torch

from bnelearn.experiment.configurations import (ExperimentConfiguration, LearningConfiguration,
                                                LoggingConfiguration, RunningConfiguration)
from bnelearn.experiment.gpu_controller import GPUController
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
    enable_logging = True

    # General run configs
    n_runs = 1
    n_epochs = 20000
    n_players = []


    # running_configuration, logging_configuration, experiment_configuration, experiment_class = \
    #     fire.Fire()
    running_configuration, logging_configuration, experiment_configuration, experiment_class = \
         single_item_uniform_symmetric(1, 5, [2], 'first_price', model_sharing=True, logging=enable_logging)
    # logging_configuration.save_tb_events_to_binary_detailed = True
    # logging_configuration.save_tb_events_to_csv_detailed = True

    #running_configuration, logging_configuration, experiment_configuration, experiment_class = \
    #    single_item_gaussian_symmetric(1, 7, [2], 'second_price', logging=enable_logging, eval_batch_size=2**8)

    # running_configuration, logging_configuration, experiment_configuration, experiment_class =\
    #    llg(2,100,'nearest_zero',specific_gpu=1, logging=enable_logging)
    # running_configuration, logging_configuration, experiment_configuration, experiment_class = \
    #    llllgg(2,10,'first_price',model_sharing=False, logging=enable_logging)
    #running_configuration, logging_configuration, experiment_configuration, experiment_class = \
    #  multiunit(n_runs=2, n_epochs=10, n_players=[2], n_units=2, payment_rule='first_price', logging=enable_logging)

    # running_configuration, logging_configuration, experiment_configuration, experiment_class = \
    #  multiunit(n_runs=2, n_epochs=100, n_players=[2], n_units=2, payment_rule='first_price', logging=enable_logging)
    # running_configuration, logging_configuration, experiment_configuration, experiment_class = \
    #   splitaward(1, 100, [2], logging=enable_logging)
    #running_configuration, logging_configuration, experiment_configuration, experiment_class = \
    #   single_item_asymmetric_uniform_overlapping(n_runs=1, n_epochs=500, logging=enable_logging)
    #running_configuration, logging_configuration, experiment_configuration, experiment_class = \
    #   single_item_asymmetric_uniform_disjunct(n_runs=1, n_epochs=500, logging=enable_logging)

    gpu_configuration = GPUController(specific_gpu=running_configuration.specific_gpu, cuda=False)
    learning_configuration = LearningConfiguration(
        pretrain_iters=10,
        batch_size=2**8
    )

    # General logging configs
    logging_configuration.stopping_criterion_rel_util_loss_diff = 0.001
    logging_configuration.save_tb_events_to_csv_detailed=True
    logging_configuration.save_tb_events_to_binary_detailed=True

    try:
        for i in running_configuration.n_players:
            experiment_configuration.n_players = i
            experiment = experiment_class(experiment_configuration, learning_configuration,
                                          logging_configuration, gpu_configuration)
            experiment.run(epochs=running_configuration.n_epochs, n_runs=running_configuration.n_runs)

    except KeyboardInterrupt:
        print('\nKeyboardInterrupt: released memory after interruption')
        torch.cuda.empty_cache()
