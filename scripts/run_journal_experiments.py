import os
import sys

sys.path.append(os.path.realpath('.'))
sys.path.append(os.path.join(os.path.expanduser('~'), 'bnelearn'))

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
                                         single_item_gaussian_symmetric, mineral_rights,
                                         single_item_uniform_symmetric, splitaward)

if __name__ == '__main__':
    '''
    Experiments performed for Journal submission.
    Experiment, payment_rule, n_players and log_metrics are changed accordingly.
    -------------------architecture------------------
    size: ->[10,10]
    pertubation size: ->64
    lr: ->0.001
    optimizer: ->adam
    '''
    # Set Learner
    hidden_nodes_param = [10,10]
    optimizer_type_param = 'adam'
    optimizer_hyperparams_param = {'lr': 1e-3}

    # Logging Config
    enable_logging = True
    # Running Config
    n_players = [2]
    # GPU Config
    specific_gpu = 5

    # Single Item
    # _, logging_configuration, experiment_configuration, experiment_class = \
        # single_item_uniform_symmetric(0, 0, [0], 'first_price', model_sharing=True, logging=enable_logging)
        # single_item_gaussian_symmetric(0, 0, [0], 'first_price', model_sharing=True, logging=enable_logging)
        # single_item_asymmetric_uniform_overlapping(0, 0, 'first_price', logging=enable_logging)
        # single_item_asymmetric_uniform_disjunct(0, 0, 'first_price', logging=enable_logging)

    # Multi Unit
    _, logging_configuration, experiment_configuration, experiment_class = \
        multiunit(0, 0, [0], n_units=2, payment_rule='first_price', logging=enable_logging)
        # multiunit(0, 0, [0], n_units=2, payment_rule='uniform', logging=enable_logging)
        # splitaward(0, 0, [0], logging=enable_logging)

    # LLG - correlated
    # _, logging_configuration, experiment_configuration, experiment_class =\
    #     llg(0,0,'nearest_zero', gamma = 0.5, model_sharing=True, logging=enable_logging)
    #     llg(0,0,'nearest_vcg', gamma = 0.5, model_sharing=True, logging=enable_logging)
    #     llg(0,0,'nearest_bid', gamma = 0.5, model_sharing=True, logging=enable_logging)




    running_configuration = RunningConfiguration(n_runs = 2, n_epochs = 5000, specific_gpu = specific_gpu, n_players = n_players)
    logging_configuration = LoggingConfiguration(log_metrics = logging_configuration.log_metrics,
                                                 util_loss_batch_size = 2**12,util_loss_grid_size = 2**13, util_loss_frequency = 100,
                                                 save_tb_events_to_binary_detailed = True, save_tb_events_to_csv_detailed = True,
                                                 plot_frequency = 6000, plot_points=0)

    gpu_configuration = GPUController(specific_gpu=running_configuration.specific_gpu)
    learning_configuration = LearningConfiguration(
        hidden_nodes = hidden_nodes_param,
        pretrain_iters=500,
        batch_size=2**18,
        optimizer_type=optimizer_type_param,
        learner_hyperparams = {'population_size': 64,
                               'sigma': 1.,
                               'scale_sigma_by_model_size': True},
        optimizer_hyperparams= optimizer_hyperparams_param)

    try:
        for i in running_configuration.n_players:
            experiment_configuration.n_players = i
            experiment = experiment_class(experiment_configuration, learning_configuration,
                                          logging_configuration, gpu_configuration)
            experiment.run(epochs=running_configuration.n_epochs, n_runs=running_configuration.n_runs)

    except KeyboardInterrupt:
        print('\nKeyboardInterrupt: released memory after interruption')
        torch.cuda.empty_cache()
