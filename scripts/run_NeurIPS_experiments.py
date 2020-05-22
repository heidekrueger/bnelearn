import os
import sys

sys.path.append(os.path.realpath('.'))
sys.path.append(os.path.join(os.path.expanduser('~'), 'bnelearn'))

import torch

from bnelearn.experiment.configurations import (ExperimentConfiguration, LearningConfiguration,
                                                LoggingConfiguration, RunningConfiguration)
from bnelearn.experiment.gpu_controller import GPUController
from bnelearn.experiment import (LLGExperiment, LLLLGGExperiment)
from bnelearn.experiment.presets import (llg, llllgg)

if __name__ == '__main__':
    '''
    Experiments performed for NeurIPS submission.
    Only payment_rule, n_players and log_metrics are changed accordingly.
    -------------------testing------------------
    pertubation size: ->64 vs 128 
    lr: ->0.003 vs 0.001
    optimizer: ->adam vs SGD
    '''
    specific_gpu = 6
    #LLG: first_price, VCG, nearest_vcg, nearest_zero, nearest_bid
    #LLLLGG: first_price, VCG, nearest_vcg
    payment_rule = 'nearest_bid'
    #LLLLGGExperiment
    experiment_class = LLGExperiment 

    running_configuration = RunningConfiguration(n_runs = 2, n_epochs = 5000, specific_gpu = specific_gpu, n_players = [3])
    logging_configuration = LoggingConfiguration(log_metrics = ['opt','l2','util_loss'],#['util_loss'],#
                                                 util_loss_batch_size = 2**12,util_loss_grid_size = 2**10, util_loss_frequency = 100,
                                                 #stopping_criterion_rel_util_loss_diff = 0.0005, stopping_criterion_batch_size = 2**10,
                                                 #stopping_criterion_grid_size = 2**9,
                                                 save_tb_events_to_binary_detailed = True, save_tb_events_to_csv_detailed = True)

    if experiment_class is LLGExperiment:
        experiment_configuration = ExperimentConfiguration(payment_rule, u_lo = [0,0,0], u_hi = [1,1,2])
    elif experiment_class is LLLLGGExperiment:
        experiment_configuration = ExperimentConfiguration(payment_rule, u_lo = [0]*6, u_hi = [1,1,1,1,2,2], core_solver='qpth')
    else:
        SystemExit("Only local global experiments.")

    gpu_configuration = GPUController(specific_gpu=running_configuration.specific_gpu)
    learning_configuration = LearningConfiguration(
        pretrain_iters=500,
        batch_size=2**18,
        learner_hyperparams = {'population_size': 128,
                               'sigma': 1.,
                               'scale_sigma_by_model_size': True})

    try:
        for i in running_configuration.n_players:
            experiment_configuration.n_players = i
            experiment = experiment_class(experiment_configuration, learning_configuration,
                                          logging_configuration, gpu_configuration)
            experiment.run(epochs=running_configuration.n_epochs, n_runs=running_configuration.n_runs)

    except KeyboardInterrupt:
        print('\nKeyboardInterrupt: released memory after interruption')
        torch.cuda.empty_cache()
