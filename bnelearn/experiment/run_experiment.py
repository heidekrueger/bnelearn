import sys
sys.path.append('..')
import torch
import torch.nn as nn
import fire

from bnelearn.experiment.gpu_controller import GPUController
from bnelearn.experiment.configurations import *
from bnelearn.experiment.logger import LLGAuctionLogger, LLLLGGAuctionLogger, SingleItemAuctionLogger
from bnelearn.experiment.single_item_experiment import UniformSymmetricPriorSingleItemExperiment, \
    GaussianSymmetricPriorSingleItemExperiment

from bnelearn.experiment.combinatorial_experiment import LLGExperiment, LLLLGGExperiment
from bnelearn.experiment.multi_unit_experiment import MultiItemVickreyAuction2x2
import warnings

from dataclasses import dataclass, field, asdict

#TODO: Using locals() to directly create the dict 
# (https://stackoverflow.com/questions/2521901/get-a-list-tuple-dict-of-the-arguments-passed-to-a-function)
# fine with you? 
def run_single_item_uniform_symmetric(n_runs: int, n_epochs: int, 
                                      n_players: [int], payment_rule: str, model_sharing=True, u_lo=0, u_hi=1, 
                                      risk=1.0, 
                                      log_metrics = ['bne','l2','rmse','regret'], regret_batch_size=2**8, regret_grid_size=2**8,
                                      specific_gpu=1):

    running_configuration = RunningConfiguration(n_runs=n_runs, n_epochs=n_epochs, specific_gpu=specific_gpu, n_players=n_players)
    logging_configuration = LoggingConfiguration(log_metrics=log_metrics,
                                                 regret_batch_size=regret_batch_size,
                                                 regret_grid_size=regret_grid_size)
    experiment_configuration = ExperimentConfiguration(payment_rule=payment_rule, model_sharing=model_sharing,
                                                       u_lo=u_lo, u_hi=u_hi, risk=risk)
    experiment_class = UniformSymmetricPriorSingleItemExperiment
    return running_configuration, logging_configuration, experiment_configuration, experiment_class

def run_single_item_gaussian_symmetric(n_runs: int, n_epochs: int, 
                                       n_players: [int], payment_rule: str, model_sharing=True, valuation_mean=15, valuation_std=10, 
                                       risk=1.0, regret_batch_size=2**8, regret_grid_size=2**8,
                                       specific_gpu=1):
    experiment_params = locals()
    input_length = 1
    experiment_class = GaussianSymmetricPriorSingleItemExperiment
    return n_runs, n_epochs, n_players, specific_gpu, input_length, experiment_class, experiment_params

def run_llg(n_runs: int, n_epochs: int, 
            payment_rule: str, model_sharing=True, u_lo=[0,0,0], u_hi=[1,1,2],
            risk=1.0, regret_batch_size=2**8, regret_grid_size=2**8,
            specific_gpu=1):
    
    experiment_params = locals()
    n_players = [3]
    input_length = 1
    experiment_class = LLGExperiment
    return n_runs, n_epochs, n_players, specific_gpu, input_length, experiment_class, experiment_params

def run_llllgg(n_runs: int, n_epochs: int, 
               payment_rule: str, model_sharing=True, u_lo=[0,0,0,0,0,0], u_hi=[1,1,1,1,2,2],
               risk=1.0, regret_batch_size=2**8, regret_grid_size=2**8,
               specific_gpu=1):
    
    experiment_params = locals()
    n_players = [6]
    input_length = 2
    experiment_class = LLLLGGExperiment
    return n_runs, n_epochs, n_players, specific_gpu, input_length, experiment_class, experiment_params

def run_MultiItemVickreyAuction2x2(n_runs: int, n_epochs: int, 
                                   model_sharing=True, u_lo=[0,0], u_hi=[1,1],
                                   risk=1.0, regret_batch_size=2**8, regret_grid_size=2**8,
                                   specific_gpu=1):
    experiment_params = locals()
    n_players = [2]
    input_length = 2
    experiment_class = MultiItemVickreyAuction2x2
    return n_runs, n_epochs, n_players, specific_gpu, input_length, experiment_class, experiment_params

if __name__ == '__main__':
    '''
    Runs predefined experiments with individual parameters
    fire.Fire() asks you to decide for one of the experiments defined above
    by writing its name and define the required (and optional) parameters
    e.g.: 
        run_experiment.py run_single_item_uniform_symmetric 1 20 [2,3] 'first_price'

    alternatively instead of fire.Fire() use, e.g.:
        run_single_item_uniform_symmetric(1,20,[2,3],'first_price')

    '''
    #n_runs, n_epochs, n_players, specific_gpu, input_length, experiment_class, experiment_params = fire.Fire()
    #n_runs, n_epochs, n_players, specific_gpu, input_length, experiment_class, experiment_params = run_llg(1,20,'vcg')
    #n_runs, n_epochs, n_players, specific_gpu, input_length, experiment_class, experiment_params = run_single_item_uniform_symmetric(1,20, 2, 'first_price')
    #n_runs, n_epochs, n_players, specific_gpu, input_length, experiment_class, experiment_params = run_MultiItemVickreyAuction2x2(2, 20)
    running_configuration, logging_configuration, experiment_configuration, experiment_class  = run_single_item_uniform_symmetric(1,20, [2], 'first_price')

    gpu_configuration = GPUController(specific_gpu=running_configuration.specific_gpu)
    learning_configuration = LearningConfiguration()

    
    for i in running_configuration.n_players:
        experiment_configuration.n_players = i
    experiment = experiment_class(experiment_configuration, gpu_configuration, learning_configuration)
    experiment.run(epochs=running_configuration.n_epochs, n_runs=running_configuration.n_runs)
