import sys
import os
sys.path.append(os.path.realpath('.'))
import torch
import torch.nn as nn
import fire

from bnelearn.experiment.gpu_controller import GPUController
from bnelearn.experiment.configurations import *
#from bnelearn.experiment.logger import LLGAuctionLogger, LLLLGGAuctionLogger, SingleItemAuctionLogger
from bnelearn.experiment.single_item_experiment import UniformSymmetricPriorSingleItemExperiment, \
    GaussianSymmetricPriorSingleItemExperiment, TwoPlayerAsymmetricUniformPriorSingleItemExperiment

from bnelearn.experiment.combinatorial_experiment import LLGExperiment, LLLLGGExperiment
from bnelearn.experiment.multi_unit_experiment import MultiUnitExperiment, SplitAwardExperiment
import warnings

from dataclasses import dataclass, field, asdict

#TODO: Using locals() to directly create the dict 
# (https://stackoverflow.com/questions/2521901/get-a-list-tuple-dict-of-the-arguments-passed-to-a-function)
# fine with you?

def run_single_item_uniform_symmetric(n_runs: int, n_epochs: int, 
                                      n_players: [int], payment_rule: str, model_sharing=True, u_lo=0, u_hi=1, 
                                      risk=1.0, 
                                      log_metrics = ['opt','l2','regret'], regret_batch_size=2**8, regret_grid_size=2**8,
                                      specific_gpu=1):

    running_configuration = RunningConfiguration(n_runs=n_runs, n_epochs=n_epochs, specific_gpu=specific_gpu, n_players=n_players)
    logging_configuration = LoggingConfiguration(log_metrics=log_metrics,
                                                 regret_batch_size=regret_batch_size,
                                                 regret_grid_size=regret_grid_size,
                                                 max_epochs=n_epochs)
    experiment_configuration = ExperimentConfiguration(payment_rule=payment_rule, model_sharing=model_sharing,
                                                       u_lo=u_lo, u_hi=u_hi, risk=risk)
    experiment_class = UniformSymmetricPriorSingleItemExperiment
    return running_configuration, logging_configuration, experiment_configuration, experiment_class

def run_single_item_gaussian_symmetric(n_runs: int, n_epochs: int, 
                                      n_players: [int], payment_rule: str, model_sharing=True, valuation_mean=15, valuation_std=10, 
                                      risk=1.0, eval_batch_size = 2**16,
                                      log_metrics = ['opt','l2','regret'], regret_batch_size=2**8, regret_grid_size=2**8,
                                      specific_gpu=1):
    if eval_batch_size == 2**16:
        print("Using eval_batch_size of 2**16. Use at least 2**22 for proper experiment runs!")
    running_configuration = RunningConfiguration(n_runs=n_runs, n_epochs=n_epochs, specific_gpu=specific_gpu, n_players=n_players)
    logging_configuration = LoggingConfiguration(log_metrics=log_metrics,
                                                 regret_batch_size=regret_batch_size,
                                                 regret_grid_size=regret_grid_size,
                                                 eval_batch_size=eval_batch_size,
                                                 max_epochs=n_epochs)
    experiment_configuration = ExperimentConfiguration(payment_rule=payment_rule, model_sharing=model_sharing,
                                                       valuation_mean=valuation_mean, valuation_std=valuation_std, risk=risk)
    experiment_class = GaussianSymmetricPriorSingleItemExperiment
    return running_configuration, logging_configuration, experiment_configuration, experiment_class

def run_single_item_asymmetric_uniform(
        n_runs: int,
        n_epochs: int,
        payment_rule = 'first_price',
        model_sharing=False,
        u_lo=[0, 6], # [5, 5],     [0, 6]
        u_hi=[5, 7], # [15, 25],   [5, 7]
        risk=1.0,
        eval_batch_size = 2**18,
        log_metrics = ['opt','l2','regret'],
        regret_batch_size=2**8,
        regret_grid_size=2**8,
        specific_gpu=1
    ):
    n_players = [2]
    running_configuration = RunningConfiguration(n_runs=n_runs, n_epochs=n_epochs,
                                                 specific_gpu=specific_gpu, n_players=n_players)
    logging_configuration = LoggingConfiguration(log_metrics=log_metrics,
                                                 regret_batch_size=regret_batch_size,
                                                 regret_grid_size=regret_grid_size,
                                                 eval_batch_size=eval_batch_size,
                                                 max_epochs=n_epochs)
    experiment_configuration = ExperimentConfiguration(payment_rule=payment_rule, model_sharing=model_sharing,
                                                       u_lo=u_lo, u_hi=u_hi, risk=risk)
    experiment_class = TwoPlayerAsymmetricUniformPriorSingleItemExperiment
    return running_configuration, logging_configuration, experiment_configuration, experiment_class

def run_llg(n_runs: int, n_epochs: int, 
            payment_rule: str, model_sharing=True, u_lo=[0,0,0], u_hi=[1,1,2], 
            risk=1.0, 
            log_metrics = ['opt','l2','regret'], regret_batch_size=2**8, regret_grid_size=2**8,
            specific_gpu=1):

    n_players = [3]
    running_configuration = RunningConfiguration(n_runs=n_runs, n_epochs=n_epochs, specific_gpu=specific_gpu, n_players=n_players)
    logging_configuration = LoggingConfiguration(log_metrics=log_metrics,
                                                 regret_batch_size=regret_batch_size,
                                                 regret_grid_size=regret_grid_size,
                                                 max_epochs=n_epochs)
    experiment_configuration = ExperimentConfiguration(payment_rule=payment_rule, model_sharing=model_sharing,
                                                       u_lo=u_lo, u_hi=u_hi, risk=risk)
    experiment_class = LLGExperiment
    return running_configuration, logging_configuration, experiment_configuration, experiment_class

def run_llllgg(n_runs: int, n_epochs: int, 
            payment_rule: str, model_sharing=True, u_lo=[0,0,0,0,0,0], u_hi=[1,1,1,1,2,2], 
            risk=1.0,  eval_batch_size = 2**12,
            log_metrics = ['regret'], regret_batch_size=2**8, regret_grid_size=2**8,
            core_solver = "NoCore", specific_gpu=1):

    n_players = [6]
    running_configuration = RunningConfiguration(n_runs=n_runs, n_epochs=n_epochs, specific_gpu=specific_gpu, n_players=n_players)
    logging_configuration = LoggingConfiguration(log_metrics=log_metrics,
                                                 regret_batch_size=regret_batch_size,
                                                 regret_grid_size=regret_grid_size,
                                                 eval_batch_size=eval_batch_size,
                                                 max_epochs=n_epochs)
    experiment_configuration = ExperimentConfiguration(payment_rule=payment_rule, model_sharing=model_sharing,
                                                       u_lo=u_lo, u_hi=u_hi, risk=risk, core_solver=core_solver)
    experiment_class = LLLLGGExperiment
    return running_configuration, logging_configuration, experiment_configuration, experiment_class

def run_multiunit(
        n_runs: int, n_epochs: int,
        n_players: list=[2],
        payment_rule: str='vcg',
        n_units=2,
        log_metrics = ['opt','l2', 'regret'],
        model_sharing=True,
        u_lo=[0,0], u_hi=[1,1],
        risk=1.0,
        constant_marginal_values: bool=False,
        item_interest_limit: int=None,
        regret_batch_size=2**8,
        regret_grid_size=2**8,
        specific_gpu=1
    ):

    running_configuration = RunningConfiguration(
        n_runs=n_runs, n_epochs=n_epochs,
        specific_gpu=specific_gpu, n_players=[2]
    )
    logging_configuration = LoggingConfiguration(
        log_metrics=log_metrics,
        regret_batch_size=regret_batch_size,
        regret_grid_size=regret_grid_size,
        plot_points=1000,
        max_epochs=n_epochs
    )
    experiment_configuration = ExperimentConfiguration(
        payment_rule=payment_rule, n_units=n_units,
        model_sharing=model_sharing,
        u_lo=u_lo, u_hi=u_hi, risk=risk,
        constant_marginal_values=constant_marginal_values,
        item_interest_limit=item_interest_limit
    )
    experiment_class = MultiUnitExperiment
    return running_configuration, logging_configuration, experiment_configuration, experiment_class

def run_splitaward(
        n_runs: int, n_epochs: int,
        n_players: list=[2],
        payment_rule: str='first_price',
        n_units=2,
        model_sharing=True,
        log_metrics = ['opt','l2','regret'],
        u_lo=[1,1], u_hi=[1.4,1.4],
        risk=1.0,
        constant_marginal_values: bool=False,
        item_interest_limit: int=None,
        efficiency_parameter: float=0.3,
        regret_batch_size=2**8,
        regret_grid_size=2**8,
        specific_gpu=1
    ):

    running_configuration = RunningConfiguration(
        n_runs=n_runs, n_epochs=n_epochs,
        specific_gpu=specific_gpu, n_players=[2]
    )
    logging_configuration = LoggingConfiguration(
        log_metrics=log_metrics,
        regret_batch_size=regret_batch_size,
        regret_grid_size=regret_grid_size,
        max_epochs=n_epochs
    )
    experiment_configuration = ExperimentConfiguration(
        payment_rule=payment_rule, n_units=n_units,
        model_sharing=model_sharing,
        u_lo=u_lo, u_hi=u_hi, risk=risk,
        constant_marginal_values=constant_marginal_values,
        item_interest_limit=item_interest_limit,
        efficiency_parameter=efficiency_parameter
    )
    experiment_class = SplitAwardExperiment
    return running_configuration, logging_configuration, experiment_configuration, experiment_class

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
    #n_runs, n_epochs, n_players, specific_gpu, input_length, experiment_class, experiment_params = \
    #       run_single_item_uniform_symmetric(1,20, 2, 'first_price')

    # running_configuration, logging_configuration, experiment_configuration, experiment_class = \
    #      run_single_item_uniform_symmetric(1,20, [2], 'first_price', model_sharing=False)
    # running_configuration, logging_configuration, experiment_configuration, experiment_class = \
    #     run_single_item_gaussian_symmetric(1,20, [2], 'second_price')
    #running_configuration, logging_configuration, experiment_configuration, experiment_class =\
    #    run_llg(1,110,'nearest_zero',specific_gpu=1)
    #running_configuration, logging_configuration, experiment_configuration, experiment_class = \
    #    run_llllgg(1,310,'first_price')#,model_sharing=False)
    running_configuration, logging_configuration, experiment_configuration, experiment_class = \
       run_multiunit(n_runs=3, n_epochs=4000, n_players=[2], n_units=2, payment_rule='first_price')
    # running_configuration, logging_configuration, experiment_configuration, experiment_class = \
    #   run_splitaward(1, 500, [2])
    # running_configuration, logging_configuration, experiment_configuration, experiment_class = \
    #    run_single_item_asymmetric_uniform(n_runs=1, n_epochs=4000)

    gpu_configuration = GPUController(specific_gpu=running_configuration.specific_gpu)
    input_length = experiment_configuration.n_units \
        if experiment_configuration.n_units is not None else 1
    learning_configuration = LearningConfiguration(
        input_length=input_length,
        pretrain_iters=100
    )

    try:
        for i in running_configuration.n_players:
            experiment_configuration.n_players = i
            #TODO: filename needs a smarter solution.
            logging_configuration.update_file_name()
            experiment = experiment_class(experiment_configuration, learning_configuration,
                                          logging_configuration, gpu_configuration)
            experiment.run(epochs=running_configuration.n_epochs, n_runs=running_configuration.n_runs)

    except KeyboardInterrupt:
        print('\nKeyboardInterrupt: released memory after interruption')
        torch.cuda.empty_cache()
