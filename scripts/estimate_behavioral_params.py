"""
Main Script for estimating parameters given experimental data.
"""

import os
import datetime
import sys
import pandas as pd

from torch.utils.tensorboard import SummaryWriter

from copy import deepcopy


import torch

# put bnelearn imports after this.
# pylint: disable=wrong-import-position
sys.path.append(os.path.realpath('.'))
sys.path.append(os.path.join(os.path.expanduser('~'), 'bnelearn'))

sys.path.remove('/home/kohring/bnelearn')


from bnelearn.parameter_estimators.optimizers import Bayesian_Optimizer  
from bnelearn.parameter_estimators.estimation_module import ClosedEstimator
from bnelearn.parameter_estimators.evaluation_module import CurveEvaluation, StaticsEvaluation


def run_optim(param_bounds, metric, run_id, full, path, n_players, data_loc, method,
              initial_samples, gpu, iterations, maximization, risk_function, curveEval, 
              n_id, exp_id, closed_only):

    writer = SummaryWriter(path)

    if curveEval:
        evaluation_module = CurveEvaluation(n_players=n_players, data_location=data_loc, 
                                        full_feedback=full, method=method, writer=writer)
    else:
        evaluation_module = StaticsEvaluation(data_location=data_loc, writer=writer)

    if metric == "Regret" and closed_only:

        estimation_module = ClosedEstimator("Regret")

        optimizer = Bayesian_Optimizer(initial_samples=initial_samples, param_bounds=param_bounds, n_players=n_players, gpu=gpu, 
                                                log_root_dir=path, evaluation_module=evaluation_module, iterations=iterations, 
                                                maximization=maximization, risk_function=risk_function, estimation_module=estimation_module)
    else:
        optimizer = Bayesian_Optimizer(initial_samples=initial_samples, param_bounds=param_bounds, n_players=n_players, gpu=gpu, 
                                                log_root_dir=path, evaluation_module=evaluation_module, iterations=iterations, 
                                                maximization=maximization, risk_function=risk_function)

    performances, params, estimations = optimizer.optimize()
    # plot metrics
    metrics = pd.DataFrame(params)
    metrics.columns = ['Risk', 'Regret_Beta', 'Regret_Gamma', 'Sub_ID']
    metrics['Acc'] = performances
    metrics['Type'] = metric 
    metrics['Run'] = run_id
    metrics['N_id'] = n_id
    metrics['exp_id'] = exp_id
    metrics['N'] = n_players
    metrics['Full_Feedback'] = full
    if risk_function == "default" and metric != 'Regret':
        metrics['Exp_Type'] = f'{n_players}_{full}_POW'
    elif risk_function == "CRRA":
        metrics['Exp_Type'] = f'{n_players}_{full}_CRRA'
    else:
        metrics['Exp_Type'] = f'{n_players}_{full}_Regret'

    # plot estimations
    est = pd.DataFrame()
    
    for estimation in estimations:
        est = est.append(pd.DataFrame(estimation.squeeze().tolist()))

    est.columns = ['Value', 'Estimation', 'Sub_ID']
    est['Run'] = run_id
    est['N_id'] = n_id
    est['exp_id'] = exp_id

    return metrics, est

def optimize(bo_params: dict, log_root_dir: str, gpu: int):

    """ Function to identify the (behavioral) parameters of a specific behavioral model """
    
    # Check if multiple time steps or single file
    data_path = bo_params['data_location']

    if os.path.isfile(data_path):
        bo_params['evaluation_module'] = bo_params['evaluation_module'](data_location=data_path, treatment=bo_params['treatment'], gpu=gpu, n_players=bo_params['n_players'], u_hi=bo_params['u_hi'])  # instantiate evaluation module
        optimizer = Bayesian_Optimizer(bo_params=bo_params, log_root_dir=log_root_dir, gpu=gpu)
        performance = optimizer.optimize()
    else:
        
        files = os.listdir(data_path)

        for f in files: # maybe do this more intelligent
            bo_params['evaluation_module'] = bo_params['evaluation_module'](data_location=f, n_players=bo_params['n_players'])  # instantiate evaluation module
            optimizer = Bayesian_Optimizer(bo_params=bo_params, log_root_dir=log_root_dir, gpu=gpu)
            performance, estimation = optimizer.optimize()
        

    return performance


def run_estimation(log_root_dir: str, run: int, behaviroal_models: dict, gpu: int, bo_params_clean: dict, experiment: dict):

    """ Estimates the behavioral parameter for each behavioral model """
    
    res = {}

    for bm in behaviroal_models:

        bo_params = deepcopy(bo_params_clean)

        bo_params['behavioral_model'] = {'type': bm, 'param_bounds': behavioral_models[bm]} 
        performance, param, model = optimize(bo_params=bo_params, log_root_dir=log_root_dir, gpu=gpu)   
        res[f'{bm}'] = {
            'performance': performance,
            'param': param, 
            'model': model
        }

    return res

def _generate_result_table(res: dict):

    cols = ['Experiment', 'N_Players', 'Treatment', 'Run', 'Behavioral Model','Behavioral Parameter', 'Performance']

    results = pd.DataFrame(columns=cols)

    for exp in res:
        for n_player in res[exp]:
            for treatment in res[exp][n_player]:
                for run in res[exp][n_player][treatment]:
                    for bm in res[exp][n_player][treatment][run]:
                        results = results.append(pd.DataFrame([[exp, n_player, treatment, run, bm, res[exp][n_player][treatment][run][bm]['param'], res[exp][n_player][treatment][run][bm]['performance']]], columns=cols))
                        # tbd: use best model for plotting purposes


    return results

if __name__ == '__main__':

    # configuration
    num_runs = 5

    scaled_value = True

    if scaled_value:
        experiments = {
            'first_price': {
                'n_players':  [5, 10],
                'treatment': ['cross_over', 'dual'],
                'data_location': '/home/ewert/data/Kagel/FP.csv' if not scaled_value else '/home/ewert/data/Kagel/FP_scaled.csv',
                'payment_rule': 'first_price'
            },
            'second_price': {
                'n_players':  [5, 10],
                'treatment': ['cross_over', 'dual'],
                'data_location': '/home/ewert/data/Kagel/SP.csv' if not scaled_value else '/home/ewert/data/Kagel/SP_scaled.csv',
                'payment_rule': 'second_price'
            },
            'third_price': {
                'n_players':  [5, 10],
                'treatment': ['cross_over', 'dual'],
                'data_location': '/home/ewert/data/Kagel/TP_norm.csv' if not scaled_value else '/home/ewert/data/Kagel/TP_scaled.csv',
                'payment_rule': 'third_price'
            },
        }
    else:
        experiments = {
            'first_price': {
                'n_players':  [5, 10],
                'treatment': ['cross_over', 'dual'],
                'data_location': '/home/ewert/data/Kagel/FP.csv',
                'payment_rule': 'first_price'
            },
            'second_price': {
                'n_players':  [5, 10],
                'treatment': ['cross_over', 'dual'],
                'data_location': '/home/ewert/data/Kagel/SP.csv',
                'payment_rule': 'second_price'
            },
            'third_price': {
                'n_players':  [5, 10],
                'treatment': ['cross_over', 'dual'],
                'data_location': '/home/ewert/data/Kagel/TP.csv',
                'payment_rule': 'third_price'
            },
        }
    
    eval_technique = "curve"
    evaluation_module = CurveEvaluation
 
    behavioral_models = {
        'risk': [0.001, 1],
        'regret': [[0, 4], [0, 4]]
    }

    gpu = 3

    # bo config

    bo_params_clean = {
        'initial_samples': 5,
        'iterations': 7,
        'experiment': 'single_item_uniform_symmetric',
        'epochs': 1,
        'objective': torch.min if eval_technique == 'curve' else torch.max,
        'evaluation_module': evaluation_module,
        'learner': 'PGLearner', # PGLearner, ESPGLearner
        'closed_form': None,
        'u_hi': [10] if scaled_value else [1]
    }

    log_root_dir = os.path.join(os.path.expanduser('~'), 'logging', 'behavioral', f'{eval_technique}', f'{bo_params_clean["learner"]}', '{}'.format(datetime.datetime.now()))
    path_res = f'{log_root_dir}/results/'

    os.makedirs(log_root_dir)
    os.mkdir(path_res)

    exp_res = {}
    
    for exp_name in experiments:

        exp = experiments[exp_name]

        player_res = {}

        for n_players in exp['n_players']:

            treatment_res = {}

            for treatment in exp['treatment']:

                runs = {}

                # run experiments mulitple times 
                for run in range(num_runs):

                    bo_params = deepcopy(bo_params_clean)

                    # add experimental settings to config
                    bo_params['n_players'] = n_players
                    bo_params['treatment'] = treatment
                    bo_params['data_location'] = exp['data_location']
                    bo_params['payment_rule'] = exp['payment_rule']

                    res_run = run_estimation(log_root_dir=log_root_dir, run=run, behaviroal_models=behavioral_models, gpu=gpu, bo_params_clean=bo_params, experiment=exp) 

                    runs[f'{run}'] = res_run

                treatment_res[f'{treatment}'] = runs

            player_res[f'{n_players}'] = treatment_res

        exp_res[f'{exp_name}'] = player_res 
            

    results = _generate_result_table(exp_res)

    # write to disk
    results.to_csv(f'{path_res}/results_{datetime.datetime.now()}')

    print("Done")
