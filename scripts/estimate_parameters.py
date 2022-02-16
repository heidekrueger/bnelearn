"""
Main Script for estimating parameters given experimental data.
"""

from multiprocessing import get_context

import os
import datetime
import sys
import pandas as pd

import multiprocessing as mp
from torch.utils.tensorboard import SummaryWriter


import torch
import torch.nn as nn

# put bnelearn imports after this.
# pylint: disable=wrong-import-position
sys.path.append(os.path.realpath('.'))
sys.path.append(os.path.join(os.path.expanduser('~'), 'bnelearn'))

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

def optimize(n_players: int, gpu: int, data_location: str, iterations: int = 15, 
             initial_samples: int = 5, evaluation_method: str = 'Tobit', full_feedback=True, closed_only=False,
             log_root_dir: str = ".", curveEval: bool = False, run_id: int = 1, n_id: int = 1):

    # path is user-specific
    if curveEval:
        eval_mode = "Curve"
        maximization = False
    else: 
        eval_mode = "Statistics"
        maximization = True

    if full_feedback:
        log_root_dir = os.path.join(log_root_dir, f'{run_id}', f'{n_players}', 'full_feedback', f'{eval_mode}', '{}'.format(datetime.datetime.now()))
    else:
        log_root_dir = os.path.join(log_root_dir, f'{run_id}', f'{n_players}', 'partial_feedback', f'{eval_mode}', '{}'.format(datetime.datetime.now()))

 
    risk_path = f'{log_root_dir}/risk/'
    crra_path = f'{log_root_dir}/crra/'
    regret_path = f'{log_root_dir}/regret/'
    
    # Run experiments
    ## Start processes

    manager = mp.Manager()
    global results
    results = manager.list()

    with get_context("spawn").Pool() as pool:

        # Definition of parameter bounds
        param_bounds_risk = {"risk": [0.01, 0.99], "regret_beta": [0.0, 0.0], "regret_gamma": [0.0, 0.0]}
        if full_feedback:
            param_bounds_regret = {"risk": [1.0, 1.0], "regret_beta": [0.0, 5.0], "regret_gamma": [0.0, 5.0]}
        else:
            param_bounds_regret = {"risk": [1.0, 1.0], "regret_beta": [0.0, 5.0], "regret_gamma": [0.0, 0.0]}

        # if not closed_only:

        #     # Risk
        #     res_risk = pool.apply_async(run_optim, (param_bounds_risk, "Risk", run_id, full_feedback, risk_path, n_players, data_location, 
        #                         evaluation_method, initial_samples, gpu[0], iterations, maximization, "default", curveEval,
        #                         n_id, 0, closed_only))

        #     # CRRA
        #     res_crra = pool.apply_async(run_optim, (param_bounds_risk, "CRRA", run_id, full_feedback, crra_path, n_players, data_location, 
        #                         evaluation_method, initial_samples, gpu[1], iterations, maximization, "CRRA", curveEval,
        #                         n_id, 1, closed_only))

        # Regret                 
        res_regret = pool.apply_async(run_optim, (param_bounds_regret, "Regret", run_id, full_feedback, regret_path, n_players, data_location, 
                               evaluation_method, initial_samples, gpu[2], iterations, maximization, "default", curveEval,
                               n_id, 2, closed_only))

        pool.close()
        pool.join()

    # if not closed_only:
    #     results = [res_risk.get()[0], res_crra.get()[0], res_regret.get()[0]]
    #     estimations = [res_risk.get()[1], res_crra.get()[1], res_regret.get()[1]]
    # else:
    #     results = [res_regret.get()[0]]
    #     estimations = [res_regret.get()[1]]

    results = [res_regret.get()[0]]
    estimations = [res_regret.get()[1]]

    return results, estimations


def run_estimation(log_root_dir, i, closed_only):

    run_id = i * 10

    # 2 Player Experiments
    ## Full Feedback
    n_players = 2
    gpu = [1, 1, 1]
    data_location_2_full = "/home/ewert/data/AllPay/data_2_full.csv"

    res = []
    estimations = []

    res_2_full, estimations_2_full = optimize(n_players, gpu, data_location_2_full, 12, 5, "Tobit", True, closed_only, log_root_dir, True, run_id, 0)

    for k, r in enumerate(res_2_full):
        res.append(r)
        estimations.append(estimations_2_full[k])
    
    ## Parial Feedback
    # data_location_2_partial = "/home/ewert/data/AllPay/data_2_partial.csv"

    # res_2_partial, estimations_2_partial = optimize(n_players, gpu, data_location_2_partial, 12, 5, "Tobit", False, closed_only, log_root_dir, False, run_id, 1)

    # for k, r in enumerate(res_2_partial):
    #     res.append(r)
    #     estimations.append(estimations_2_partial[k])

    # # 4 Player Experiments
    # n_players = 4
    # data_location_4 = "/home/ewert/data/AllPay/data.csv"
    
    # res_4, estimations_4 = optimize(n_players, gpu, data_location_4, 12, 5, "Tobit", True, closed_only, log_root_dir, False, run_id, 2)

    # for k, r in enumerate(res_4):
    #     res.append(r)
    #     estimations.append(estimations_4[k])

    return res, estimations


if __name__ == '__main__':

    log_root_dir = os.path.join(os.path.expanduser('~'), 'bnelearn', 'estimations', 'all_pay', '{}'.format(datetime.datetime.now()))
    path_res = f'{log_root_dir}/results/'

    os.mkdir(log_root_dir)
    os.mkdir(path_res)

    res = []
    est = []

    closed_only = False

    for i in range(1):
        res_run, est_run = run_estimation(log_root_dir, i, closed_only)
        res.append(res_run)
        est.append(est_run)
        if i % 10 == 0 or not closed_only:
            os.mkdir(f'{path_res}/{i}')
            # pre-save
            pre = pd.DataFrame()
            pre_est = pd.DataFrame()
            for j in range(len(res_run)):
                pre = pre.append(res_run[j], ignore_index=True)
                pre_est = pre_est.append(est_run[j], ignore_index=True)

            pre_est.to_csv(f'{path_res}/{i}/estimations.csv')
            pre.to_csv(f'{path_res}/{i}/results.csv')


    #res.append(run_estimation(log_root_dir, 0))

    final_table = pd.DataFrame()
    final_estimations = pd.DataFrame()

    for i in range(len(res)):
        final_table = final_table.append(res[i], ignore_index=True)
        final_estimations = final_estimations.append(est[i], ignore_index=True)

    final_table.to_csv(f'{path_res}/results.csv')
    final_estimations.to_csv(f'{path_res}/estimations.csv')

    print("Done")

