"""
Main Script for estimating parameters given experimental data.
"""
import os
import datetime
import sys
from numpy import log
import matplotlib.pyplot as plt
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
from bnelearn.parameter_estimators.evaluation_module import CurveEvaluation, StaticsEvaluation


def optimize(n_players: int, gpu: int, data_location: str, iterations: int = 15, 
             initial_samples: int = 5, evaluation_method: str = 'Tobit', full_feedback=True, only_loss=False,
             log_root_dir: str = ".", curveEval: bool = False, id: int = 1):

    # path is user-specific
    if curveEval:
        eval_mode = "Curve"
        maximization = False
    else: 
        eval_mode = "Statistics"
        maximization = True

    if full_feedback:
        log_root_dir = os.path.join(log_root_dir, f'{n_players}', 'full_feedback', f'{eval_mode}', '{}'.format(datetime.datetime.now()))
    else:
        log_root_dir = os.path.join(log_root_dir, f'{n_players}', 'partial_feedback', f'{eval_mode}', '{}'.format(datetime.datetime.now()))

 
    # Logging
    writer = SummaryWriter(log_root_dir)

    if curveEval:
        evaluation_module = CurveEvaluation(n_players=n_players, data_location=data_location, 
                                            full_feedback=full_feedback, method=evaluation_method, writer=writer)
    else:
        evaluation_module = StaticsEvaluation(data_location=data_location, writer=writer)

    risk_bound = [0.1, 1.0]
    regret_beta_bound = [0, 3.0]
    if full_feedback:
        regret_gamma_bound = [0, 3.0]
    else:
        regret_gamma_bound = [0.0, 0.0]

    loss_eta_bound = [1.0, 1.0]
    loss_lambda_bound = [1.1, 3.0]

    if only_loss:
            regret_gamma_bound = [0.0, 0.0]
            regret_beta_bound = [0.0, 0.0]

    param_bounds_risk = {"risk": risk_bound, "regret_beta": regret_beta_bound, "regret_gamma": regret_gamma_bound, 
                         "loss_eta": [0.0, 0.0], "loss_lambda": [0.0, 0.0]}

    param_bounds_loss = {"risk": [1.0, 1.0], "regret_beta": regret_beta_bound, "regret_gamma": regret_gamma_bound, 
                         "loss_eta": loss_eta_bound, "loss_lambda": loss_lambda_bound}

    # Risk-Averse World
    if not only_loss:
        optimizer_risk = Bayesian_Optimizer(initial_samples=initial_samples, param_bounds=param_bounds_risk, n_players=n_players, gpu=gpu, 
                                            log_root_dir=log_root_dir, evaluation_module=evaluation_module, iterations=iterations, maximization=maximization)

        performances, params = optimizer_risk.optimize()

        # plot metrics
        metrics_risk = pd.DataFrame(params)
        metrics_risk.columns = ['Risk', 'Regret_Beta', 'Regret_Gamma', 'Loss_Eta', 'Loss_Lambda']
        metrics_risk['Acc'] = performances
        fig, ax = plt.subplots()
        ax.table(cellText=metrics_risk.values, colLabels=metrics_risk.columns, loc="center")
        writer.add_figure('IMAGES/metrics', fig)

        writer.close()

        os.makedirs(f'{log_root_dir}/res/{n_players}/risk')
        metrics_risk.to_csv(f'{log_root_dir}/res/{n_players}/risk/res.csv')

        

    # Loss-Averse World
    ## Re-init modules
    writer = SummaryWriter(log_root_dir)

    evaluation_module = CurveEvaluation(n_players=n_players, data_location=data_location, 
                                        full_feedback=full_feedback, method=evaluation_method, writer=writer)


    optimizer_loss = Bayesian_Optimizer(initial_samples=initial_samples, param_bounds=param_bounds_loss, n_players=n_players, gpu=gpu, 
                                        log_root_dir=log_root_dir, evaluation_module=evaluation_module, iterations=iterations, maximization=maximization)

    performances, params = optimizer_loss.optimize()

    # plot metrics
    metrics = pd.DataFrame(params)
    metrics.columns = ['Risk', 'Regret_Beta', 'Regret_Gamma', 'Loss_Eta', 'Loss_Lambda']
    metrics['Acc'] = performances
    fig, ax = plt.subplots()
    ax.table(cellText=metrics.values, colLabels=metrics.columns, loc="center")
    writer.add_figure('IMAGES/metrics', fig)

    writer.close()

    os.makedirs(f'{log_root_dir}/res/{n_players}/loss')
    metrics.to_csv(f'{log_root_dir}/res/{n_players}/loss/res.csv')

    # join metrics and return 
    if not only_loss:
        return_dict[id] = metrics.append(metrics_risk, ignore_index=True).assign(n_player=2, eval_mode=eval_mode, full_feedback=full_feedback)
    else:
        return_dict[id] = metrics.assign(n_player=2, eval_mode=eval_mode, full_feedback=full_feedback)

if __name__ == '__main__':

    log_root_dir = os.path.join(os.path.expanduser('~'), 'bnelearn', 'estimations', 'all_pay', '{}'.format(datetime.datetime.now()))

    os.mkdir(log_root_dir)

    # 2 Player Experiments
    ## Full Feedback
    n_players = 2
    gpu = [2, 3, 4]
    data_location_2_full = "/home/ewert/data/AllPay/data_2_full.csv"

    p1 = mp.Process(target=optimize, args=(n_players, gpu[0], data_location_2_full, 15, 8, "Tobit", True, False, log_root_dir, False, 1))

    ## Parial Feedback
    data_location_2_partial = "/home/ewert/data/AllPay/data_2_partial.csv"

    p2 = mp.Process(target=optimize, args=(n_players, gpu[1], data_location_2_partial, 15, 8, "Tobit", False, False, log_root_dir, False, 2))


    # 4 Player Experiments
    n_players = 4
    data_location_4 = "/home/ewert/data/AllPay/data.csv"
    
    p3 = mp.Process(target=optimize, args=(n_players, gpu[2], data_location_4, 15, 8, "Tobit", True, False, log_root_dir, False, 3))

    ### Start processes
    manager = mp.Manager()
    return_dict = manager.dict()

    p1.start()
    p2.start()
    p3.start()

    p1.join()
    p2.join()
    p3.join()

    res = pd.DataFrame()

    for k in return_dict.keys():
        res = res.append(return_dict[k], ignore_index=True)

    # Write overall res
    os.mkdir(f'{log_root_dir}/result/')
    res.to_csv(f'{log_root_dir}/result/result.csv')