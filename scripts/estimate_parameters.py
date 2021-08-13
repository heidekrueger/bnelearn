"""
Main Script for estimating parameters given experimental data.
"""
import os
import sys
from numpy import log

import torch
import torch.nn as nn

# put bnelearn imports after this.
# pylint: disable=wrong-import-position
sys.path.append(os.path.realpath('.'))
sys.path.append(os.path.join(os.path.expanduser('~'), 'bnelearn'))

from bnelearn.parameter_estimators.optimizers import Bayesian_Optimizer  
from bnelearn.parameter_estimators.evaluation_module import CurveEvaluation


if __name__ == '__main__':

    # path is user-specific
    log_root_dir = os.path.join(os.path.expanduser('~'), 'bnelearn', 'estimations', 'all_pay')

    initial_samples = 2
    n_players = 2
    gpu = 4
    data_location = "/home/ewert/data/AllPay/data.csv"
    evaluation_method = "Tobit"
    evaluation_module = CurveEvaluation(data_location, evaluation_method)

    risk_bound = [0.1, 1.0]
    regret_beta_bound = [0.1, 0.5]
    regret_gamma_bound = [0.1, 0.5]
    loss_eta_bound = [0.0, 0.0]
    loss_lambda_bound = [0.0, 0.0]

    param_bounds_risk = {"risk": risk_bound, "regret_beta": regret_beta_bound, "regret_gamma": regret_gamma_bound, 
                         "loss_eta": [0.0, 0.0], "loss_lambda": [0.0, 0.0]}

    param_bounds_loss = {"risk": None, "regret_beta": regret_beta_bound, "regret_gamma": regret_gamma_bound, 
                         "loss_eta": loss_eta_bound, "loss_lambda": loss_lambda_bound}

    optimizer_risk = Bayesian_Optimizer(initial_samples=initial_samples, param_bounds=param_bounds_risk, n_players=n_players, gpu=gpu, 
                                        log_root_dir=log_root_dir, evaluation_module=evaluation_module)
    optimizer_risk.optimize()
