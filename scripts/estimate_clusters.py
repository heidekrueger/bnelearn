"""
Main Script for estimating parameters given experimental data.
"""
import os
import sys
import datetime
import pandas as pd
import numpy as np
from torch.utils.tensorboard import SummaryWriter


# put bnelearn imports after this.
# pylint: disable=wrong-import-position
sys.path.append(os.path.realpath('.'))
sys.path.append(os.path.join(os.path.expanduser('~'), 'bnelearn'))

from bnelearn.parameter_estimators.cluster_optimizer import Cluster_Optimizer  
from bnelearn.parameter_estimators.optimizers import MultiObjectiveBayesianOptimizer
from bnelearn.parameter_estimators.evaluation_module import StaticsEvaluation

def run_cluster_experiment(data_location: str, n_players: int, full_feedback: bool = True, only_loss=False):
    

    ## Split data into two clusters: male vs female
    file = pd.read_csv(data_location)

    log_root_dir = os.path.join(os.path.expanduser('~'), 'bnelearn', 'estimations', 'all_pay', 'risk', "{}".format(datetime.datetime.now()))

    ###########################################################################################################################################
    ## TODO - Remove me at some point in time, since I am just here because we cannot calculate the cluster center
    file = file.assign(cluster=lambda df: df["male"])
    clusters = np.unique(file["cluster"])
    cluster_sizes = [len(file[file["cluster"] == k]["cluster"])/len(file) for k in range(len(clusters))]

    clusters_sorted = [x for _, x in sorted(zip(cluster_sizes, clusters))]

    ###########################################################################################################################################

    # Logging
    writer = SummaryWriter(log_root_dir)
    evaluation_module = StaticsEvaluation(data_location=data_location, writer=writer)

    if n_players == len(np.unique(file["cluster"])):
        # single scenario: each player represents a cluster
        model_sharing = list(range(n_players))
    elif n_players > len(np.unique(file["cluster"])):
        model_sharing = clusters_sorted
        index = len(clusters_sorted) - 1
        while len(model_sharing) < n_players:
            model_sharing.append(clusters[index])

            index = index - 1

            if index < 0:
                index = len(clusters_sorted) - 1

    else:
        # TODO: Create scenarios
        raise NotImplementedError

    
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

    optimizer = MultiObjectiveBayesianOptimizer(param_bounds=param_bounds_risk, num_bidder_types=2, model_sharing=model_sharing, eval_module=evaluation_module)
    optimizer.optimize()

if __name__ == '__main__':

    # 4 Player Experiment

    ## Define parameters
    data_location = "/home/ewert/data/AllPay/data.csv"


    #optimizer = Cluster_Optimizer(data_location=data_location)
    #optimizer.get_clusters(k=3)

    run_cluster_experiment(data_location=data_location, n_players=4, full_feedback=True, only_loss=False)

    