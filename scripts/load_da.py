import os
import sys
root_path = os.path.join(os.path.expanduser('~'), 'bnelearn')
if root_path not in sys.path:
    sys.path.append(root_path)

import time

import torch
import torch.nn as nn
import torch.nn.utils as ut

from bnelearn.strategy import NeuralNetStrategy
from bnelearn.experiment.configuration_manager import ConfigurationManager

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# set up matplotlib
is_ipython = 'inline' in plt.get_backend()
if is_ipython:
    from IPython import display
plt.rcParams['figure.figsize'] = [10, 7]

device = 'cuda:1'
save_path = root_path + '/experiments/eff'

def childs(path):
    return next(os.walk(path))[1]

columns = ['price', 'risk', 'mean', 'std']
df = pd.DataFrame(columns=columns)

# path of experiment and models to be loaded
path = '/home/pieroth/bnelearn/experiments/bargaining_paper_results/exp-9_experiment/double_auction/single_item'
pricing_rules = childs(path)

for pricing_rule in pricing_rules:
    path_sub_post = '/0.5/uniform/symmetric' if pricing_rule == 'k_price' else '/uniform/symmetric'
    path_sub0 = path + '/' + pricing_rule + path_sub_post
    for risk in [i/10. for i in range(1, 11)]:
        risk_str = f'risk_{risk}'
        path_sub1 = path_sub0 + '/' + risk_str + '/1b1s/'
        path_sub1 += childs(path_sub1)[0] + '/'

        experiment_config = ConfigurationManager.load_experiment_config(path_sub1)
        experiment_class = ConfigurationManager \
            .get_class_by_experiment_type(experiment_config.experiment_class)

        experiment_config.logging.eval_batch_size = 2
        experiment_config.hardware.device = device

        experiment = experiment_class(experiment_config)
        
        # TODO: Some inits have to be performed: Check!!!
        experiment.running.n_runs = 1
        experiment.running.seeds = [0]
        experiment.running.n_epochs = -1
        experiment.learning.pretrain_iters = 0
        experiment.learning.optimizer = ConfigurationManager._set_optimizer(experiment.learning.optimizer_type)
        experiment.config.hardware.specific_gpu = device
        experiment.hardware.device = device
        experiment.sampler.default_device = device
        experiment.epoch = -1
        experiment.learning.batch_size = 2**18
        experiment.experiment_log_dir = save_path

        try:
            experiment.run()
        except:
            pass
        experiment.env.mechanism.device = device

        runs = childs(path_sub1)
        e = np.zeros(len(runs))
        for j, run in enumerate(runs):
            path_sub2 = path_sub1 + '/' + run + '/models/'

            # models have to be set after `run()`
            model_paths = [f'{path_sub2}model_{i}.pt' for i in ['buyer', 'seller']]
            models = [NeuralNetStrategy.load(mp, device=device).to(device) for mp in model_paths]
            experiment.models = models
            for i, b in enumerate(experiment.bidders):
                b.strategy = models[experiment._bidder2model[i]]
            experiment.env.agents = experiment.bidders
            experiment.env.draw_valuations()

            e[j] = experiment.env.get_da_strategy_metrics()[-1]

        df = pd.concat([df, pd.DataFrame([[pricing_rule, risk, e.mean(), e.std()]], columns=columns)])
        df.to_csv(save_path + '/df.csv')
