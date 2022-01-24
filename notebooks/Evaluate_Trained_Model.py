# %%
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

# set up matplotlib
is_ipython = 'inline' in plt.get_backend()
if is_ipython:
    from IPython import display
plt.rcParams['figure.figsize'] = [10, 7]

device = 'cuda:3'

# %%

# path of experiment and models to be loaded
path = f'/home/kohring/bnelearn/experiments/debug/nearest_vcg/6p/2022-01-14 Fri 09.23'
model_path = f'/home/kohring/bnelearn/experiments/debug/nearest_vcg/6p/2022-01-14 Fri 09.23/00 09:23:53 0/models/'
## cannot write in same place if different user -- define temporary output dir (needed in experiment.run())
user = 'heidekrueger'
write_path = f'/home/{user}/bnelearn/experiments/debug/nearest_vcg/6p/temp'

experiment_config = ConfigurationManager.load_experiment_config(path)
experiment_class = ConfigurationManager \
    .get_class_by_experiment_type(experiment_config.experiment_class)

experiment = experiment_class(experiment_config)

experiment.running.n_runs = 1
experiment.running.seeds = [0]
experiment.running.n_epochs = -1
experiment.learning.pretrain_iters = 0
experiment.learning.optimizer = ConfigurationManager._set_optimizer(experiment.learning.optimizer_type)
experiment.learning.scheduler = ConfigurationManager._set_scheduler(experiment.learning.scheduler_type)
experiment.config.hardware.specific_gpu = device
experiment.hardware.device = device
experiment.sampler.default_device = device
experiment.epoch = -1
experiment.experiment_log_dir = write_path


experiment.run()
experiment.env.mechanism.device = device

# models have to be set after `run()`

model_paths = [f'{model_path}model_{i}.pt' for i in [0, 4]]
models = [NeuralNetStrategy.load(mp, device=device).to(device) for mp in model_paths]
experiment.models = models
for i, b in enumerate(experiment.bidders):
    b.strategy = models[experiment._bidder2model[i]]
experiment.env.agents = experiment.bidders
experiment.env.draw_valuations()

experiment._plot_current_strategies()


# %%
ex_ante_util_loss, ex_interim_max_util_loss, estimated_relative_ex_ante_util_loss = \
    experiment._calculate_metrics_util_loss(
        create_plot_output=True, epoch=-1,
        batch_size=2**7, grid_size=2**8, opponent_batch_size=2**7
    )

print('\n--- REEVALUATED PERFORMANCE ---')
print('ex_ante_util_loss:', ex_ante_util_loss)
print('ex_interim_max_util_loss:', ex_interim_max_util_loss)
print('estimated_relative_ex_ante_util_loss:', estimated_relative_ex_ante_util_loss)
print('\n-------------------------------')

# %%



