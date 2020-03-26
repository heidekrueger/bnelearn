# ToDo
# The idea is to create a test which would run all the types of experiments in with very minimalistic settings to check
# that nothing is broken. I am not sure which assertions to make, so strictly speaking this is not yet a test but
# simply a script which should run without runtime errors

import torch.nn as nn
from bnelearn.experiment import GPUController, LearningConfiguration
from bnelearn.experiment.logger import *
from bnelearn.experiment.multi_unit_experiment import *
from bnelearn.experiment.single_item_experiment import *
from bnelearn.experiment.combinatorial_experiment import *

gpu_config = GPUController(specific_gpu=0)

learner_hyperparams = {
    'population_size': 128,
    'sigma': 1.,
    'scale_sigma_by_model_size': True
}

optimizer_hyperparams = {
    'lr': 3e-3
}

experiment_params_ls = ['n_players', 'model_sharing', 'u_lo', 'u_hi', 'valuation_prior', 'common_prior','payment_rule']
experiment_params = dict(zip(experiment_params_ls, [None]*len(experiment_params_ls)))

experiment_params['n_players'] = 3
experiment_params['model_sharing'] = True
experiment_params['u_lo'] = [0] * experiment_params['n_players']
experiment_params['u_hi'] = [1,1,1]#2]#1,1,2,2]
experiment_params['payment_rule'] = 'first_price'#'first_price'
experiment_params['risk'] = 1.0
experiment_params['regret_batch_size'] = 2**8
experiment_params['regret_grid_size'] = 2**6

input_length = 1
hidden_nodes = [5, 5]
hidden_activations = [nn.SELU(), nn.SELU()]

l_config = LearningConfiguration(learner_hyperparams=learner_hyperparams,
                                 optimizer_type='adam',
                                 optimizer_hyperparams=optimizer_hyperparams,
                                 input_length=input_length,
                                 hidden_nodes=hidden_nodes,
                                 hidden_activations=hidden_activations,
                                 pretrain_iters=300, batch_size=2 ** 14,
                                 eval_batch_size=2 ** 10,
                                 cache_eval_actions=True)

logger = SingleItemAuctionLogger(experiment_params=experiment_params, l_config=l_config)
experiment1 = UniformSymmetricPriorSingleItemExperiment(experiment_params, gpu_config=gpu_config, logger=logger,
                                                       l_config=l_config)

experiment1.run(epochs=10, n_runs=1)
