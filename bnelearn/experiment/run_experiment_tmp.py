import sys
sys.path.append('..')
import torch
import torch.nn as nn

from bnelearn.experiment.gpu_controller import GPUController
from bnelearn.experiment.learning_configuration import LearningConfiguration
from bnelearn.experiment.logger import LLGAuctionLogger, LLLLGGAuctionLogger, SingleItemAuctionLogger
from bnelearn.experiment.single_item_experiment import UniformSymmetricPriorSingleItemExperiment, \
    GaussianSymmetricPriorSingleItemExperiment

from bnelearn.experiment.combinatorial_experiment import LLGExperiment, LLLLGGExperiment

gpu_config = GPUController(specific_gpu=1)

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
experiment_params['u_hi'] = [1,1,1]
#experiment_params['u_hi'] = [1,1,1,1,2,2]
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

logger = SingleItemAuctionLogger(experiment_params, l_config)
experiment1 = GaussianSymmetricPriorSingleItemExperiment(experiment_params, gpu_config=gpu_config, logger=logger,
                                                       l_config=l_config)
# experiment2 = UniformSymmetricPriorSingleItemExperiment(2, gpu_config=gpu_config, logger=logger,
                                                    #    mechanism_type='first_price', l_config=l_config, risk=1.0)

experiment1.run(epochs=10000, n_runs=2)
#experiment2.run(epochs=100, n_runs=1)
