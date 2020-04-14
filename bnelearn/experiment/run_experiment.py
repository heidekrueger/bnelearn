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
import warnings
gpu_config = GPUController(specific_gpu=1)

learner_hyperparams = {
    'population_size': 128,
    'sigma': 1.,
    'scale_sigma_by_model_size': True
}

optimizer_hyperparams = {
    'lr': 3e-3
}

# experiment_params = {
#         'model_sharing': False,
#         'u_lo': 0,
#         'u_hi': 1,
#         'payment_rule': 'first_price',
#         'risk': 1.0,
#         'regret_batch_size': 2**8,
#         'regret_grid_size': 2**8
#     }
experiment_params = {
        'model_sharing': True,
        'valuation_mean': 15,
        'valuation_std': 5,
        'payment_rule': 'first_price',
        'risk': 1.0,
        'regret_batch_size': 2**8,
        'regret_grid_size': 2**8
    }

input_length = 1
hidden_nodes = [5, 5, 5]
hidden_activations = [nn.SELU(), nn.SELU(), nn.SELU()]

l_config = LearningConfiguration(learner_hyperparams=learner_hyperparams,
                                 optimizer_type='adam',
                                 optimizer_hyperparams=optimizer_hyperparams,
                                 input_length=input_length,
                                 hidden_nodes=hidden_nodes,
                                 hidden_activations=hidden_activations,
                                 pretrain_iters=300, batch_size=2 ** 18,
                                 eval_batch_size=2 ** 15,
                                 cache_eval_actions=True)


#warnings.simplefilter("ignore")
for i in [2]:
    experiment_params['n_players'] = i
    experiment = GaussianSymmetricPriorSingleItemExperiment(experiment_params, gpu_config=gpu_config, l_config=l_config)
    experiment.run(epochs=101, n_runs=2)
                            
