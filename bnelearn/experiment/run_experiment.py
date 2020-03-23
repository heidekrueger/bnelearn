import sys
sys.path.append('..')
import torch
import torch.nn as nn

from bnelearn.experiment import GPUController, Logger, LearningConfiguration
from bnelearn.experiment.single_item_experiment import UniformSymmetricPriorSingleItemExperiment, \
    GaussianSymmetricPriorSingleItemExperiment

from bnelearn.experiment.combinatorial_experiment import LLGExperiment

gpu_config = GPUController(specific_gpu=1)
logger = Logger()

learner_hyperparams = {
    'population_size': 128,
    'sigma': 1.,
    'scale_sigma_by_model_size': True
}

optimizer_hyperparams = {
    'lr': 3e-3
}

input_length = 1
hidden_nodes = [5, 5]
hidden_activations = [nn.SELU(), nn.SELU()]

l_config = LearningConfiguration(learner_hyperparams=learner_hyperparams,
                                 optimizer_type='adam',
                                 optimizer_hyperparams=optimizer_hyperparams,
                                 input_length=input_length,
                                 hidden_nodes=hidden_nodes,
                                 hidden_activations=hidden_activations,
                                 pretrain_iters=300, batch_size=2 ** 18,
                                 eval_batch_size=2 ** 10,
                                 cache_eval_actions=True)

experiment1 = LLGExperiment(mechanism_type='vcg', gpu_config=gpu_config, logger=logger,
                                                       l_config=l_config)
#experiment2 = GaussianSymmetricPriorSingleItemExperiment(gpu_config=gpu_config, logger=logger,
#                                                       mechanism_type='first_price', l_config=l_config, risk=1.0)

experiment1.run(epochs=3000, n_runs=3)
#experiment2.run(epochs=100, n_runs=1)