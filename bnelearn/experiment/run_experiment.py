import warnings

import torch
import torch.nn as nn

from bnelearn.experiment import GPUController, Logger, LearningConfiguration
from bnelearn.experiment.single_item_experiment import UniformSymmetricPriorSingleItemExperiment

gpu_config = GPUController()
logger = Logger()

learner_hyperparams = {
    'population_size': 64,
    'sigma': 1.,
    'scale_sigma_by_model_size': True
}

optimizer_hyperparams = {
    'lr': 3e-3
}

input_length = 1
hidden_nodes = [5, 5, 5]
hidden_activations = [nn.SELU(), nn.SELU(), nn.SELU()]

l_config = LearningConfiguration(learner_hyperparams=learner_hyperparams,
                                 optimizer_type=torch.optim.Adam,
                                 optimizer_hyperparams=optimizer_hyperparams,
                                 input_length=input_length,
                                 hidden_nodes=hidden_nodes,
                                 hidden_activations=hidden_activations,
                                 pretrain_iters=500, batch_size=2 ** 13,
                                 eval_batch_size=2 * 12,
                                 cache_eval_actions=True)

experiment = UniformSymmetricPriorSingleItemExperiment(gpu_config=gpu_config, logger=logger,
                                                       mechanism_type='first_price', l_config=l_config, risk=1.0)

experiment.run(epochs=50, n_runs=1)
