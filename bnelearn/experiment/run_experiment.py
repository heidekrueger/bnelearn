import warnings

import torch
import torch.nn as nn

from bnelearn.experiment import GPUController, Logger, Plotter
from bnelearn.experiment.SingleItemExperiment import UniformSymmetricPriorSingleItemExperiment

n_runs = 1
epochs = 200
seeds = list(range(n_runs))

gpu_config = GPUController()
logger = Logger()
plotter = Plotter()

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


def run(seed, run_comment, epochs):
    if seed is not None:
        torch.random.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    experiment = UniformSymmetricPriorSingleItemExperiment(n_players=2, gpu_config=gpu_config, logger=logger,
                                                           mechanism_type='first_price', plotter=plotter,
                                                           learner_hyperparams=learner_hyperparams,
                                                           optimizer_type=torch.optim.Adam,
                                                           optimizer_hyperparams=optimizer_hyperparams,
                                                           input_length=input_length,
                                                           hidden_nodes=hidden_nodes,
                                                           hidden_activations=hidden_activations,
                                                           pretrain_iters=500, batch_size=2 ** 13,
                                                           eval_batch_size=2 * 12,
                                                           cache_eval_actions=True, risk=1.0, n_runs=1)

    experiment.run(epochs, run_comment)

    del experiment
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    if torch.cuda.memory_allocated() > 0:
        warnings.warn('Theres a memory leak')


for seed in seeds:
    print('Running experiment {}'.format(seed))
    run(seed, str(seed), epochs)
