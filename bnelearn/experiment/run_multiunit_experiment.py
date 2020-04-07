import sys
import os
sys.path.append(os.path.realpath('.'))
import torch
import torch.nn as nn

from bnelearn.experiment.gpu_controller import GPUController
from bnelearn.experiment.learning_configuration import LearningConfiguration
from bnelearn.experiment.logger import MultiUnitAuctionLogger
from bnelearn.experiment.multi_unit_experiment import (
    MultiItemVickreyAuction2x2,
    MultiItemUniformPriceAuction2x2,
    MultiItemUniformPriceAuction2x3limit2,
    MultiItemDiscriminatoryAuction2x2,
    MultiItemDiscriminatoryAuction2x2CMV,
    FPSBSplitAwardAuction2x2
)

gpu_config = GPUController(specific_gpu=7)



settings = [
    {
        'Auction': MultiItemVickreyAuction2x2,
        'model_sharing': True,
        'u_lo': [0] * 2,
        'u_hi': [1] * 2,
    },
    {
        'Auction': MultiItemUniformPriceAuction2x2,
        'model_sharing': True,
        'u_lo': [0] * 2,
        'u_hi': [1] * 2,
    },
    {
        'Auction': MultiItemUniformPriceAuction2x3limit2,
        'model_sharing': True,
        'u_lo': [0] * 3,
        'u_hi': [1] * 3,
    },
    {
        'Auction': MultiItemDiscriminatoryAuction2x2,
        'model_sharing': True,
        'u_lo': [0] * 2,
        'u_hi': [1] * 2,
    },
    {
        'Auction': MultiItemDiscriminatoryAuction2x2CMV,
        'model_sharing': True,
        'u_lo': [0] * 2,
        'u_hi': [1] * 2,
    },
    {
        'Auction': FPSBSplitAwardAuction2x2,
        'model_sharing': True,
        'u_lo': [1] * 2,
        'u_hi': [1.4] * 2,
        'efficiency_parameter': 0.3
    }
]


for experiment_params in settings:

    print('experiment_params\n-----------')
    for k, v in experiment_params.items():
        print('{}: {}'.format(k, v))
    print('-----------\n')

    learner_hyperparams = {
        'population_size': 128,
        'sigma': 1.,
        'scale_sigma_by_model_size': True,
        'pretrain_iters': 300
    }
    optimizer_hyperparams = {
        'lr': 3e-3
    }

    experiment_params['risk'] = 1.0

    # experiment_params['regret_batch_size'] = 2**8
    # experiment_params['regret_grid_size'] = 2**8

    input_length = experiment_params['Auction'].class_experiment_params['n_items']
    hidden_nodes = [5, 5, 5]
    hidden_activations = [nn.SELU(), nn.SELU(), nn.SELU()]

    l_config = LearningConfiguration(
        learner_hyperparams = learner_hyperparams,
        optimizer_type = 'Adam',
        optimizer_hyperparams = optimizer_hyperparams,
        input_length = input_length,
        hidden_nodes = hidden_nodes,
        hidden_activations = hidden_activations,
        pretrain_iters = 500,
        batch_size = 2 ** 17,
        eval_batch_size = 2 ** 22,
        cache_eval_actions = True
    )

    logger = MultiUnitAuctionLogger(experiment_params, l_config)
    experiment = experiment_params['Auction'](
        experiment_params,
        gpu_config = gpu_config,
        logger = logger,
        l_config = l_config
    )
    experiment.run(epochs=2000, n_runs=1)
