# ToDo
# The idea is to create a test which would run all the types of experiments in with very minimalistic settings to check
# that nothing is broken. I am not sure which assertions to make, so strictly speaking this is not yet a test but
# simply a script which should run without runtime errors. Also it should be logging to a separate folder.
# Also parameters are rather meaningless as I don't understand the boundaries for values for each specific experiment

import torch.nn as nn
from bnelearn.experiment import GPUController, LearningConfiguration
from bnelearn.experiment.logger import *
from bnelearn.experiment.multi_unit_experiment import *
from bnelearn.experiment.single_item_experiment import *
from bnelearn.experiment.combinatorial_experiment import *


def set_params_single_item():
    global learner_hyperparams
    global optimizer_hyperparams
    global experiment_params_ls
    global experiment_params
    global optimizer_hyperparams

    learner_hyperparams = {
        'population_size': 128,
        'sigma': 1.,
        'scale_sigma_by_model_size': True
    }

    optimizer_hyperparams = {
        'lr': 3e-3
    }

    experiment_params_ls = ['n_players', 'model_sharing', 'u_lo', 'u_hi', 'valuation_prior', 'common_prior',
                            'payment_rule']
    experiment_params = dict(zip(experiment_params_ls, [None] * len(experiment_params_ls)))

    experiment_params['n_players'] = 3
    experiment_params['model_sharing'] = True
    experiment_params['u_lo'] = [0] * experiment_params['n_players']
    experiment_params['u_hi'] = [1, 1, 1]  # 2]#1,1,2,2]
    experiment_params['payment_rule'] = 'first_price'  # 'first_price'
    experiment_params['risk'] = 1.0
    experiment_params['regret_batch_size'] = 2 ** 8
    experiment_params['regret_grid_size'] = 2 ** 6


def set_params_multi_item():
    global learner_hyperparams
    global optimizer_hyperparams
    global experiment_params_ls
    global experiment_params
    global l_config
    global gpu_config

    learner_hyperparams = {
        'seed': [np.random.randint(1e4) for _ in range(10)],
        'population_size': 64,
        'sigma': 1.,
        'scale_sigma_by_model_size': True,
        'normalize_gradients': False,
        'lr': 0.01,
        'weight_decay': 0.00,  # (float, optional) – weight decay (L2 penalty) (default: 0)
        'momentum': 0.8,
        'pretrain_epoch': 500,
        'pretrain_transform': lambda x: x
    }

    optimizer_hyperparams = {
        'lr': 3e-3
    }

    experiment_params_ls = ['n_players', 'model_sharing', 'u_lo', 'u_hi', 'valuation_prior', 'common_prior',
                            'payment_rule']
    experiment_params = dict(zip(experiment_params_ls, [None] * len(experiment_params_ls)))

    experiment_params['n_players'] = 2
    experiment_params['model_sharing'] = True
    experiment_params['u_lo'] = [1.0] * experiment_params['n_players']
    experiment_params['u_hi'] = [1.4] * experiment_params['n_players']
    experiment_params['payment_rule'] = 'first_price'  # 'first_price'
    experiment_params['risk'] = 1.0
    experiment_params['regret_batch_size'] = 2 ** 8
    experiment_params['regret_grid_size'] = 2 ** 6
    experiment_params['item_interest_limit'] = 2
    experiment_params['efficiency_parameter'] = 0.3

    l_config = LearningConfiguration(learner_hyperparams=learner_hyperparams,
                                     optimizer_type='adam',
                                     optimizer_hyperparams=optimizer_hyperparams,
                                     input_length=1,
                                     hidden_nodes=[5, 5],
                                     hidden_activations=[nn.SELU(), nn.SELU()],
                                     pretrain_iters=100, batch_size=2 ** 10,
                                     eval_batch_size=2 ** 8,
                                     cache_eval_actions=True)



def set_params_FPSBSplitAwardAuction2x2():
    global learner_hyperparams
    global optimizer_hyperparam
    global experiment_params_ls
    global experiment_params
    global l_config
    global gpu_config

    learner_hyperparams = {
        'seed': [np.random.randint(1e4) for _ in range(10)],
        'population_size': 64,
        'sigma': 1.,
        'scale_sigma_by_model_size': True,
        'normalize_gradients': False,
        'lr': 0.01,
        'weight_decay': 0.00,  # (float, optional) – weight decay (L2 penalty) (default: 0)
        'momentum': 0.8,
        'pretrain_epoch': 500,
        'pretrain_transform': FPSBSplitAwardAuction2x2.exp_no_6_transform
    }

    optimizer_hyperparams = {
        'lr': 3e-3
    }

    experiment_params_ls = ['n_players', 'model_sharing', 'u_lo', 'u_hi', 'valuation_prior', 'common_prior',
                            'payment_rule', 'item_interest_limit', 'efficiency_parameter']
    experiment_params = dict(zip(experiment_params_ls, [None] * len(experiment_params_ls)))

    experiment_params['n_players'] = 2
    experiment_params['model_sharing'] = True
    experiment_params['u_lo'] = [1.0] * experiment_params['n_players']
    experiment_params['u_hi'] = [1.4] * experiment_params['n_players']
    experiment_params['payment_rule'] = 'first_price'  # 'first_price'
    experiment_params['risk'] = 1.0
    experiment_params['regret_batch_size'] = 2 ** 8
    experiment_params['regret_grid_size'] = 2 ** 6
    experiment_params['item_interest_limit'] = 2
    experiment_params['efficiency_parameter'] = 0.3

    l_config = LearningConfiguration(learner_hyperparams=learner_hyperparams,
                                     optimizer_type='adam',
                                     optimizer_hyperparams=optimizer_hyperparams,
                                     input_length=1,
                                     hidden_nodes=[5, 5],
                                     hidden_activations=[nn.SELU(), nn.SELU()],
                                     pretrain_iters=300, batch_size=2 ** 10,
                                     eval_batch_size=2 ** 8,
                                     cache_eval_actions=True)

    gpu_config = GPUController(specific_gpu=0)


def run_UniformSymmetricPriorSingleItemExperiment():
    logger = SingleItemAuctionLogger(experiment_params=experiment_params, l_config=l_config)
    experiment = UniformSymmetricPriorSingleItemExperiment(experiment_params, gpu_config=gpu_config, logger=logger,
                                                           l_config=l_config)
    experiment.run(epochs=epochs, n_runs=n_runs)


def run_MultiItemVickreyAuctionExperiment():
    logger = MultiUnitAuctionLogger(experiment_params=experiment_params, l_config=l_config)
    experiment = MultiItemVickreyAuction(experiment_params, gpu_config=gpu_config, logger=logger,
                                         l_config=l_config)
    experiment.run(epochs=epochs, n_runs=n_runs)


def run_MultiItemUniformPriceAuction2x2():
    logger = MultiUnitAuctionLogger(experiment_params=experiment_params, l_config=l_config)
    experiment = MultiItemUniformPriceAuction2x2(experiment_params, gpu_config=gpu_config, logger=logger,
                                                 l_config=l_config)
    experiment.run(epochs=epochs, n_runs=n_runs)


def run_MultiItemUniformPriceAuction2x3limit2():
    logger = MultiUnitAuctionLogger(experiment_params=experiment_params, l_config=l_config)
    experiment = MultiItemUniformPriceAuction2x3limit2(experiment_params, gpu_config=gpu_config, logger=logger,
                                                       l_config=l_config)
    experiment.run(epochs=epochs, n_runs=n_runs)


def run_MultiItemDiscriminatoryAuction2x2():
    logger = MultiUnitAuctionLogger(experiment_params=experiment_params, l_config=l_config)
    experiment = MultiItemDiscriminatoryAuction2x2(experiment_params, gpu_config=gpu_config, logger=logger,
                                                   l_config=l_config)
    experiment.run(epochs=epochs, n_runs=n_runs)


def run_MultiItemDiscriminatoryAuction2x2CMV():
    logger = MultiUnitAuctionLogger(experiment_params=experiment_params, l_config=l_config)
    experiment = MultiItemDiscriminatoryAuction2x2CMV(experiment_params, gpu_config=gpu_config, logger=logger,
                                                      l_config=l_config)
    experiment.run(epochs=epochs, n_runs=n_runs)


def run_FPSBSplitAwardAuction2x2():
    logger = MultiUnitAuctionLogger(experiment_params=experiment_params, l_config=l_config)
    experiment = FPSBSplitAwardAuction2x2(experiment_params, gpu_config=gpu_config, logger=logger,
                                          l_config=l_config)
    experiment.run(epochs=epochs, n_runs=n_runs)


experiment_params_ls = []
experiment_params = {}
learner_hyperparams = {
    'population_size': 128,
    'sigma': 1.,
    'scale_sigma_by_model_size': True
}
optimizer_hyperparams = {
    'lr': 3e-3
}
gpu_config = GPUController(specific_gpu=0)
l_config = LearningConfiguration(learner_hyperparams=learner_hyperparams,
                                 optimizer_type='adam',
                                 optimizer_hyperparams=optimizer_hyperparams,
                                 input_length=1,
                                 hidden_nodes=[5, 5],
                                 hidden_activations=[nn.SELU(), nn.SELU()],
                                 pretrain_iters=100, batch_size=2 ** 4,
                                 eval_batch_size=2 ** 4,
                                 cache_eval_actions=True)

epochs = 10
n_runs = 1

#set_params_single_item()
set_params_multi_item()
#set_params_FPSBSplitAwardAuction2x2()


# run_UniformSymmetricPriorSingleItemExperiment()

run_MultiItemVickreyAuctionExperiment()
#run_MultiItemDiscriminatoryAuction2x2()
#run_MultiItemUniformPriceAuction2x3limit2()
#run_MultiItemUniformPriceAuction2x2()
#run_MultiItemDiscriminatoryAuction2x2CMV()

#run_FPSBSplitAwardAuction2x2()
#Doesn't run yet, get an error
#RuntimeError: expected device cuda:0 and dtype Double but got device cuda:0 and dtype Float
