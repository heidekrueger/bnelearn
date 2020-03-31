"""
Author Nils Kohring
Date   Oct 2019
Desc   Testing of learning in multi unit auction formats
"""


# ## Imports
import sys
import os
import time
import random
from functools import partial
from copy import deepcopy

import torch
import torch.nn as nn
# import torch.nn.utils as ut
from torch.optim.lr_scheduler import ReduceLROnPlateau
# from torch.optim.optimizer import Optimizer, required
#from utils_nils import *
from scripts_work_in_progress.utils_nils import *

sys.path.append(os.path.realpath('.'))
from bnelearn.strategy import NeuralNetStrategy, ClosureStrategy
from bnelearn.bidder import Bidder, ReverseBidder
from bnelearn.mechanism import (MultiItemDiscriminatoryAuction,
                                MultiItemUniformPriceAuction,
                                MultiItemVickreyAuction,
                                FPSBSplitAwardAuction)
from bnelearn.learner import ESPGLearner
from bnelearn.environment import AuctionEnvironment

from torch.utils.tensorboard import SummaryWriter
import numpy as np
import matplotlib.pyplot as plt
from itertools import product


"""
Notes & Todo
    - random or adversary disturbances to policy output
"""

# ## Settings

## Experiment setup
param_dict = dict()
param_dict["exp_no"] = 6

if param_dict["exp_no"] == 0:
    mechanism = MultiItemVickreyAuction(cuda=True)
    param_dict["n_players"] = 2
    param_dict["n_items"] = 2
    param_dict["u_lo"] = 0
    param_dict["u_hi"] = 1
    param_dict["BNE1"] = "Truthful"
    param_dict["BNE2"] = "Truthful"
    def strat_to_bidder(strategy, batch_size, player_position):
        """
        Standard strat_to_bidder method.
        """
        return Bidder.uniform(
            lower = param_dict["u_lo"], upper = param_dict["u_hi"],
            strategy = strategy,
            n_items = param_dict["n_items"],
            descending_valuations = True,
            constant_marginal_values = True,
            player_position = player_position,
            batch_size = batch_size
        )

elif param_dict["exp_no"] == 1:
    mechanism = MultiItemUniformPriceAuction(cuda=True)
    param_dict["n_players"] = 2
    param_dict["n_items"] = 2
    param_dict["u_lo"] = 0
    param_dict["u_hi"] = 1
    def exp_no_1_transform(input_tensor):
        output_tensor = torch.clone(input_tensor)
        output_tensor[:,1] = 0
        return output_tensor
    param_dict["BNE1"] = "BNE1"
    param_dict["BNE2"] = "Truthful"
    def strat_to_bidder(strategy, batch_size, player_position):
        """
        Standard strat_to_bidder method.
        """
        return Bidder.uniform(
            lower = param_dict["u_lo"], upper = param_dict["u_hi"],
            strategy = strategy,
            n_items = param_dict["n_items"],
            descending_valuations = True,
            player_position = player_position,
            batch_size = batch_size
        )

elif param_dict["exp_no"] == 2:
    mechanism = MultiItemUniformPriceAuction(cuda=True)
    param_dict["n_players"] = 2
    param_dict["n_items"] = 3
    param_dict["u_lo"] = 0
    param_dict["u_hi"] = 1
    param_dict["item_interest_limit"] = 2
    param_dict["BNE1"] = "BNE1"
    param_dict["BNE2"] = "Truthful"
    def strat_to_bidder(strategy, batch_size, player_position):
        """
        Standard strat_to_bidder method.
        """
        return Bidder.uniform(
            lower = param_dict["u_lo"], upper = param_dict["u_hi"],
            strategy = strategy,
            n_items = param_dict["n_items"],
            item_interest_limit = param_dict["item_interest_limit"],
            descending_valuations = True,
            player_position = player_position,
            batch_size = batch_size
        )

elif param_dict["exp_no"] == 3:
    """ ´See Large Multi-Unit Auctions with a Large Bidder´ by Brian Baisa
        and Justin Burkett.
    """
    raise NotImplementedError('Experiment discarded.')

elif param_dict["exp_no"] == 4:
    mechanism = MultiItemDiscriminatoryAuction(cuda=True)
    param_dict["n_players"] = 2
    param_dict["n_items"] = 2
    param_dict["u_lo"] = 0
    param_dict["u_hi"] = 1
    param_dict["BNE1"] = "BNE1"
    param_dict["BNE2"] = "Truthful"
    def strat_to_bidder(strategy, batch_size, player_position):
        """
        Standard strat_to_bidder method.
        """
        return Bidder.uniform(
            lower = param_dict["u_lo"], upper = param_dict["u_hi"],
            strategy = strategy,
            n_items = param_dict["n_items"],
            descending_valuations = True,
            player_position = player_position,
            batch_size = batch_size
        )

elif param_dict["exp_no"] == 5:
    mechanism = MultiItemDiscriminatoryAuction(cuda=True)
    param_dict["n_players"] = 2
    param_dict["n_items"] = 2
    param_dict["u_lo"] = 0
    param_dict["u_hi"] = 1
    param_dict["constant_marginal_values"] = True
    param_dict["BNE1"] = "BNE1"
    param_dict["BNE2"] = "Truthful"
    def strat_to_bidder(strategy, batch_size, player_position):
        """
        Standard strat_to_bidder method.
        """
        return Bidder.uniform(
            lower = param_dict["u_lo"], upper = param_dict["u_hi"],
            strategy = strategy,
            n_items = param_dict["n_items"],
            descending_valuations = True,
            constant_marginal_values = True,
            player_position = player_position,
            batch_size = batch_size
        )

elif param_dict["exp_no"] == 6:
    mechanism, param_dict, split_award_dict = create_splitaward_setting()
    def exp_no_6_transform(input_tensor):
        """
        Transformation for Split-Award auciton.
        """
        temp = input_tensor.clone().detach()
        if input_tensor.shape[1] == 1:
            output_tensor = torch.cat((
                temp,
                param_dict["efficiency_parameter"] * temp
            ), 1)
        else:
            output_tensor = temp
        return output_tensor
    def strat_to_bidder(strategy, batch_size, player_position):
        """
        Standard strat_to_bidder method.
        """
        return ReverseBidder.uniform(
            efficiency_parameter = param_dict["efficiency_parameter"],
            lower = param_dict["u_lo"], upper = param_dict["u_hi"],
            strategy = strategy,
            n_items = param_dict["n_items"],
            descending_valuations = param_dict["exp_no"] != 6,
            player_position = player_position,
            batch_size = batch_size
        )

# Log in folder
log_root = os.path.abspath('/home/kohring/bnelearn/experiments')
save_figure_data_to_disk = False
save_figure_to_disk = False

auction_type_str = str(type(mechanism))
auction_type_str = str(auction_type_str[len(auction_type_str) \
                       - auction_type_str[::-1].find('.'):-2])
log_name = auction_type_str + '_' + str(param_dict["n_players"]) \
        + 'players_' + str(param_dict["n_items"]) + 'items'



## Environment settings
batch_size = 2**18
# regret_batch_size = 2**6
epoch = 2000
model_sharing = True
epo_n = 2 # for ensure positive output of initialization
plot_epoch = 10
specific_gpu = 7
logging = True


# Use specific cuda gpu if desired (i.e. for running multiple experiments in parallel)
cuda = torch.cuda.is_available()
device = 'cuda' if cuda else 'cpu'
if cuda and specific_gpu:
    torch.cuda.set_device(specific_gpu)
print('device:', device, end=' ')
if cuda:
    print(torch.cuda.current_device())

# strategy model architecture
param_dict["input_length"] = param_dict["n_items"]

model_dict = {
    "hidden_nodes": [5, 5, 5],
    "hidden_activations": [nn.SELU(), nn.SELU(), nn.SELU()]
}


hyperparams = {
    'seed':                      [np.random.randint(1e4) for _ in range(10)],
    'population_size':           [64],
    'sigma':                     [1.],
    'scale_sigma_by_model_size': [True],
    'normalize_gradients':       [False],
    'lr':                        [0.01],
    'weight_decay':              [0.00], # (float, optional) – weight decay (L2 penalty) (default: 0)
    'momentum':                  [0.8],
    'pretrain_epoch':            [500],
    'pretrain_transform':        [(lambda x: x) if param_dict["exp_no"] != 6 else exp_no_6_transform]
}                                       # lambda x: x,
                                        # exp_no_1_transform
                                        # exp_no_6_transform



for vals in product(*hyperparams.values()):
    seed, population_size, sigma, scale_sigma_by_model_size, normalize_gradients, \
    lr, weight_decay, momentum, pretrain_epoch, pretrain_transform = vals

    print('\nhyperparams\n-----------')
    for k in hyperparams.keys():
        print('{}: {}'.format(k, eval(k)))
    print('-----------\n')

    learner_hyperparams = {
        'population_size': population_size,
        'sigma': sigma,
        'scale_sigma_by_model_size': scale_sigma_by_model_size,
        'normalize_gradients': normalize_gradients
    }

    # Set random seeds
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)

    # Setting up optimizer
    optimizer_type = torch.optim.Adam
    optimizer_hyperparams = {
        'lr': lr,
        'weight_decay': weight_decay,
        # 'momentum': momentum,
        # 'amsgrad': True # for Adam
    }

    # ## Setting up the Environment

    # Initialize models
    ensure_positive_output = torch.zeros(epo_n, param_dict["input_length"]) \
        .uniform_(param_dict["u_lo"], param_dict["u_hi"])\
        .sort(dim=1, descending=True)[0]
    n_models = 1 if model_sharing else param_dict["n_players"]
    models = [
        NeuralNetStrategy(
            param_dict["input_length"],
            hidden_nodes = model_dict["hidden_nodes"],
            hidden_activations = model_dict["hidden_activations"],
            ensure_positive_output = ensure_positive_output,
            output_length = param_dict["n_items"]
        ).to(device)
        for _ in range(n_models)
    ]

    # Pretrain
    pretrain_points = round(100 ** (1 / param_dict["input_length"]))
    # pretrain_valuations = multi_unit_valuations(
    #     device = device,
    #     bounds = [param_dict["u_lo"], param_dict["u_hi"]],
    #     dim = param_dict["n_items"],
    #     batch_size = pretrain_points,
    #     selection = 'random' if param_dict["exp_no"] != 6 else split_award_dict
    # )
    pretrain_valuations = strat_to_bidder(lambda x: x, batch_size, 0).draw_valuations_()[:pretrain_points,:]

    n_parameters = list()
    for model in models:
        n_parameters.append(sum([p.numel() for p in model.parameters()]))
        model.pretrain(pretrain_valuations, pretrain_epoch, pretrain_transform)


    bidders = [
        strat_to_bidder(models[0 if model_sharing else i], batch_size, i)
        for i in range(param_dict["n_players"])
    ]
    # print('warning: BNE initialization')
    # models[1].pretrain(pretrain_valuations, 2*pretrain_epoch,
    #     optimal_bid(mechanism, param_dict)
    # )

    env = AuctionEnvironment(
        mechanism,
        agents = bidders,
        n_players = param_dict["n_players"],
        batch_size = batch_size,
        strategy_to_player_closure = strat_to_bidder
    )

    learners = [
        ESPGLearner(
            model = model,
            environment = env,
            hyperparams = learner_hyperparams,
            optimizer_type = optimizer_type,
            optimizer_hyperparams = optimizer_hyperparams,
            strat_to_player_kwargs = {"player_position": i}
        )
        for i, model in enumerate(models)
    ]


    # ## Set up equilibrium-environment
    bne_strategies = [
        ClosureStrategy(
            partial(
                optimal_bid(mechanism, param_dict),
                player_position = i
            )
        )
        for i in range(param_dict["n_players"])
    ]

    bne_env = AuctionEnvironment(
        mechanism,
        agents = [
            strat_to_bidder(bne_strategy, batch_size, i)
            for i, bne_strategy in enumerate(bne_strategies)
        ],
        n_players = param_dict["n_players"],
        batch_size = batch_size,
        strategy_to_player_closure = strat_to_bidder
    )


    # ## Training
    run_name = str(time.strftime('%Y%m%d_%H%M%S', time.localtime()))

    if logging:
        logdir = os.path.join(
            log_root, 'expiriments_nils', auction_type_str,
            str(param_dict["n_players"]) + 'players_' \
                + str(param_dict["n_items"]) + 'items',
            run_name
        )
        print('logdir:', logdir)
        os.makedirs(logdir, exist_ok=False)

        if save_figure_to_disk:
            os.mkdir(os.path.join(logdir, 'plots'))
    else:
        logdir = None

    # calculate utility vs BNE
    bne_utilities = list()
    for agent in bne_env.agents:
        u = bne_env.get_reward(agent, draw_valuations=True)
        bne_utilities.append(u)
    print('bne_utilities', bne_utilities)

    with SummaryWriter(logdir, flush_secs=60) as writer:

        # torch.cuda.empty_cache()

        if logging:
            log_once(writer, 0, epoch, param_dict["n_players"], log_name,
                     n_parameters, seed, models, batch_size, learner_hyperparams,
                     optimizer_type, optimizer_hyperparams, pretrain_epoch, env)

        for e in range(epoch + 1):

            torch.cuda.empty_cache()

            # plotting
            if e % plot_epoch == 0 and logging:
                plot_bid_function(
                    bidders,
                    optimal_bid(mechanism, param_dict, return_payoff_dominant=False),
                    optimal_bid_2(mechanism, param_dict),
                    log_name, logdir, writer, e=e,
                    bounds = [param_dict["u_lo"], param_dict["u_hi"]],
                    split_award = {
                        'split_award': True,
                        "efficiency_parameter": param_dict["efficiency_parameter"],
                        "input_length": param_dict["input_length"] \
                    } if param_dict["exp_no"] == 6 else None,
                    save_fig_to_disk = save_figure_to_disk,
                    device = device
                )
                # if param_dict["n_items"] == 2 \
                # and param_dict["n_players"] < 4 \
                # and param_dict["exp_no"] != 6 \
                # or param_dict["exp_no"] == 2:
                #     plot_bid_function_3d(
                #         writer, e, param_dict["exp_no"],
                #         param_dict["n_items"], log_name, logdir, bidders,
                #         batch_size, device, #bounds=[param_dict["u_lo"], param_dict["u_hi"]],
                #         split_award = param_dict["exp_no"]==6,
                #         save_fig_to_disk = save_figure_to_disk
                #     )

            start_time = time.time()
            # torch.cuda.reset_max_memory_allocated(device=device)

            env.prepare_iteration()

            # record utilities and do optimizer step
            utilities = list()
            for i, learner in enumerate(learners):
                u = learner.update_strategy_and_evaluate_utility()
                utilities.append(u)
            # print('util:', np.round(u.detach().cpu().numpy(), 4), end='\t')

            elapsed = time.time() - start_time

            # memory = torch.cuda.max_memory_allocated(device=device) * (2**-17)

            # log relative utility loss induced by not playing the BNE
            against_bne_utilities = list()
            for i, model in enumerate(models):
                u = bne_env.get_strategy_reward(model, player_position=i, draw_valuations=True)
                against_bne_utilities.append(u)
            # print(' util_vs_bne:', np.round(u.detach().cpu().numpy(), 4), end='\t')

            # logging
            if logging:
                log_metrics(
                    writer = writer,
                    utilities = utilities,
                    bne_utilities = bne_utilities,
                    against_bne_utilities = against_bne_utilities,
                    overhead = elapsed,
                    e = e,
                    log_name = log_name,
                    n_players = param_dict["n_players"],
                    models = models,
                    policy_metrics = {
                        param_dict["BNE1"]: [
                            policy_metric(
                                model.forward,
                                optimal_bid(mechanism, param_dict),
                                param_dict["n_items"],
                                selection = split_award_dict \
                                    if param_dict["exp_no"] == 6 else 'random',
                                bounds = [param_dict["u_lo"], param_dict["u_hi"]],
                                item_interest_limit = param_dict["item_interest_limit"] if \
                                    "item_interest_limit" in param_dict.keys() else None,
                                eval_points_max = 2 ** 18,
                                device = device
                            )
                            for model in models],
                        param_dict["BNE2"]: [
                            policy_metric(
                                model.forward,
                                optimal_bid_2(mechanism, param_dict),
                                param_dict["n_items"],
                                selection = split_award_dict \
                                    if param_dict["exp_no"] == 6 else 'random',
                                bounds = [param_dict["u_lo"], param_dict["u_hi"]],
                                item_interest_limit = param_dict["item_interest_limit"] if \
                                    "item_interest_limit" in param_dict.keys() else None,
                                eval_points_max = 2 ** 18,
                                device = device
                            )
                            for model in models]
                    }
                )

            print('epoch {}:\t{}s'.format(e, round(elapsed, 2)))



    if logging:
        for i, model in enumerate(models):
            torch.save(model.state_dict(), os.path.join(logdir, 'saved_model_' + str(i) + '.pt'))
