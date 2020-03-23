"""
Author Nils Kohring
Date   Nov 2019
Desc   Helper file for testing of learning in multi unit auction formats
"""

import sys
import os

sys.path.append(os.path.realpath('.'))
from copy import deepcopy

from scripts_work_in_progress.experiments_nils import *
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import time
import warnings
from typing import Callable
import numpy as np
import pandas as pd
from functools import partial
import itertools

import scipy.integrate as integrate
from scipy import interpolate

import torch
import torch.nn as nn

from bnelearn.bidder import Bidder
from bnelearn.mechanism import *
from bnelearn.strategy import NeuralNetStrategy, ClosureStrategy
from bnelearn.environment import AuctionEnvironment
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator



colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
          '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
          '#bcbd22', '#17becf']
colors_warm = ['maroon', 'firebrick', 'red', 'salmon',
               'coral', 'lightsalmon', 'mistyrose', 'lightgrey',
               'white']


def create_splitaward_setting():
    param_dict = dict()
    param_dict["exp_no"] = 6
    mechanism = FPSBSplitAwardAuction(cuda=True)
    param_dict["n_players"] = 2
    param_dict["n_items"] = 2
    param_dict["u_lo"] = 1.0
    param_dict["u_hi"] = 1.4
    param_dict["efficiency_parameter"] = 0.3
    param_dict["BNE1"] = "PD_Sigma_BNE"
    param_dict["BNE2"] = "WTA_BNE"
    param_dict["input_length"] = param_dict["n_items"] - 1 \
        if param_dict["exp_no"] == 6 else param_dict["n_items"]
    split_award_dict = {
        'split_award': True,
        'efficiency_parameter': param_dict["efficiency_parameter"],
        'input_length': param_dict["input_length"],
        'linspace': False
    }
    return mechanism, param_dict, split_award_dict

def create_discriminatory_setting():
    param_dict = dict()
    param_dict["exp_no"] = 4
    mechanism = MultiItemDiscriminatoryAuction(cuda=True)
    param_dict["n_players"] = 2
    param_dict["n_items"] = 2
    param_dict["u_lo"] = 0.0
    param_dict["u_hi"] = 1.0
    param_dict["input_length"] = param_dict["n_items"]
    return mechanism, param_dict



# ## Optimal Policies

def value_cdf(u_lo, u_hi):
    """
    CDF for uniform valuations on [u_lo, u_hi].
    """
    def cdf(v: torch.Tensor):
        warnings.warn("Uniform valuations only!", Warning)

        out = (v - u_lo) / (u_hi - u_lo)

        try:
            out[out < 0] = 0
            out[out > 1] = 1
        except: 
            pass

        return out
    return cdf

def optimal_bid(
        mechanism,
        param_dict,
        return_payoff_dominant = True
    ) -> torch.Tensor:
    """
    BNE bidding
    """
    def ob(
            valuation: torch.Tensor or np.ndarray or float,
            player_position: int = 0
        ):
        if not isinstance(valuation, torch.Tensor):
            valuation = torch.tensor(valuation, dtype=torch.float, device=device)
        else:
            valuation = valuation.clone().detach()

        # unsqueeze if simple float
        if valuation.dim() == 0:
            valuation.unsqueeze_(0)

        elif valuation.shape[1] == 1:
            valuation = torch.cat((valuation, param_dict["efficiency_parameter"] * valuation), 1)

        if isinstance(mechanism, MultiItemVickreyAuction):
            return valuation

        elif isinstance(mechanism, MultiItemDiscriminatoryAuction): # is inefficient
            if param_dict["exp_no"] == 3:
                raise NotImplementedError()

            elif param_dict["exp_no"] == 4:
                # 2x2 uniform vauluations

                def b_approx(v, s, t):
                    b = torch.clone(v)
                    lin_e = np.array([[1, 1, 1], [2*t, 1, 0], [t**2, t, 1]])
                    lin_s = np.array([0.47, s/t, s])
                    x = np.linalg.solve(lin_e, lin_s)
                    b[v < t] *= s/t
                    b[v >= t] = x[0]*b[v >= t]**2 + x[1]*b[v >= t] + x[2]
                    return b

                b1 = lambda v: b_approx(v, s=0.42, t=0.90)
                b2 = lambda v: b_approx(v, s=0.35, t=0.55)

                opt_bid = valuation
                opt_bid[:,0] = b1(opt_bid[:,0])
                opt_bid[:,1] = b2(opt_bid[:,1])
                opt_bid = opt_bid.sort(dim=1, descending=True)[0]
                return opt_bid

            elif param_dict["exp_no"] == 5:
                if isinstance(strat_to_bidder(None, 1, None).value_distribution,
                              torch.distributions.uniform.Uniform):
                    return valuation / 2

                elif isinstance(strat_to_bidder(None, 1, None).value_distribution,
                                torch.distributions.normal.Normal):

                    # TODO: just calc once and then interpolate with that?
                    def muda_tb_cmv_bne(
                            value_pdf: callable,
                            value_cdf: callable = None,
                            lower_bound: int = 0,
                            epsabs = 1e-3
                        ):
                        if value_cdf is None:
                            def _value_cdf(x):
                                return integrate.quad(value_pdf, lower_bound, x, epsabs=epsabs)[0]
                            value_cdf = _value_cdf
                        def inner(s, x):
                            return integrate.quad(lambda t: value_pdf(t) / value_cdf(t),
                                                  s, x, epsabs=epsabs)[0]
                        def outer(x):
                            return integrate.quad(lambda s: np.exp(-inner(s, x)),
                                                  lower_bound, x, epsabs=epsabs)[0]
                        def bidding(x):
                            if not hasattr(x, '__iter__'):
                                return x - outer(x)
                            else:
                                return np.array([xi - outer(xi) for xi in x])
                        return bidding

                    dist = strat_to_bidder(None, 1, None).value_distribution
                    bidding = muda_tb_cmv_bne(lambda x: torch.exp(dist.log_prob(x)).cpu().numpy(),
                                              lambda x: dist.cdf(x).cpu().numpy())

                    opt_bid = np.zeros_like(valuation.cpu().numpy())
                    for agent in range(n_players):
                        opt_bid[agent] = bidding(valuation[agent,:])
                    return torch.tensor(opt_bid)

            else:
                warnings.warn("No explict BNE for MultiItemDiscriminatoryAuction known.", Warning)
                return valuation

        elif isinstance(mechanism, MultiItemUniformPriceAuction): # is inefficient
            if param_dict["n_players"] == 2 and param_dict["n_items"] == 3 \
                    and param_dict["u_lo"] == 0 and param_dict["u_hi"] == 1 \
                    and param_dict["item_interest_limit"] == 2:
                opt_bid = torch.clone(valuation)
                opt_bid[:,1] = opt_bid[:,1] ** 2
                opt_bid[:,2] = 0
                return opt_bid
            elif param_dict["n_players"] == 2 and param_dict["n_items"] == 2:
                opt_bid = torch.clone(valuation)
                opt_bid[:,1] = 0
                return opt_bid
            elif param_dict["n_players"] == param_dict["n_items"]:
                opt_bid = torch.zeros_like(valuation)
                opt_bid[:,0] = param_dict["u_hi"]
                return opt_bid
            elif param_dict["n_players"] > param_dict["n_items"]: # & cdf of v_1 is strictly increasing
                opt_bid = torch.clone(valuation)
                opt_bid[:,1:] = 0
                warnings.warn("Only BNE bidding for 0th item known.", Warning)
                return opt_bid
            else:
                warnings.warn("No explict BNE for MultiItemUniformPriceAuction known.", Warning)
                return valuation

        elif isinstance(mechanism, FPSBSplitAwardAuction):
            # sigma/pooling equilibrium
            if param_dict["exp_no"] == 6: # Anton and Yao, 1992, Proposition 2

                if param_dict["input_length"] == 1 and param_dict["n_items"] == 2 \
                and valuation.shape[1] == 1:
                    valuation = torch.cat(
                        (valuation, param_dict["efficiency_parameter"]*valuation),
                        axis = 1
                    )

                sigma_bounds = torch.ones_like(valuation, device=valuation.device)
                sigma_bounds[:,0] = param_dict["efficiency_parameter"] * param_dict["u_hi"]
                sigma_bounds[:,1] = (1 - param_dict["efficiency_parameter"]) * param_dict["u_lo"]
                # [:,0]: lower bound and [:,1] upper

                _p_sigma = (1 - param_dict["efficiency_parameter"]) * param_dict["u_lo"]
                # highest possible p_sigma

                def G(theta):
                    return _p_sigma \
                        + (_p_sigma - param_dict["u_hi"]*param_dict["efficiency_parameter"] \
                            * value_cdf(param_dict["u_lo"], param_dict["u_hi"])(theta)) \
                        / (1 - value_cdf(param_dict["u_lo"], param_dict["u_hi"])(theta))

                wta_bounds = 2 * sigma_bounds
                wta_bounds[:,1] = G(valuation[:,0])

                # cutoff value: otherwise would go to inf
                # lim = 4 * param_dict["u_hi"]
                # wta_bounds[wta_bounds > lim] = lim

            if return_payoff_dominant:
                return torch.cat((
                    wta_bounds[:,1].unsqueeze(0).t_(),
                    sigma_bounds[:,1].unsqueeze(0).t_()),
                    axis = 1
                )
            else:
                return {'sigma_bounds': sigma_bounds, 'wta_bounds': wta_bounds}

        else:
            warnings.warn("No explict BNE known for {}.".format(type(mechanism)), Warning)
            return valuation

    return ob

def optimal_bid_2(
        mechanism,
        param_dict
    ) -> torch.Tensor:
    """
    2nd BNE bidding
    """
    def ob(
            valuation: torch.Tensor or np.ndarray or float,
            player_position: int = 0
        ):
        if not isinstance(valuation, torch.Tensor):
            valuation = torch.tensor(valuation, dtype=torch.float, device=device)
        else:
            valuation = valuation.clone().detach()

        if isinstance(mechanism, MultiItemUniformPriceAuction) and param_dict["exp_no"] == 1 and \
            (param_dict["n_players"] == 2 and param_dict["n_items"] == 2):
            opt_bid = torch.clone(valuation)
            opt_bid[:,0] = param_dict["u_hi"]
            opt_bid[:,1] = 0
            return opt_bid

        elif isinstance(mechanism, FPSBSplitAwardAuction):
            # WTA equilibrium
            if param_dict["exp_no"] == 6: # Anton and Yao, 1992: Proposition 4

                if param_dict["input_length"] == 1 and param_dict["n_items"] == 2 \
                and valuation.shape[1] == 1:
                    valuation = torch.cat(
                        (valuation, param_dict["efficiency_parameter"]*valuation),
                        axis=1
                    )
                opt_bid_batch_size = 2 ** 12
                opt_bid = np.zeros(shape=(opt_bid_batch_size, valuation.shape[1]))

                if 'opt_bid_function' not in globals():
                    # do one-time approximation via integration
                    eps = 1e-4
                    val_lin = np.linspace(param_dict["u_lo"], param_dict["u_hi"] - eps, opt_bid_batch_size)

                    def integral(theta):
                        return np.array(
                            [integrate.quad(
                                lambda x: (1 - value_cdf(param_dict["u_lo"],
                                    param_dict["u_hi"])(x))**(param_dict["n_players"] - 1),
                                v, param_dict["u_hi"],
                                epsabs = eps
                             )[0]
                             for v in theta]
                        )

                    def opt_bid_100(theta):
                        return theta + (integral(theta) / \
                            ((1 - value_cdf(param_dict["u_lo"], \
                            param_dict["u_hi"])(theta))**(param_dict["n_players"] - 1)))

                    opt_bid[:,0] = opt_bid_100(val_lin)
                    opt_bid[:,1] = opt_bid_100(val_lin) - \
                        param_dict["efficiency_parameter"] * param_dict["u_lo"]
                        # or more

                    global opt_bid_function
                    opt_bid_function = [
                        interpolate.interp1d(
                            val_lin, opt_bid[:,0],
                            fill_value = 'extrapolate'
                        ),
                        interpolate.interp1d(
                            val_lin,
                            opt_bid[:,1], fill_value = 'extrapolate'
                        )
                    ]

                # (re)use interpolation of opt_bid done on first batch
                opt_bid = torch.tensor(
                    [
                        opt_bid_function[0](valuation[:,0].cpu().numpy()),
                        opt_bid_function[1](valuation[:,0].cpu().numpy())
                    ],
                    device = valuation.device
                ).t_()

                opt_bid[opt_bid < 0] = 0
                opt_bid[torch.isnan(opt_bid)] = 0

                return opt_bid

        else:
            warnings.warn("No 2nd explicit BNE known.", Warning)
            return valuation

    return ob



# ## Policy Evaluation

def rmse(y, y_hat):
    """
    Root mean squared error.
    """
    return torch.sqrt(torch.mean((torch.clone(y_hat) - torch.clone(y)) ** 2))

def policy_metric(
        policy_1: Callable,
        policy_2: Callable,
        dim: int,
        bounds = [0, 1],
        eval_points_max: int = 2**7,
        selection = 'random',
        item_interest_limit = None,
        dim_of_interest = None,
        device = None,
    ):
    """
    Calculate the p-norm on a grid between the two policy functions.

    TODO: Consider sampleing according to underlying distribution instead of
        the uniform grid sampleing!
    """
    valuations = multi_unit_valuations(device, bounds, dim, eval_points_max, selection, item_interest_limit)

    policy_1_bidding = policy_1(valuations).detach()
    policy_2_bidding = policy_2(valuations).detach()

    if dim_of_interest is not None:
        policy_1_bidding = policy_1_bidding[:,dim_of_interest]
        policy_2_bidding = policy_2_bidding[:,dim_of_interest]

    metric = rmse(policy_1_bidding, policy_2_bidding)

    return metric.detach().cpu().numpy()

def fpsbaw_sigma_e_metric(
        policy: Callable or torch.Tensor, 
        eval_points_max = 10000
    ):

    if isinstance(policy, torch.Teonsr):
        # TODO
        pass

    def G(theta, p_sigma):
        return p_sigma \
            + (p_sigma - u_hi*efficiency_parameter*value_cdf(theta)) \
            / (1 - value_cdf(theta))
    def is_strictly_increasing(b: torch.Tensor) -> bool:
        return b == b.sort()[0]
    def is_constant(b: torch.Tensor, abs_tol=1e-4) -> bool:
        return b.max() - b.min() < abs_tol
    def is_sigma_equilibrium(opt_bid: torch.Tensor) -> bool:
        u_los = valuation[:,0] == u_lo
        return (is_strictly_increasing(opt_bid[:,0])
                and opt_bid[:,0] >= valuation[:,0]
                and opt_bid[u_los,0] == 2*opt_bid[0,1]
                and is_constant(opt_bid[:,1])
                and opt_bid[:,1] >= u_hi*efficiency_parameter
                and opt_bid[:,1] <= u_lo*(1-efficiency_parameter)
                and opt_bid[:,1] <= G(valuation[:,0], opt_bid[0,1]))

def meta_game(
        mechanism,
        policies,
        policy_names = None,
        device = None,
        bounds = [0, 1],
        n_players = 2,
        n_items = 2,
        batch_size = 1e7,
        policy_of_interest = 1,
        split_award = None
    ):
    assert n_players == 2, 'Only n_players = 2 supported at this time!'
    assert len(policies) == len(policy_names), '´policy_names´ doesn´t match policies.'

    n = len(policies)
    payoff_table = np.zeros([n] * n_players)

    valuation = multi_unit_valuations(device, bounds, n_items, batch_size,
            'random' if split_award is None else split_award)
    true_batch_size = len(valuation)
    valuations = list()
    biddings = list()
    for policy in policies:
        random_idx = torch.randperm(true_batch_size).long()
        valuations.append(valuation[random_idx,:])
        biddings.append(policy(valuations[-1]))

    fig, axs = plt.subplots(n, 1, sharex=True, figsize=(5, 1.3*n))
    plt.suptitle('Utility Density of Startegy ' + policy_names[policy_of_interest], x=.5, y=.98)
    i = 0
    for idxs in itertools.combinations_with_replacement(range(len(biddings)), n_players):
        bids_commit = torch.zeros((true_batch_size, n_players, n_items), device=device)
        for p in range(n_players):
            bids_commit[:,p,:] = biddings[idxs[p]]

        allocations, payments = mechanism.run(bids_commit)

        utility_player_0 = (valuations[idxs[0]] * allocations[:,0,:]).sum(dim=1) - payments[:,0]
        utility_player_1 = (valuations[idxs[1]] * allocations[:,1,:]).sum(dim=1) - payments[:,1]

        if split_award:
            utility_player_0 *= -1
            utility_player_1 *= -1

        if policy_of_interest in idxs:
            (p0_idx, p1_idx) = (idxs[0], idxs[1]) if policy_of_interest == idxs[0] else (idxs[1], idxs[0])
            util = utility_player_0 if policy_of_interest == idxs[0] else utility_player_1
            util_opp = utility_player_1 if policy_of_interest == idxs[0] else utility_player_0
            axs[i].hist(util.detach().cpu().numpy(), 100, density=True, #histtype='step',
                        color=colors[0], label=policy_names[p0_idx])
            axs[i].hist(util_opp.detach().cpu().numpy(), 100, density=True, histtype='step', 
                        color=colors[1], label=policy_names[p1_idx])
            axs[i].plot([util.mean(), util.mean()], [0, 10], '--', color=colors[0])
            axs[i].plot([util_opp.mean(), util_opp.mean()], [0, 10], '--', color=colors[1])
            axs[i].set_xlim([-1, 1])
            axs[i].set_ylim([0, 2])
            axs[i].set_yticks([], [])
            axs[i].legend(loc='upper left' if utility_player_0.mean() >= 0 else 'upper right')
            i += 1

        payoff_table[idxs] += utility_player_0.mean()
        if tuple(reversed(idxs)) != idxs:
            payoff_table[tuple(reversed(idxs))] += utility_player_1.mean()

    plt.xlabel('utility')
    plt.tight_layout(rect=(0, 0, 1, .96))
    plt.savefig(os.path.join('/home/kohring/bnelearn/experiments/' \
        + 'expiriments_nils/plots/_strathist_' + str(policy_names[policy_of_interest]) \
        + '_plot.png'), dpi=256)
    plt.close()

    # plot
    fig, ax = plt.subplots()
    ax.imshow(payoff_table, cmap='winter', interpolation='nearest')
    ax.xaxis.set_ticks_position('none')
    ax.yaxis.set_ticks_position('none')
    if policy_names is not None:
        ax.set_xticks(np.arange(len(policy_names)))
        ax.set_yticks(np.arange(len(policy_names)))
        ax.set_xticklabels(policy_names)
        ax.set_yticklabels(policy_names)
        for i in range(len(policy_names)):
            for j in range(len(policy_names)):
                ax.text(j, i, round(payoff_table[i, j], 2), ha="center", va="center")
    ax.set_xlabel('opponent policy')
    ax.set_ylabel('mean utility of policy')
    ax.set_xlim([-.5, n-.5])
    ax.set_ylim([-.5, n-.5])
    ax.set_title("Strategy Comparisons\n(sample size of {})".format(true_batch_size))
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    plt.setp(ax.get_yticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    # ax.legend(title="zeros: bidding 0 on both itmes\nones: bidding 1 on both itmes",
    #           loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))
    plt.tight_layout()
    plt.savefig(os.path.join('/home/kohring/bnelearn/experiments/' \
        + 'expiriments_nils/plots/_meta_game_plot.png'), dpi=256)
    plt.close()

    return payoff_table

def replicator_dynamics(
        payoff_table: np.array,
        strategy_names: list = None,
        step_size: float = 1e-1,
        eps: float = 1e-3,
        max_iter: int = int(1e3),
        verbose: bool = False,
        path = '/home/kohring/bnelearn/experiments/expiriments_nils/plots/'
    ):
    """Replicator Dynamics"""

    n_policies = payoff_table.shape[0]
    assert n_policies == payoff_table.shape[1], 'payoff table must be symmetric.'

    x = (1/n_policies) * np.ones((n_policies, 1))

    def derivative(x):
        Ax = np.matmul(A, x)
        return x * (Ax - np.matmul(x.transpose(), Ax))

    if verbose:
        change = np.zeros((n_policies + 2, max_iter))
        change[-1,:] = 1

    for i in range(max_iter):
        step = step_size * derivative(x)
        x += step
        if verbose:
            for j in range(n_policies):
                change[1+j,i] = sum(x[:1+j])
        if (x > 1 - eps).any() or sum(np.abs(step / step_size)) < eps:
            print('population converged')
            break

    if verbose and i > 3:
        fig, ax = plt.subplots()
        for p in reversed(range(n_policies)):
            plt.fill_between(np.arange(max_iter), change[p,:], change[p+1,:],
                label = p if strategy_names is None else strategy_names[p] \
                        + ' (' + str(round(float(x[p]), 2)) + ')',
                color = cm.hot(p / n_policies, 1)
            )
        plt.legend(loc='upper right', title='policy (surviving rate)')
        plt.xlim([0, i-1])
        plt.ylim([0, 1])
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        plt.xlabel('time')
        plt.ylabel('proportion in population')
        plt.title('Replicator Dynamcis Learning')
        os.makedirs(path, exist_ok=True)
        plt.savefig(os.path.join(path, '_replicator_dynamics_plot.png'), dpi=256)

    return np.argmax(x)

def multi_unit_valuations(
        device = None,
        bounds = [0, 1],
        dim = 2,
        batch_size = 100,
        selection = 'random',
        item_interest_limit = False,
        sort = False,
    ):
    """Returns uniformly sampled valuations for multi unit auctions."""
    # for uniform vals and 2 items <=> F1(v)=v**2, F2(v)=2v-v**2

    eval_points_per_dim = round((2*batch_size) ** (1/dim))
    valuations = torch.zeros(eval_points_per_dim ** dim, dim, device=device)

    if selection == 'random':
        valuations.uniform_(bounds[0], bounds[1])
        valuations = valuations.sort(dim=1, descending=True)[0]

    elif 'split_award' in selection.keys():
        if 'linspace' in selection.keys() and selection['linspace']:
            valuations[:,0] = torch.linspace(bounds[0], bounds[1],
                eval_points_per_dim ** dim, device=device)
        else:
            valuations.uniform_(bounds[0], bounds[1])
        valuations[:,1] = selection['efficiency_parameter'] * valuations[:,0]
        # if 'input_length' in selection.keys():
        #     valuations = valuations[:,:selection['input_length']]

    else:
        lin = torch.linspace(bounds[0], bounds[1], eval_points_per_dim, device=device)
        mesh = torch.meshgrid([lin] * dim)
        for n in range(dim):
            valuations[:,n] = mesh[n].reshape(eval_points_per_dim ** dim)

        mask = valuations.sort(dim=1, descending=True)[0]
        mask = (mask == valuations).all(dim=1)
        valuations = valuations[mask]

    if isinstance(item_interest_limit, int):
        valuations[:,item_interest_limit:] = 0
    if sort:
        valuations = valuations.sort(dim=1)[0]

    return valuations

def MIUP2x2_utility(
        bid, valuation,
        h1: Callable = lambda x: 2*x,
        h2: Callable = lambda x: 0,
    ):
    """

    """
    eps = 1e-4
    def H1(xx):
        return torch.tensor([
            integrate.quad(h1, 0, x, epsabs=eps)[0]
        for x in xx])
    def H2(xx):
        return torch.tensor([
            integrate.quad(h2, 0, x, epsabs=eps)[0]
        for x in xx])
    price_2items = torch.tensor(
        [integrate.quad(lambda c1: c1*h1(c1), 0, b, epsabs=eps)[0]
         for b in bid[:,1]]
    )
    price_1item = torch.tensory(
        [integrate.quad(lambda c2: c2*h2(c2), b2, b1, epsabs=eps)[0]
         for b2, b1 in zip(bid[:,1], bid[:,0])]
    )
    
    utility = H1(bid[:,1])*(valuation[:,0] + valuation[:,1]) - 2*price_2items \
        + (H2(bid[:,0]) - H1(bid[:,1]))*valuation[:,0] \
        - (H2(bid[:,1]) - H1(bid[:,1]))*bid[:,1] - price_1item
    return utility.mean()



# ## Plotting

def plot_bid_function(
        bidders,
        optimal_bid,
        optimal_bid_2,
        log_name = None,
        logdir = None,
        writer = None,
        e = None,
        bounds = [0., 1.],
        # sort_by_bids = False,
        split_award = None,
        save_vectors_to_disc = False,
        save_fig_to_disc = False,
        format = 'pdf',
        device = None
    ):
    """Method for plotting"""

    n_items = bidders[0].n_items
    n_players = len(bidders)
    plot_points = 25

    if split_award is not None:
        split_award['linspace'] = True
    # valuations = multi_unit_valuations(device, bounds, n_items, plot_points,
    #     'random' if split_award is None else split_award, sort=split_award is not None)
    valuations = deepcopy(bidders[0]).draw_valuations_()[:plot_points,:]
    if split_award is not None:
        valuations, _ = valuations.sort(0)

    b_opt = optimal_bid(valuations)
    b_opt_2 = optimal_bid_2(valuations).cpu().numpy()

    if split_award is None:
        b_opt = b_opt.cpu().numpy()
    else:
        for k, v in b_opt.items():
            b_opt[k] = v.cpu().numpy()
        temp = b_opt
        b_opt = b_opt_2
        b_opt_2 = temp

    actions = list()
    for bidder in bidders:
        try:
            dim = bidder.strategy.input_length
        except:
            try:
                dim = n_items
            except Exception as exc:
                print(exc)
        try:
            actions.append(bidder.strategy.play(valuations[:,:dim]))
        except:
            actions.append(bidder.strategy(valuations[:,:dim]))

    # sorting of points, s.t. 1st plot corresponds to 1st item, etc.
    # (from sorted values to sorted bids)
    acts = list()
    for act in actions:
        # if sort_by_bids:
        #     sorted_idx = torch.sort(act, dim=1, descending=True)[1]
        #     acts.append(batched_index_select(act, 1, sorted_idx).detach().cpu().numpy())
        # else:
        acts.append(act.detach().cpu().numpy())

    fig, axs = plt.subplots(nrows=1, ncols=n_items, sharey=True, figsize=[7, 4])
    plt.cla()

    if not isinstance(axs, np.ndarray): # only one item/plot
        axs = [axs]

    if split_award is not None and n_items == 2:
        if valuations.shape[1] == 1:
            valuations = torch.cat(
                (valuations, split_award["efficiency_parameter"] * valuations), 1
            )

    valuations = valuations.cpu().numpy()

    for item in range(n_items):
        if split_award is not None:
            plot = list(reversed(range(n_items)))[item]
        else:
            plot = item

        for agent_idx in range(n_players):
            if split_award is None:
                zeros = acts[agent_idx][:,item] < 1e-9
                axs[plot].scatter(
                    valuations[:,item][~zeros], acts[agent_idx][:,item][~zeros],
                    marker = '.',
                    color = colors[agent_idx % len(colors)],
                    label = 'agent ' + str(agent_idx + 1),
                )
                axs[plot].plot(
                    valuations[:,item][zeros], acts[agent_idx][:,item][zeros],
                    marker = "x",
                    color = colors[agent_idx % len(colors)],
                )
            else:
                zeros = acts[agent_idx][:,item] < 1e-9
                axs[plot].plot(
                    valuations[:,item][~zeros], acts[agent_idx][:,item][~zeros],
                    '.-',
                    color = colors[agent_idx % len(colors)],
                    label = 'agent ' + str(agent_idx + 1),
                )
                axs[plot].plot(
                    valuations[:,item][zeros], acts[agent_idx][:,item][zeros],
                    marker = "x",
                    color = colors[agent_idx % len(colors)],
                )

        axs[plot].plot(
            valuations[:,item], b_opt[:,item],
            '.' if split_award is None else '-', color='black',
            label = 'WTA BNE strategy' if split_award is not None
                else 'BNE strategy'
        )

        if split_award is not None:
            x_label = 'cost'
            axs[plot].yaxis.grid(which="major", linestyle=':')
            axs[plot].set_title('100% share' if item==0 else '50% share')

            select = 'wta_bounds' if item == 0 else 'sigma_bounds'
            axs[plot].plot(
                valuations[:,0 if item == 0 else 1],
                b_opt_2[select][:,0], '--',
                label = 'pooling BNE bounds',
                color = 'black'
            )
            axs[plot].plot(
                valuations[:,0 if item == 0 else 1],
                b_opt_2[select][:,1], '--',
                color = 'black'
            )

        if split_award is not None:
            axs[plot].set_xlim(
                [bounds[0], bounds[1]] if item == 0 else
                [split_award["efficiency_parameter"]*bounds[0],
                 split_award["efficiency_parameter"]*bounds[1]]
            )
            axs[plot].set_ylim(
                [0, 1.9*bounds[1]] #if item == 0 else
                # [0, 3.2*split_award["efficiency_parameter"]*bounds[1]]
            )

        else:
            x_label = 'valuation'
            axs[plot].set_title(str(item+1) + '. bid')
            axs[plot].set_xlim([bounds[0], bounds[1]])
            axs[plot].set_ylim([bounds[0], bounds[1]])

        axs[plot].set_xlabel(x_label)

        if plot == 0:
            axs[plot].set_ylabel('bid')
            if n_players < 10:
                axs[plot].legend(loc='upper left')

    axs[plot].locator_params(axis='x', nbins=5)
    fig.tight_layout()

    if save_fig_to_disc and logdir is not None:
        try:
            os.mkdir(os.path.join(logdir, 'plots'))
        except FileExistsError:
            pass
        print(os.path.join(logdir, 'plots', f'_{e:05}.' + format))
        plt.savefig(os.path.join(logdir, 'plots', f'_{e:05}.' + format))
    if writer and log_name is not None:
        writer.add_figure('plot/plot', fig, e)

    # plt.show()
    
def plot_bid_function_3d(
        writer,
        e,
        exp_no,
        n_items,
        log_name,
        logdir,
        bidders,
        batch_size,
        device,
        split_award = False,
        bounds = [0, 1],
        save_fig_to_disc = False
    ):
    assert n_items == 2 or exp_no == 2, 'Only case of n_items equals 2 can be plotted.'

    plot_points = 10

    n_players = len(bidders)

    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib.ticker import LinearLocator, FormatStrFormatter

    lin = torch.linspace(bounds[0], bounds[1], plot_points)
    x, y = torch.meshgrid([lin, lin])

    plot_valuations = torch.zeros(plot_points**2, n_players, 3 if exp_no == 2 else 2, device=device)
    strategy = [None] * n_players
    for agent_idx in range(n_players):
        plot_valuations[:,agent_idx,0] = x.reshape(plot_points**2)
        plot_valuations[:,agent_idx,1] = y.reshape(plot_points**2)
        strategy[agent_idx] = bidders[agent_idx].strategy.play(
                plot_valuations[:,agent_idx,:].view(plot_points**2, -1)
            )
        if exp_no == 2:
            strategy[agent_idx] = strategy[agent_idx][:,:2]

    x, y = x.cpu().numpy(), y.cpu().numpy()
    if split_award:
        mask = x == x
    else:
        mask = x >= y # better way?

    fig = plt.figure(figsize=[9, 7])
    for agent_idx in range(n_players):
        for item_idx in range(2):
            ax = fig.add_subplot(n_players, 2, agent_idx*2+item_idx+1, projection='3d')
            b = strategy[agent_idx][:,item_idx].reshape(plot_points, plot_points).detach().cpu().numpy()
            ax.plot_trisurf(x[mask], y[mask], b[mask],
                    cmap = 'plasma',
                    # linewidth = 0,
                    antialiased = True
                )
            ax.set_xlim(bounds[0], bounds[1])
            ax.set_ylim(bounds[0], bounds[1])
            ax.set_zlim(bounds[0], bounds[1]+.1*(bounds[1]-bounds[0]))
            if split_award:
                x_label = '100% share cost'
                y_label = '50% share cost'
            else:
                x_label = 'item 0 value'
                y_label = 'item 1 value'
            ax.set_xlabel(x_label); ax.set_ylabel(y_label)#; ax.set_zlabel('bid')
            ax.zaxis.set_major_locator(LinearLocator(10))
            ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
            if split_award:
                share = '100' if item_idx==0 else '50'
                title = 'agent {} offer for the {}% share'.format(agent_idx, share)
            else:
                title = 'agent {} bidding for her {}. item'.format(agent_idx, item_idx)
            ax.set_title(title)
            ax.view_init(20, -135)
    fig.suptitle('iteration {}'.format(e), size=16)
    fig.tight_layout()

    if save_fig_to_disc and logdir is not None:
        plt.savefig(os.path.join(logdir, 'plots', f'_{e:05}_3d.png'))
    if writer and log_name is not None:
        writer.add_figure('plot/plot_3d', fig, e)

    plt.show()

def plot_utilities(
        path = '/home/kohring/bnelearn/experiments/expiriments_nils/MultiItemUniformPriceAuction' \
               + '/2players_2items/',
        metric = 'utilities_vs_bne_',
        run_name_conatains = ('20191129_'),
        epoch_lim = 200,
        save_fig = True,
    ):

    run_dirs = os.listdir(path)    
    if run_name_conatains is not None and len(run_name_conatains) > 0:
        new_rundirs = []
        for selecter in run_name_conatains:
            [new_rundirs.append(r) for r in run_dirs if r.find(selecter) != -1]
        run_dirs = new_rundirs

    df = pd.DataFrame()
    for run_name in run_dirs:
        run_df, run_hypers = read_logs(path + run_name)
        print(run_name)
        columns = []
        for column in run_df.columns:
            if column.find(metric) != -1:
                columns.append(column)
        df[[run_name + '_' + c[-7:] for c in columns]] = run_df[columns]

    mean = df.mean(1).to_numpy()
    median = df.median(1).to_numpy()
    std = df.std(1).to_numpy()
    epochs = np.arange(0, len(mean))

    plt.figure(figsize=[9, 7])
    plt.plot(epochs, median, label='median value')
    plt.fill_between(epochs, mean-std, mean+std, label='standard deviation around mean', alpha=0.2)
    plt.title('Average Utility compared to BNE-Utility')
    plt.xlabel('epochs'); plt.ylabel('$u / u_{BNE}$')
    plt.hlines(1, 0, 4000, colors='grey', linestyles='solid', alpha=0.5)
    plt.xlim([0, epoch_lim])
    plt.legend(title='number of runs: {}'.format(len(run_dirs)), loc='lower right')
    plt.tight_layout()
    if save_fig:
        plt_path = path + 'plots'
        if not os.path.isdir(plt_path):
            os.makedirs(plt_path, exist_ok=False)
        plt.savefig(os.path.join(plt_path, '_util_plot.png'), dpi=256)
    else:
        plt.show()

def plot_MIUP22_v2(
        epoch_lim = 2000,
        logdir = '/home/kohring/bnelearn/experiments/expiriments_nils/' \
                 + 'MultiItemUniformPriceAuction/2players_2items/',
        logsubdir_bne_metrics = 'bne_metrics',
        run_name_conatains = ('20191202_09', '20191202_10', '20191202_11'),
        save_fig = True
    ):

    logdir = logdir[:logdir.rfind('/')]

    # read in saved data
    run_dirs = []
    for i in os.walk(logdir):
        if i[0].find(logsubdir_bne_metrics) != -1:
            run_dirs.append(i[0] + '/bne_metrics.npy')

    if run_name_conatains is not None and len(run_name_conatains) > 0:
        new_rundirs = []
        for selecter in run_name_conatains:
            [new_rundirs.append(r) for r in run_dirs if r.find(selecter) != -1]
        run_dirs = new_rundirs

    n_max_epochs = 10000
    epochs = 0
    data = np.zeros((n_max_epochs, len(2 * run_dirs)))
    for run_dir in run_dirs:
        idx_agent = 0
        with open(run_dir, 'rb') as f:
            i = 0
            while f:
                try:
                    loaded = np.load(f)
                    data[i,idx_agent] = loaded[1]
                    data[i,idx_agent+1] = loaded[3]
                    i += 1
                except ValueError:
                    break
            idx_agent += 2
        epochs = max(i, epochs)

    data = data[:epochs,:]
    n_runs = len(run_dirs)

    epochs = np.arange(epochs)
    mean = data.mean(1)
    median = np.median(data, 1)
    std = data.std(1)

    # plot config
    plt.figure(figsize=[7, 7])
    plt.plot(epochs, mean, label='mean value')
    plt.plot(epochs, median, label='median value')
    plt.fill_between(epochs, mean-std, mean+std, label='standard deviation around mean', alpha=0.2)
    plt.xlabel('epoch'); plt.ylabel('distance to BNE')
    plt.title('BNE Selection in Games with Coordination Problems')
    plt.legend(title='{} runs in total'.format(n_runs), loc='upper right')
    plt.xlim(0, epoch_lim); plt.ylim(0, 0.3)
    plt.tight_layout()
    if save_fig:
        plt_path = logdir + '/plots'
        if not os.path.isdir(plt_path):
            os.makedirs(plt_path, exist_ok=False)
        plt.savefig(os.path.join(plt_path, '_bne_plot.png'), dpi=256)

def plot_MIUP22(
        optimal_bid,
        optimal_bid_2,
        logdir,
        logsubdir_bne_metrics = 'bne_metrics',
        run_name_conatains = '201912',
        device = None
    ):

    current_logdir = logdir
    logdir = logdir[:logdir.rfind('/')]

    # read in saved data
    run_dirs = []
    for i in os.walk(logdir):
        if i[0].find(logsubdir_bne_metrics) != -1:
            run_dirs.append(i[0] + '/bne_metrics.npy')

    run_dirs = [r for r in run_dirs if r.find(run_name_conatains) != -1]

    # calc exact bnes
    bne_distance = policy_metric(optimal_bid, optimal_bid_2, 2, device=device)
    bne1 = [0, bne_distance]
    bne2 = [bne_distance, 0]

    # plot data
    fig = plt.figure(figsize=[7, 7])
    ax = fig.gca()
    for r, run_dir in zip(range(len(run_dirs)), run_dirs):
        data = np.zeros((10000, 5))
        with open(run_dir, 'rb') as f:
            i = 0
            while f:
                try:
                    data[i,:] = np.load(f)
                    i += 1
                except ValueError:
                    break
        epochs = i
        data = data[:epochs,:]

        try:
            plt.plot(data[:, 1], data[:, 3], color=colors[r],
                    label='run {} agent {}'.format(run_dir[len(logdir)+1:run_dir.find(logsubdir_bne_metrics)-1], 0))
            ax.arrow(data[-2, 1], data[-2, 3], data[-1, 1]-data[-2, 1], data[-1, 3]-data[-2, 3],
                    color=colors[r], width=0.00001, head_width=0.01)
            plt.plot(data[:, 2], data[:, 4], '--', color=colors[r],
                    label='run {} agent {}'.format(run_dir[len(logdir)+1:run_dir.find(logsubdir_bne_metrics)-1], 1))
            ax.arrow(data[-2, 2], data[-2, 4], data[-1, 2]-data[-2, 2], data[-1, 4]-data[-2, 4],
                    color=colors[r], width=0.00001, head_width=0.01)
        except Exception as e:
            print(e)

    plt.plot(bne1[0], bne1[1], 'x', color='black')
    ax.annotate('BNE1', [bne1[0]+0.01, bne1[1]+0.01])
    plt.plot(bne2[0], bne2[1], 'x', color='black')
    ax.annotate('BNE2', [bne2[0]+0.01, bne2[1]+0.01])
    plt.plot(0.28881958, 0.40845257, '.', color='black')
    ax.annotate('truthful', [0.28881958+0.01, 0.40845257+0.01])

    # plot config
    plt.xlabel('BNE1 metric'); plt.ylabel('BNE2 metric')
    plt.title('BNE Selection in Games with Coordination Problems')
    plt.legend(loc='upper right')
    lim = max(plt.xlim()[1], plt.ylim()[1])
    plt.xlim(-.05*lim, 1.3*lim); plt.ylim(-.05*lim, 1.3*lim)
    plt.gca().set_aspect('equal', adjustable='box')
    ax.grid(); plt.tight_layout()
    plt.savefig(os.path.join(current_logdir, 'plots', '_bne_plot.png'), dpi=256)

def plot_gradients(
        path: str = '/home/kohring/bnelearn/experiments/expiriments_nils/' \
            + 'FPSBSplitAwardAuction/2players_2items/'
    ):

    for run_dir in os.listdir(path):
        if run_dir.find('_gradients.csv') != -1:
            df = pd.read_csv(path + run_dir, index_col=0)
        elif run_dir.find('weight_norm.csv') != -1:
            df_base = pd.read_csv(path + run_dir, index_col=0)
    df = df / df_base

    mean = df.mean(axis=1).rolling(window=32).mean().values
    iters = np.arange(len(mean))

    plt.figure(figsize=[7, 4])
    plt.plot([-1], [-1], color='C1', label='actual values')
    for col in df.columns:
        plt.plot(iters, df[col], color='C1', alpha=.05)
    plt.plot(iters, mean, label='smoothed mean')

    plt.xlim([0, iters[-1]])
    plt.ylim([0, 8])
    plt.xlabel('iteration $i$')
    plt.ylabel('$||\\nabla_\\theta u_i||_2 / ||u_i||_2$')
    plt.grid()
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(os.path.join(path, 'gradient_plot_.pdf'))

def plot_saved_model(
        run_dir: str,
        mechanism,
        strat_to_bidder,
        model_dict: dict,
        param_dict: dict,
        batch_size = 100,
        format = 'png'
    ):
    """
    TODO
    """

    model_structure = NeuralNetStrategy(
        model_dict["input_length"],
        hidden_nodes = model_dict["hidden_nodes"],
        hidden_activations = model_dict["hidden_activations"],
        ensure_positive_output = None,
        output_length = param_dict["n_items"]
    ).to(device)
    model_structure.load_state_dict(torch.load(run_dir))

    plot_bid_function(
        [strat_to_bidder(model_structure, batch_size, 0)],
        optimal_bid(mechanism, param_dict, return_payoff_dominant=False),
        optimal_bid_2(mechanism, param_dict),
        None, run_dir[:run_dir.rfind('/')], None, e=-1,
        bounds = [param_dict["u_lo"], param_dict["u_hi"]],
        split_award = {
            'split_award': True,
            "efficiency_parameter": param_dict["efficiency_parameter"],
            "input_length": param_dict["input_length"] \
        } if param_dict["exp_no"] == 6 else None,
        save_fig_to_disc = True,
        format = format,
        device = device
    )



# ## Logging

def log_once(writer, e, max_epochs, n_players, log_name, n_parameters,
             seed, models, batch_size, learner_hyperparams, optimizer_type,
             optimizer_hyperparams, pretrain_epoch, env, experience=None):
    """Everything that should be logged only once on initialization."""
    for agent in range(len(models)):
        writer.add_scalar('hyperparameters/p{}_model_parameters'.format(agent),
                          n_parameters[agent], e)
    writer.add_scalar('hyperparameters/model_parameters', sum(n_parameters), e)

    for model in models:
        writer.add_text('hyperparameters/neural_net_spec', str(model), e)

    writer.add_scalar('hyperparameters/batch_size', batch_size, e)
    writer.add_scalar('hyperparameters/epochs', max_epochs, e)
    writer.add_scalar('hyperparameters/pretrain_epoch', pretrain_epoch, e)
    writer.add_scalar('hyperparameters/seed', seed, e)

    if experience is not None:
        for key, value in experience.items():
            try:
                writer.add_scalar('hyperparameters/experience_' + str(key), float(value), e)
            except:
                writer.add_text('hyperparameters/experience_' + str(key), value, e)

    for key, value in learner_hyperparams.items():
        writer.add_scalar('hyperparameters/' + str(key), value, e)

    writer.add_text('hyperparameters/optimizer', str(optimizer_type), e)
    for key, value in optimizer_hyperparams.items():
        writer.add_scalar('hyperparameters/' + str(key), value, e)
    # writer.add_graph(models[0], env.agents[0].valuations)

def log_metrics(writer, utilities, bne_utilities,
                against_bne_utilities, overhead,
                e, log_name, n_players, models,
                policy_metrics):
    """Log scalar for each player"""

    agent_name_list = ['agent_{}'.format(i) for i in range(n_players)]

    # log agents´ change in learning direction
    # writer.add_scalars('eval/learning_rate', dict(zip(agent_name_list, lr)), e)

    # # log agents´ learning rates
    # writer.add_scalars('eval/direction_change', dict(zip(agent_name_list, change_in_direction)), e)

    # log agents´ utilities
    writer.add_scalars('eval/utility', dict(zip(agent_name_list, utilities)), e)
    # writer.add_scalars('eval/utility_in_bne', dict(zip(agent_name_list, bne_utilities)), e)
    writer.add_scalars(
        'eval/utility_selfplay_vs_bne',
        dict(zip(
            agent_name_list,
            [1 - u/bne_u for u, bne_u in zip(utilities, bne_utilities)]
        )), e
    )
    writer.add_scalars(
        'eval/utility_against_bne',
        dict(zip(
            agent_name_list,
            [1 - u/bne_u for u, bne_u in zip(against_bne_utilities, bne_utilities)]
        )), e
    )
    # log agents´ welfare
    # writer.add_scalars('eval/welfare', dict(zip(agent_name_list, welfares)), e)
    # writer.add_scalars('eval/welfare_in_bne', dict(zip(agent_name_list, bne_welfare)), e)
    # writer.add_scalars(
    #     'eval/welfare_vs_bne',
    #     dict(zip(
    #         agent_name_list,
    #         [1 - w/bne_w for w, bne_w in zip(utilities, bne_welfare)]
    #     )), e
    # )

    # log models´ learning directions
    for name, policy_metric in policy_metrics.items():
        writer.add_scalars(
            'eval/distance_to_' + name,
            dict(zip(agent_name_list, policy_metric)),
            e
        )

    # # log agents´ allocations
    # for i in range(len(models)):
    #     writer.add_histogram('hist/allocations_' + agent_name_list[i], allocations[i], e)

    # log model parameters
    model_paras = [torch.norm(torch.nn.utils.parameters_to_vector(model.parameters()), p=2)
                   for model in models]
    writer.add_scalars('eval/weight_norm', dict(zip(agent_name_list, model_paras)), e)
    # writer.add_scalars('eval/gradient_norm', dict(zip(agent_name_list, gradient_norm)), e)

    # # log population rewards
    # for i in range(len(models)):
    #     writer.add_histogram('hist/rewards_' + agent_name_list[i], rewards[i], e)

    # # log stopping criterion
    # writer.add_scalars('eval/stopping', dict(zip(agent_name_list, stoppings)), e)

    # # log changes in output
    # writer.add_scalars('eval/changes_in_output', dict(zip(agent_name_list, changes_in_output)), e)

    # log time
    writer.add_scalar('eval/overhead_mins', overhead, e)

def tflog2pandas(
        path: str,
        use_tag = True
    ) -> pd.DataFrame:
    """
    Convert single tensorflow log file to pandas DataFrame

    Based on https://github.com/theRealSuperMario/supermariopy/blob/master/scripts/tflogs2pandas.py.

    Parameters
    ----------
    path : str
        path to tensorflow log file
    use_tag : bool
        could be used to cutoff tag prefix

    Returns
    -------
    pd.DataFrame
        converted dataframe
    """

    runlog_data = pd.DataFrame()
    hyperparameters = dict()

    DEFAULT_SIZE_GUIDANCE = {
        "compressedHistograms": 1,
        "images": 1,
        "scalars": 0, # 0 means to load all
        "histograms": 0,
    }
    event_acc = EventAccumulator(path, DEFAULT_SIZE_GUIDANCE)
    event_acc.Reload()
    tags = event_acc.Tags()

    if use_tag: # could be used to cutoff tag prefix
        idx = 1
    else:
        idx = 6

    # loading of texts
    for tag in tags['tensors']:
        texts = event_acc.Tensors(tag)
        for text in texts:
            try:
                opt_text = str(text.__getitem__(2).string_val)
                l_idx = opt_text.find('\'')
                r_idx = opt_text.rfind('\'')
                hyperparameters['optimizer'] = opt_text[l_idx+1:r_idx]
            except:
                pass
        # TODO other text and neural net spec could be loaded here

    # loading of histograms
    for tag in tags['histograms']:
        # loading only last hist epoch due to time
        histograms = event_acc.Histograms(tag)
        n_epoch = len(histograms)
        histogram = np.repeat(
            np.array(histograms[-1].histogram_value.bucket_limit),
            np.array(histograms[-1].histogram_value.bucket).astype(np.int)
        )
        hist_values, hist_count = np.unique(
            histogram.astype(np.int),
            return_counts = True
        )
        hist_log = np.zeros((n_epoch, len(hist_values)))
        hist_log[-1,:] = hist_count / sum(hist_count)
        for i, hist_val in enumerate(hist_values):
            runlog_data[tag + '_' + str(hist_val)] = hist_log[:,i]

    # loading of scalars
    for tag in tags["scalars"]:
        event_list = event_acc.Scalars(tag)
        values = list(map(lambda x: x.value, event_list))
        step = list(map(lambda x: x.step, event_list))
        if len(step) > 1:
            runlog_data[tag] = values
        else:
            hyperparameters[tag] = values[0]

    return runlog_data, hyperparameters

def read_logs(path: str, save=False):
    """ read in saved data

        returns
            df: DataFrame, of all logged data each epoch.
            hyperparameters: dict, of all hyperparameters just logged once.
    """
    df = pd.DataFrame()
    hyperparameters = dict()
    for i in os.walk(path):
        try:
            if i[0].find('agent') == -1:
                use_tag = False
            else:
                use_tag = True
            new_data, new_hypers = tflog2pandas(i[0], use_tag)
            if df.empty:
                df = new_data
            else:
                df = df.join(new_data)
            hyperparameters = {**hyperparameters, **new_hypers}
        except:
            pass

    if save:
        df.to_csv(path + '/_data.csv')

    return df, hyperparameters

def log_evaulation(
        path: str = '/home/kohring/bnelearn/experiments/expiriments_nils/' \
                    + 'FPSBSplitAwardAuction/2players_2items/',
        save = True
    ):
    """
        Collects all logs in subdirs of ´path´ and saves last epochs of them
        in a csv-file.
    """
    df = pd.DataFrame()
    for i, run_dir in enumerate(os.listdir(path)):

        if not os.path.isfile(run_dir):
            print(run_dir)
            data, hypers = read_logs(path + run_dir)

            n_epoch = data.shape[0] - 1
            hypers['computed_epochs'] = n_epoch
            data = data.tail(1)

            run_dict = {'run_id': run_dir, **hypers, **data.to_dict(orient='list')}

            if len(run_dict) > 1:
                for k, v in run_dict.items():
                    if isinstance(v, list):
                        run_dict[k] = v[0]

                # add new columns if necessary
                known_keys = df.columns
                for column in run_dict.keys():
                    if column not in known_keys:
                        df[column] = [None] * df.shape[0]
                df.loc[i] = pd.Series(run_dict)

    df.apply(pd.to_numeric, errors='ignore')

    if path.find('FPSBSplitAwardAuction/2players_2items') != -1:
        eps = .16
        def bne(row):
            try:
                if row['hist/allocations_agent_0_1'] < eps:
                    return 'wta'
                else:
                    return 'pool'
            except:
                pass
            return
        df['converged_bne'] = df.apply(bne, axis=1)

        # Calculate distance for each dim
        mechanism, param_dict, split_award_dict = create_splitaward_setting()

        bne_pool = optimal_bid(mechanism, param_dict, return_payoff_dominant=True)
        bne_wta = optimal_bid_2(mechanism, param_dict)

        def bne_dist(dim, bne):
            def temp(row):
                try:
                    return policy_metric(
                        load_model(path + row['run_id']).forward, bne,
                        param_dict["n_items"],
                        selection = split_award_dict \
                            if param_dict["exp_no"] == 6 else 'random',
                        bounds = [param_dict["u_lo"], param_dict["u_hi"]],
                        eval_points_max = 2 ** 18,
                        dim_of_interest = dim,
                        device = 'cuda'
                    )
                except:
                    return
            return temp
        df['dist_pool_50'] = df.apply(bne_dist(1, bne_pool), axis=1)
        df['dist_wta_100'] = df.apply(bne_dist(0, bne_wta), axis=1)

        def dist_to_converged_bne_component(row):
            if row['converged_bne'] == 'wta':
                return row['dist_wta_100']
            elif row['converged_bne'] == 'pool':
                return row['dist_pool_50']
        df['dist_conv_bne'] = df.apply(dist_to_converged_bne_component, axis=1)

    t = str(time.strftime('%Y%m%d_%H%M%S', time.gmtime()))
    if save:
        df.to_csv(path + '/' + t + '_analysis.csv')
    return df

def read_eval(
        path: str = '/home/kohring/bnelearn/experiments/expiriments_nils/' \
            + 'FPSBSplitAwardAuction/2players_2items/',
        feature = 'eval/gradient_norm',
        save = True
    ):
    """
    Reads one feature ´feature´ over all iterations of all runs in
    directory ´pass´.
    """
    df = pd.DataFrame()
    for i, run_dir in enumerate(os.listdir(path)):
        if not os.path.isfile(run_dir):
            print(run_dir)
            data, _ = read_logs(path + run_dir)
            try:
                df[run_dir] = data[feature]
            except Exception as e:
                print(e)
                pass
    if save:
        t = str(time.strftime('%Y%m%d_%H%M%S', time.gmtime()))
        df.to_csv(path + t + feature + '.csv')
    return df



# ## Help for logging

def angle(a: torch.Tensor, b: torch.Tensor) -> float:
    """
    Calculate the angle between tensors ´a´ and ´b´.
    """

    assert a.squeeze_().shape == b.squeeze_().shape, \
        'dimensions missmatch'

    a = a.detach().cpu()
    b = b.detach().cpu()

    return np.arccos(sum(a * b) / (torch.norm(a) * torch.norm(b)))

def stopping_criterion(gradient_vectors: torch.Tensor) -> float:
    """
    As in Mahsereci, M. et. al, Early Stopping without a Validation Set, 2017.

    Parameters
    ----------
        gradient_vectors: torch.Tensor of shape (batch size, number of parameters)
    """
    gradient_var = gradient_vectors.var(dim=0)
    return 1 - (gradient_vectors.shape[0] / gradient_vectors.shape[1]) \
        * sum((gradient_vectors.mean(dim=0) ** 2) / gradient_var)

def load_model(
        path,
        i = 0,
        model_dict = {
            "input_length": 1,
            "hidden_nodes": [5, 5, 5],
            "hidden_activations": [nn.SELU(), nn.SELU(), nn.SELU()]
        },
        n_items = 2
    ):
    """
    Loads a NeuralNetStrategy from ´path´.
    """
    model_structure = NeuralNetStrategy(
        model_dict["input_length"],
        hidden_nodes = model_dict["hidden_nodes"],
        hidden_activations = model_dict["hidden_activations"],
        ensure_positive_output = None,
        output_length = n_items
    ).to(device)
    try:
        model_structure.load_state_dict(
            torch.load(os.path.join(path, 'saved_model_' + str(i) + '.pt'))
        )
    except Exception as e:
        # print('\n', path ,'\n', e, '\n')
        model_structure = None
    return model_structure

def compare_models(
        path: str = '/home/kohring/bnelearn/experiments/expiriments_nils/' \
                    + 'MultiItemUniformPriceAuction/2players_2items/',
        batch_size = 2 ** 18,
        model_dict = {
            "input_length": 2,
            "hidden_nodes": [5, 5, 5],
            "hidden_activations": [nn.SELU(), nn.SELU(), nn.SELU()]
        },
        # param_dict = {
        #     "n_items": 2, "exp_no": 6,
        #     "u_lo": 1.0, "u_hi": 1.4
        # },
        # split_award_dict = {},
        device = None,
        save = False,
    ):
    """
    Crowls through all runs in ´path´ and returns a list of names and a matrix consisting
    of the difference in polixy space between all agents.
    """

    names, models = list(), list()

    param_dict = dict()
    if path.find('FPSBSplitAwardAuction/2players_2items/') != -1:
        param_dict["exp_no"] = 6
        mechanism = FPSBSplitAwardAuction(cuda=True)
        param_dict["n_players"] = 2
        param_dict["n_items"] = 2
        param_dict["u_lo"] = 1.0
        param_dict["u_hi"] = 1.4
        param_dict["efficiency_parameter"] = 0.3
        param_dict["BNE1"] = "PD_Sigma_BNE"
        param_dict["BNE2"] = "WTA_BNE"
        param_dict["input_length"] = param_dict["n_items"] - 1 \
            if param_dict["exp_no"] == 6 else param_dict["n_items"]
        split_award_dict = {
            'split_award': True,
            'efficiency_parameter': param_dict["efficiency_parameter"],
            'input_length': param_dict["input_length"],
            'linspace': False
        }
        def strat_to_bidder(strategy, batch_size, player_position):
            return Bidder.uniform(
                lower = param_dict["u_lo"], upper = param_dict["u_hi"],
                strategy = strategy,
                n_items = param_dict["n_items"],
                player_position = player_position,
                split_award = isinstance(mechanism, FPSBSplitAwardAuction),
                efficiency_parameter = param_dict["efficiency_parameter"] \
                    if "efficiency_parameter" in param_dict.keys() else None,
                batch_size = batch_size
            )

        bne_pool = optimal_bid(mechanism, param_dict)
        bne_wta = optimal_bid_2(mechanism, param_dict)
    elif path.find('MultiItemUniformPriceAuction/2players_2items/') != -1:
        param_dict["exp_no"] = 1
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

    # df = pd.read_csv('experiments/expiriments_nils/FPSBSplitAwardAuction' +\
    #     '/2players_2items/final_splitaward.csv', index_col=0)
    # wta_models = list()
    # pool_models = list()
    for run_dir in os.listdir(path):
        i = 0 # model idx in that path
        while True:
            model = load_model(os.path.join(path, run_dir), i, model_dict, param_dict['n_items'])
            if model is not None:
                print(os.path.join(path, run_dir))
                names.append(run_dir + '_model_' + str(i))
                models.append(model)

                # if str(df[df['run_id'] == run_dir]['converged_bne'].values[0]) == 'pool':
                #     pool_models.append(model)
                # else:
                #     wta_models.append(model)
            else:
                break
            i += 1
    # print(len(wta_models))
    # print(len(pool_models))
    # for i, model_list in enumerate([wta_models, pool_models]):
    plot_bid_function(
        [strat_to_bidder(model, batch_size, 0) for model in models],
        optimal_bid(mechanism, param_dict, return_payoff_dominant=False),
        optimal_bid_2(mechanism, param_dict),
        None, path, None, e=-(i+1),
        bounds = [param_dict["u_lo"], param_dict["u_hi"]],
        split_award = {
            'split_award': True,
            "efficiency_parameter": param_dict["efficiency_parameter"],
            "input_length": param_dict["input_length"] \
        } if param_dict["exp_no"] == 6 else None,
        save_fig_to_disc = True,
        format = 'png',
        device = device
    )
    n_models = len(models)
    results = np.zeros((n_models, n_models))

    # for i, m0 in enumerate(models):
    #     for j, m1 in enumerate(models):
    #         results[i,j] = policy_metric(
    #             m0, m1,
    #             model_dict["input_length"],
    #             selection = split_award_dict \
    #                 if param_dict["exp_no"] == 6 else 'random',
    #             bounds = [param_dict["u_lo"], param_dict["u_hi"]],
    #             eval_points_max = batch_size,
    #             device = device
    #         )

    # single_bid_diffs = np.zeros((param_dict["n_items"], n_models, n_models))
    # for dim in range(param_dict["n_items"]):
    #     for i, m0 in enumerate(models[:2]):
    #         print(names[i])
    #         for j, m1 in enumerate(models):
    #             print('   ', names[j])
    #             single_bid_diffs[dim,i,j] = policy_metric(
    #                 m0, m1,
    #                 param_dict["n_items"],
    #                 selection = split_award_dict \
    #                     if param_dict["exp_no"] == 6 else 'random',
    #                 bounds = [param_dict["u_lo"], param_dict["u_hi"]],
    #                 eval_points_max = batch_size,
    #                 dim_of_interest = dim,
    #                 device = device
    #             )

    if save:
        # df = pd.DataFrame(results, columns=names, index=names)
        # df.to_csv(path + 'compared_models.csv')

        for dim in range(param_dict["n_items"]):
            df = pd.DataFrame(single_bid_diffs[dim,:,:], columns=names, index=names)
            df.to_csv(path + 'compared_models_dim_' + str(dim) + '.csv')

    return (names, results)

def summary_stats_splitaward(
        path: str = '/home/kohring/bnelearn/experiments/expiriments_nils/' \
            + 'FPSBSplitAwardAuction/2players_2items/',
        file_path: str = '/home/kohring/bnelearn/experiments/expiriments_nils/' \
            + 'FPSBSplitAwardAuction/2players_2items/20200319_100209_analysis.csv',
        model_dict = {
            "input_length": 1,
            "hidden_nodes": [5, 5, 5],
            "hidden_activations": [nn.SELU(), nn.SELU(), nn.SELU()]
        },
        batch_size = 2**19
    ):
    df = pd.read_csv(file_path)

    # set up env and load models
    mechanism, param_dict, split_award_dict = create_splitaward_setting()
    names, models = list(), list()
    for run_dir in os.listdir(path):
        i = 0 # model idx in that path
        while True:
            model = load_model(os.path.join(path, run_dir), i, model_dict, param_dict['n_items'])
            if model is not None:
                names.append(run_dir)
                models.append(model)
            else:
                break
            i += 1
    def strat_to_bidder(strategy, batch_size, player_position, adversarial=False):
        return Bidder.uniform(
            lower = param_dict["u_lo"], upper = param_dict["u_hi"],
            strategy = strategy,
            n_items = param_dict["n_items"],
            player_position = player_position,
            split_award = isinstance(mechanism, FPSBSplitAwardAuction),
            efficiency_parameter = param_dict["efficiency_parameter"] \
                if "efficiency_parameter" in param_dict.keys() else None,
            batch_size = batch_size
        )

    pool_strategies = [
        ClosureStrategy(
            partial(
                optimal_bid(
                    mechanism, param_dict,
                    return_payoff_dominant = True
                ),
                player_position = i
            )
        )
        for i in range(param_dict["n_players"])
    ]
    wta_strategies = [
        ClosureStrategy(
            partial(
                optimal_bid_2(mechanism, param_dict),
                player_position = i
            )
        )
        for i in range(param_dict["n_players"])
    ]
    pool_env = AuctionEnvironment(
        mechanism,
        agents = [
            strat_to_bidder(bne_strategy, batch_size, i)
            for i, bne_strategy in enumerate(pool_strategies)
        ],
        n_players = param_dict["n_players"],
        batch_size = batch_size,
        strategy_to_player_closure = strat_to_bidder,
    )
    wta_env = AuctionEnvironment(
        mechanism,
        agents = [
            strat_to_bidder(bne_strategy, batch_size, i)
            for i, bne_strategy in enumerate(wta_strategies)
        ],
        n_players = param_dict["n_players"],
        batch_size = batch_size,
        strategy_to_player_closure = strat_to_bidder,
    )
    utility_in_pool = pool_env.get_strategy_reward(
        pool_strategies[0], player_position=0, draw_valuations=True
    ).cpu().numpy()
    utility_in_wta = wta_env.get_strategy_reward(
        wta_strategies[0], player_position=0, draw_valuations=True
    ).cpu().numpy()

    print('utility_in_pool', utility_in_pool)
    print('utility_in_wta', utility_in_wta)

    metrics = ['utility_in_selfplay', 'utility_vs_bne', 'relative_loss', 'rmse']
    def utility_in_selfplay(row):
        model_name = row['run_id']
        try:
            model = models[names.index(model_name)]
            return AuctionEnvironment(
                mechanism,
                agents = [
                    strat_to_bidder(model, batch_size, i)
                    for _ in range(param_dict["n_players"])
                ],
                n_players = param_dict["n_players"],
                batch_size = batch_size,
                strategy_to_player_closure = strat_to_bidder,
            ).get_strategy_reward(
                model, player_position=0, draw_valuations=True
            ).cpu().numpy()
        except:
            return
    def utility_vs_bne(row):
        model_name = row['run_id']
        try:
            model = models[names.index(model_name)]
            if row['converged_bne'] == 'pool':
                env = pool_env
            elif row['converged_bne'] == 'wta':
                env = wta_env
            return env.get_strategy_reward(
                model, player_position=0, draw_valuations=True
            ).cpu().numpy()
        except:
            return
    def relative_loss(row):
        try:
            if row['converged_bne'] == 'pool':
                utility_in_bne = utility_in_pool
            elif row['converged_bne'] == 'wta':
                utility_in_bne = utility_in_wta
            return 1 - float(row['utility_vs_bne']) / utility_in_bne
        except:
            return
    def rmse(row):
        model_name = row['run_id']
        try:
            model = models[names.index(model_name)]
            if row['converged_bne'] == 'pool':
                bne_strat = pool_strategies[0].play
                dim = 1
            elif row['converged_bne'] == 'wta':
                bne_strat = wta_strategies[0].play
                dim = 0
            return policy_metric(
                model, bne_strat,
                param_dict["n_items"],
                selection = split_award_dict \
                    if param_dict["exp_no"] == 6 else 'random',
                bounds = [param_dict["u_lo"], param_dict["u_hi"]],
                eval_points_max = batch_size,
                dim_of_interest = dim,
                device = device
            )
        except:
            return

    for metric in metrics:
        df[metric] = df.apply(eval(metric), axis=1)
    df.to_csv(path + '_analysis_.csv')
    df.apply(pd.to_numeric, errors='ignore')

    summary = pd.DataFrame(
        index = [
            'pool_mean', 'pool_std',
            'wta_mean','wta_std',
            'total_mean', 'total_std'
        ]
    )
    for metric in metrics:
        pm = df[df['converged_bne'] == 'pool'][metric].mean()
        ps = df[df['converged_bne'] == 'pool'][metric].std()
        wm = df[df['converged_bne'] == 'wta'][metric].mean()
        ws = df[df['converged_bne'] == 'wta'][metric].std()
        tm = np.nanmean(np.array(df[metric].values, dtype=float))
        ts = np.nanstd(np.array(df[metric].values, dtype=float))
        summary[metric] = [pm, ps, wm, ws, tm, ts]

    summary.to_csv(path + '_summary_analysis_.csv')

def summary_stats_discirminatory(
        path: str = '/home/kohring/bnelearn/experiments/expiriments_nils/' \
            + 'MultiItemDiscriminatoryAuction/2players_2items/',
        file_path: str = '/home/kohring/bnelearn/experiments/expiriments_nils/' \
            + 'MultiItemDiscriminatoryAuction/2players_2items/20200316_082911_analysis.csv',
        model_dict = {
            "input_length": 2,
            "hidden_nodes": [5, 5, 5],
            "hidden_activations": [nn.SELU(), nn.SELU(), nn.SELU()]
        },
        batch_size = 2**19
    ):
    df = pd.read_csv(file_path)

    # set up env and load models
    mechanism, param_dict = create_discriminatory_setting()
    names, models = list(), list()
    for run_dir in os.listdir(path):
        i = 0 # model idx in that path
        while True:
            model = load_model(os.path.join(path, run_dir), i, model_dict, param_dict['n_items'])
            if model is not None:
                names.append(run_dir)
                models.append(model)
            else:
                break
            i += 1

    def strat_to_bidder(strategy, batch_size, player_position):
        return Bidder.uniform(
            lower = param_dict["u_lo"], upper = param_dict["u_hi"],
            strategy = strategy,
            descending_valuations = True,
            n_items = param_dict["n_items"],
            player_position = player_position,
            batch_size = batch_size
        )

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
        strategy_to_player_closure = strat_to_bidder,
    )

    utility_in_bne = bne_env.get_reward(
        bne_env.agents[0], draw_valuations=True
    ).cpu().numpy()

    print('utility_in_bne', utility_in_bne)

    metrics = ['utility_in_selfplay', 'utility_vs_bne', 'relative_loss', 'rmse']
    def utility_in_selfplay(row):
        model_name = row['run_id']
        try:
            model = models[names.index(model_name)]
            return AuctionEnvironment(
                mechanism,
                agents = [
                    strat_to_bidder(model, batch_size, i)
                    for _ in range(param_dict["n_players"])
                ],
                n_players = param_dict["n_players"],
                batch_size = batch_size,
                strategy_to_player_closure = strat_to_bidder,
            ).get_strategy_reward(
                model, player_position=0, draw_valuations=True
            ).detach().cpu().numpy()
        except Exception as e:
            print(e)
            return
    def utility_vs_bne(row):
        model_name = row['run_id']
        try:
            model = models[names.index(model_name)]
            return bne_env.get_strategy_reward(
                model, player_position=0, draw_valuations=True
            ).detach().cpu().numpy()
        except Exception as e:
            print(e)
            return
    def relative_loss(row):
        try:
            return 1 - float(row['utility_vs_bne']) / utility_in_bne
        except Exception as e:
            print(e)
            return
    def rmse(row):
        model_name = row['run_id']
        try:
            model = models[names.index(model_name)]
            return policy_metric(
                model, bne_strategies[0].play,
                param_dict["n_items"],
                selection = 'random',
                bounds = [param_dict["u_lo"], param_dict["u_hi"]],
                eval_points_max = batch_size,
                device = device
            )
        except Exception as e:
            print(e)
            return

    for metric in metrics:
        df[metric] = df.apply(eval(metric), axis=1)
    # df.to_csv(path + '_analysis_.csv')
    df.apply(pd.to_numeric, errors='ignore')

    summary = pd.DataFrame(index=['mean', 'std'])
    for metric in metrics:
        m = df[metric].mean()
        s = df[metric].std()
        summary[metric] = [m, s]

    summary.to_csv(path + '_summary_analysis_.csv')



if __name__ == '__main__':
    torch.cuda.set_device(4)
    device = 'cuda'

    # log_evaulation()
    # summary_stats_discirminatory()
    # summary_stats_splitaward()
    compare_models(device=device)
    # plot_gradients()

    # ## MultiItemUniformPriceAuction
    if False:
        def bne1(v):
            b = v.clone().detach()
            b[:,1] = 0
            return b
        def bneC(v, c=0.5):
            b = c * torch.ones_like(v, device=device)
            b[:,1] = 0
            return b
        def random(v):
            b = torch.zeros_like(v, device=device).uniform_(0, 1)
            b = b.sort(dim=1, descending=True)[0]
            return b

        policies = [
            lambda v: torch.zeros_like(v, device=device),
            lambda v: v,
            bne1,
            lambda v: 0.5 * torch.ones_like(v, device=device),
            lambda v: bneC(v, 0.5),
            lambda v: bneC(v, 1),
            lambda v: torch.ones_like(v, device=device),
            random 
        ]
        policy_names = [
            'zeros',
            'truthful',
            'bne1',
            'half',
            'bne0.5',
            'bne2',
            'ones',
            'random'
        ]

        A = meta_game(MultiItemUniformPriceAuction(device), policies, policy_names,
                      device=device)

        best_policy = replicator_dynamics(A - np.diag(A), policy_names, verbose=True)

    # ## FPSBSplitAwardAuction
    if False:
        mechanism, param_dict, split_award_dict = create_splitaward_setting()

        def bne_sigma_lo(v):
            b = torch.zeros_like(v, device=device)
            b[:,0] = optimal_bid_2(mechanism, param_dict)(v)['sigma_bounds'][:,0]
            b[:,1] = optimal_bid_2(mechanism, param_dict)(v)['wta_bounds'][:,0]
            return b
        def bne_sigma_hi(v):
            b = torch.zeros_like(v, device=device).uniform_(0, 1)
            b[:,0] = optimal_bid_2(mechanism, param_dict)(v)['sigma_bounds'][:,1]
            b[:,1] = optimal_bid_2(mechanism, param_dict)(v)['wta_bounds'][:,1]
            return b
        def bne_sigma_me(v):
            b = torch.zeros_like(v, device=device).uniform_(0, 1)
            b[:,0] = optimal_bid_2(mechanism, param_dict)(v)['sigma_bounds'][:,1]
            b[:,1] = optimal_bid_2(mechanism, param_dict)(v)['wta_bounds'][:,1]
            return  0.5 * (bne_sigma_lo(v) + bne_sigma_hi(v))
        def bne_wta(v):
            return optimal_bid(mechanism, param_dict)(v)
        def random(v):
            b = torch.zeros_like(v, device=device).uniform_(0, 1)
            return b

        policies = [
            lambda v: torch.zeros_like(v, device=device),
            lambda v: v,
            bne_sigma_lo,
            bne_sigma_me,
            bne_sigma_hi,
            bne_wta,
            random,
            lambda v: torch.ones_like(v, device=device)
        ]
        policy_names = [
            'zeros',
            'truthful',
            'bne_s_lo',
            'bne_s_me',
            'bne_s_hi',
            'bne_wta',
            'random',
            'ones',
        ]

        A = meta_game(mechanism, policies, policy_names, policy_of_interest=2,
                      device=device, split_award=split_award_dict)

        best_policy = replicator_dynamics(A - np.diag(A), policy_names, verbose=True)
