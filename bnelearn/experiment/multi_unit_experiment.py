"""
In this file multi-unit experiments and their analytical BNEs (if known) are defiened.

TODO:
    - support for multiple BNE
    - valuations of plotiing BNE _optimal_bid_multidiscriminatory2x2 are wrong
    - base FPSBSplitAwardAuction2x2 on MultiUnitExperiment
"""

import os
import time
import warnings
from abc import ABC
from itertools import product
import bnelearn.util.metrics as metrics


import torch
import numpy as np
from scipy import integrate, interpolate

from bnelearn.bidder import Bidder, ReverseBidder
from bnelearn.environment import AuctionEnvironment
from bnelearn.experiment import GPUController, Experiment
from bnelearn.experiment.configurations import ExperimentConfiguration, LearningConfiguration, LoggingConfiguration
from bnelearn.learner import ESPGLearner
from bnelearn.mechanism import (
    MultiUnitVickreyAuction, MultiUnitUniformPriceAuction, MultiUnitDiscriminatoryAuction,
    FPSBSplitAwardAuction, Mechanism
)

from bnelearn.strategy import NeuralNetStrategy, ClosureStrategy
from bnelearn.util import metrics as metrics

import matplotlib.pyplot as plt
from timeit import default_timer as timer

from torch.utils.tensorboard import SummaryWriter

import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import LinearLocator, FormatStrFormatter


########################################################################################################################
###                                                 BNE STRATEGIES                                                   ###
########################################################################################################################
def multiunit_bne(experiment_config, payment_rule):
    """
    Method that returns the known BNE strategy as callable if available and None otherwise.
    """

    if payment_rule == 'vcg':
        def truthful(valuation, player_position=None):
            return torch.clone(valuation)
        return truthful

    elif payment_rule == 'discriminatory':
        if experiment_config.n_units == 2 and experiment_config.n_players == 2:
            if not experiment_config.constant_marginal_values:
                print('BNE is only approximated roughly!')
                return _optimal_bid_multidiscriminatory2x2
            else:
                # TODO get valuation_cdf from experiment_config
                raise NotImplementedError
                # return _optimal_bid_multidiscriminatory2x2CMV(valuation_cdf)

    elif payment_rule == 'uniform':
        if experiment_config.n_units == 2 and experiment_config.n_players == 2:
            return _optimal_bid_multiuniform2x2
        elif (experiment_config.n_units == 3 and experiment_config.n_players == 2 and experiment_config.item_interest_limit == 2):
            return _optimal_bid_multiuniform3x2limit2

    return None

def _optimal_bid_multidiscriminatory2x2(valuation, player_position=None):

    def b_approx(v, s, t):
        b = torch.clone(v)
        lin_e = np.array([[1, 1, 1], [2 * t, 1, 0], [t ** 2, t, 1]])
        lin_s = np.array([0.47, s / t, s])
        x = np.linalg.solve(lin_e, lin_s)
        b[v < t] *= s / t
        b[v >= t] = x[0] * b[v >= t] ** 2 + x[1] * b[v >= t] + x[2]
        return b

    b1 = lambda v: b_approx(v, s=0.42, t=0.90)
    b2 = lambda v: b_approx(v, s=0.35, t=0.55)

    opt_bid = valuation
    opt_bid[:,0] = b1(opt_bid[:,0])
    opt_bid[:,1] = b2(opt_bid[:,1])
    opt_bid = opt_bid.sort(dim=1, descending=True)[0]
    return opt_bid

def _optimal_bid_multidiscriminatory2x2CMV(valuation_cdf):

    if isinstance(valuation_cdf, torch.distributions.uniform.Uniform):
        def _optimal_bid(valuation, player_position=None):
            return valuation / 2

    elif isinstance(valuation_cdf, torch.distributions.normal.Normal):

        def muda_tb_cmv_bne(
                value_pdf: callable,
                value_cdf: callable = None,
                lower_bound: int = 0,
                epsabs=1e-3
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

        dist = self._strat_to_bidder(None, 1, None).value_distribution
        bidding = muda_tb_cmv_bne(lambda x: torch.exp(dist.log_prob(x)).cpu().numpy(),
                                  lambda x: dist.cdf(x).cpu().numpy())

        def _optimal_bid(valuation, player_position=None): 
            opt_bid = np.zeros_like(valuation.cpu().numpy())
            for agent in range(self.n_players):
                opt_bid[agent] = bidding(valuation[agent,:])
            return torch.tensor(opt_bid)

    return _optimal_bid

def _optimal_bid_multiuniform2x2(valuation, player_position=None):
    opt_bid = torch.clone(valuation)
    opt_bid[:,1] = 0
    return opt_bid

def _optimal_bid_multiuniform3x2limit2(valuation, player_position=None):
    opt_bid = torch.clone(valuation)
    opt_bid[:,1] = opt_bid[:, 1] ** 2
    opt_bid[:,2] = 0
    return opt_bid

def _optimal_bid_splitaward2x2_1(experiment_config):
    """Pooling equilibrium as in Anton and Yao, 1992."""

    efficiency_parameter = experiment_config.efficiency_parameter
    u_lo = experiment_config.u_lo
    u_hi = experiment_config.u_hi

    value_cdf = torch.distributions.Uniform(u_lo[0], u_hi[0]).cdf

    def _optimal_bid(valuation, player_position=None, return_payoff_dominant=True):
        sigma_bounds = torch.ones_like(valuation, device=valuation.device)
        sigma_bounds[:,0] = efficiency_parameter * u_hi[0]
        sigma_bounds[:,1] = (1 - efficiency_parameter) * u_lo[0]
        # [:,0]: lower bound and [:,1] upper

        _p_sigma = (1 - efficiency_parameter) * u_lo[0] # highest possible p_sigma

        def G(theta):
            return _p_sigma + (_p_sigma - u_hi[0]*efficiency_parameter * value_cdf(theta)) \
                / (1 - value_cdf(theta))

        wta_bounds = 2 * sigma_bounds
        wta_bounds[:,1] = G(valuation[:,0])

        if return_payoff_dominant:
            return torch.cat(
                (wta_bounds[:,1].unsqueeze(0).t_(),
                    sigma_bounds[:,1].unsqueeze(0).t_()),
                axis=1
            )
        return {'sigma_bounds': sigma_bounds, 'wta_bounds': wta_bounds}

    return _optimal_bid

def _optimal_bid_splitaward2x2_2(experiment_config):
    """WTA equilibrium as in Anton and Yao, 1992: Proposition 4."""

    efficiency_parameter = experiment_config.efficiency_parameter
    u_lo = experiment_config.u_lo
    u_hi = experiment_config.u_hi
    n_players = experiment_config.n_players

    def value_cdf(value):
        value = np.array(value)
        result = (value - u_lo[0]) / (u_hi[0] - u_lo[0])
        return result.clip(0, 1)

    # CONSTANTS
    opt_bid_batch_size = 2 ** 12
    eps = 1e-4

    opt_bid = np.zeros((opt_bid_batch_size, experiment_config.n_units))

    # do one-time approximation via integration
    val_lin = np.linspace(u_lo[0], u_hi[0]-eps, opt_bid_batch_size)

    def integral(theta):
        return np.array(
                [integrate.quad(
                    lambda x: (1 - value_cdf(x))**(n_players - 1), v, u_hi[0],
                    epsabs = eps
                )[0] for v in theta]
            )

    def opt_bid_100(theta):
        return theta + (integral(theta) / (
                (1 - value_cdf(theta))**(n_players - 1))
            )

    opt_bid[:,0] = opt_bid_100(val_lin)
    opt_bid[:,1] = opt_bid_100(val_lin) - efficiency_parameter * u_lo[0] # or more

    opt_bid_function = [
        interpolate.interp1d(val_lin, opt_bid[:,0], fill_value='extrapolate'),
        interpolate.interp1d(val_lin, opt_bid[:,1], fill_value='extrapolate')
    ]

    # use interpolation of opt_bid done on first batch
    def _optimal_bid(valuation, player_position=None):
        bid = torch.tensor([
                opt_bid_function[0](valuation[:,0].cpu().numpy()),
                opt_bid_function[1](valuation[:,0].cpu().numpy())
            ],
            device = valuation.device
        ).t_()
        bid[bid < 0] = 0
        bid[torch.bid(opt_bid)] = 0
        return bid

    return _optimal_bid

########################################################################################################################


class MultiUnitExperiment(Experiment, ABC):
    """
    Experiment for the standard multi-unit auctions.
    """
    def __init__(self, experiment_config: ExperimentConfiguration, learning_config: LearningConfiguration,
                 logging_config: LoggingConfiguration, gpu_config: GPUController):

        # check for available BNE strategy
        if not isinstance(self, SplitAwardExperiment):
            self._optimal_bid = multiunit_bne(experiment_config, experiment_config.payment_rule)
        else:
            if experiment_config.n_units == 2 and experiment_config.n_players == 2:
                self._optimal_bid = _optimal_bid_splitaward2x2_1(experiment_config)
                self._optimal_bid_2 = _optimal_bid_splitaward2x2_2(experiment_config) # TODO unused
            else:
                self._optimal_bid = None
        known_bne = self._optimal_bid is not None

        super().__init__(experiment_config, learning_config, logging_config, gpu_config, known_bne)
        self.n_units = self.n_items = experiment_config.n_units

        self.u_lo = experiment_config.u_lo
        self.u_hi = experiment_config.u_hi
        self.plot_xmin = self.plot_ymin = min(self.u_lo)
        self.plot_xmax = self.plot_ymax = max(self.u_hi)
        self.model_sharing = experiment_config.model_sharing

        if self.payment_rule in ('discriminatory', 'first_price'):
            self.mechanism_type = MultiUnitDiscriminatoryAuction
        elif self.payment_rule in ('vcg', 'second_price'):
            self.mechanism_type = MultiUnitVickreyAuction
        elif self.payment_rule == 'uniform':
            self.mechanism_type = MultiUnitUniformPriceAuction
        else:
            raise ValueError('payment rule unknown')

        if self.model_sharing:
            self.n_models = 1
            self._bidder2model = [0] * self.n_players
        else:
            self.n_models = self.n_players
            self._bidder2model = list(range(self.n_players))

        self.constant_marginal_values = experiment_config.constant_marginal_values
        self.item_interest_limit = experiment_config.item_interest_limit

        if experiment_config.pretrain_transform is not None:
            self.pretrain_transform = experiment_config.pretrain_transform
        else:
            self.pretrain_transform = self.default_pretrain_transform

        self.input_length =  experiment_config.input_length

        print('\n=== Hyperparameters ===')
        for k in learning_config.learner_hyperparams.keys():
            print('{}: {}'.format(k, learning_config.learner_hyperparams[k]))
        print('=======================\n')
        self._setup_mechanism_and_eval_environment()

    def _strat_to_bidder(self, strategy, batch_size, player_position=0, cache_actions=False):
        """
        Standard strat_to_bidder method.
        """
        return Bidder.uniform(
            lower=self.u_lo[player_position], upper=self.u_hi[player_position],
            strategy=strategy,
            n_items=self.n_units,
            item_interest_limit=self.item_interest_limit,
            descending_valuations=True,
            constant_marginal_values=self.constant_marginal_values,
            player_position=player_position,
            batch_size=batch_size,
            cache_actions=cache_actions
        )

    def _setup_bidders(self):
        epo_n = 2  # for ensure positive output of initialization

        self.models = [None] * self.n_models
        for i in range(self.n_models):
            ensure_positive_output = torch.zeros(epo_n, self.input_length).uniform_(self.u_lo[i], self.u_hi[i]) \
                .sort(dim=1, descending=True)[0]
            self.models[i] = NeuralNetStrategy(
                self.input_length,
                hidden_nodes=self.learning_config.hidden_nodes,
                hidden_activations=self.learning_config.hidden_activations,
                ensure_positive_output=ensure_positive_output,
                output_length=self.n_units
            ).to(self.gpu_config.device)

        # Pretrain
        pretrain_points = round(100 ** (1 / self.input_length))
        # pretrain_valuations = multi_unit_valuations(
        #     device = device,
        #     bounds = [param_dict["u_lo"], param_dict["u_hi"][0]],
        #     dim = param_dict["n_units"],
        #     batch_size = pretrain_points,
        #     selection = 'random' if param_dict["exp_no"] != 6 else split_award_dict
        # )
        pretrain_valuations = self._strat_to_bidder(
            ClosureStrategy(lambda x: x), self.learning_config.batch_size, 0).draw_valuations_()[:pretrain_points, :]

        for model in self.models:
            model.pretrain(pretrain_valuations, self.learning_config.pretrain_iters,
                           self.pretrain_transform)
        self.bidders = [
            self._strat_to_bidder(self.models[0 if self.model_sharing else i], self.learning_config.batch_size, i)
            for i in range(self.n_players)
        ]

    def _setup_mechanism(self):
        self.mechanism = self.mechanism_type(cuda=self.gpu_config.cuda)

    def _setup_learning_environment(self):
        self.env = AuctionEnvironment(
            mechanism=self.mechanism,
            agents=self.bidders,
            n_players=self.n_players,
            batch_size=self.learning_config.batch_size,
            strategy_to_player_closure=self._strat_to_bidder
        )

    def _setup_learners(self):
        self.learners = [
            ESPGLearner(
                model=model,
                environment=self.env,
                hyperparams=self.learning_config.learner_hyperparams,
                optimizer_type=self.learning_config.optimizer,
                optimizer_hyperparams=self.learning_config.optimizer_hyperparams,
                strat_to_player_kwargs={"player_position": i}
            )
            for i, model in enumerate(self.models)
        ]

    def _setup_eval_environment(self):
        self.bne_strategies = [
            ClosureStrategy(self._optimal_bid) for i in range(self.n_players)
        ]

        self.bne_env = AuctionEnvironment(
            mechanism=self.mechanism,
            agents=[
                self._strat_to_bidder(bne_strategy, self.logging_config.eval_batch_size, i)
                for i, bne_strategy in enumerate(self.bne_strategies)
            ],
            n_players=self.n_players,
            batch_size=self.logging_config.eval_batch_size,
            strategy_to_player_closure=self._strat_to_bidder
        )

        self.bne_utilities = [self.bne_env.get_reward(agent, draw_valuations=True)
                              for agent in self.bne_env.agents]

    def _get_logdir(self):
        name = ['MultiUnit', self.payment_rule, str(self.n_players) + 'players_' + str(self.n_units) + 'units']
        return os.path.join(*name)

    def _plot(self, fig, plot_data, writer: SummaryWriter or None, epoch=None,
                xlim: list=None, ylim: list=None, labels: list=None,
                x_label="valuation", y_label="bid", fmts=['o'],
                figure_name: str='bid_function', plot_points=100):

        super()._plot(fig, plot_data, writer, epoch, xlim, ylim, labels,
                    x_label, y_label, fmts, figure_name, plot_points)

        super()._plot_3d(plot_data, writer, epoch, figure_name)

    @staticmethod
    def default_pretrain_transform(input_tensor):
        return torch.clone(input_tensor)

class SplitAwardExperiment(MultiUnitExperiment):
    """
    Experiment of the first-price sealed bid split-award auction.
    """
    def __init__(self, experiment_config: ExperimentConfiguration, learning_config: LearningConfiguration,
                 logging_config: LoggingConfiguration, gpu_config: GPUController):

        self.efficiency_parameter = experiment_config.efficiency_parameter
        self.input_length = experiment_config.input_length

        super().__init__(experiment_config, learning_config, logging_config, gpu_config)

        assert all(u_lo > 0 for u_lo in experiment_config.u_lo), \
            '100% Unit must be valued > 0'

        if self.payment_rule == 'first_price':
            self.mechanism_type = FPSBSplitAwardAuction
        else:
            raise NotImplementedError('for the split-award auction only the ' + \
                'first-price payment rule is supported')

        self.plot_xmin = [self.u_lo[0], self.u_hi[0]]
        self.plot_xmax = [self.experiment_config.efficiency_parameter * self.u_lo[0],
                          self.experiment_config.efficiency_parameter * self.u_hi[0]]
        self.plot_ymin = [0, 2 * self.u_hi[0]]
        self.plot_ymax = [0, 2 * self.u_hi[0]]

    def default_pretrain_transform(self, input_tensor):
        temp = input_tensor.clone().detach()
        if input_tensor.shape[1] == 1:
            output_tensor = torch.cat((
                temp,
                self.efficiency_parameter * temp
            ), 1)
        else:
            output_tensor = temp
        return output_tensor

    def _strat_to_bidder(self, strategy, batch_size, player_position=None, cache_actions=False):
        """Standard strat_to_bidder method, but with ReverseBidder"""
        return ReverseBidder.uniform(
            lower=self.u_lo[0], upper=self.u_hi[0],
            strategy=strategy,
            n_units=self.n_units,
            item_interest_limit=self.item_interest_limit,
            descending_valuations=True,
            constant_marginal_values=self.constant_marginal_values,
            player_position=player_position,
            efficiency_parameter=self.efficiency_parameter,
            batch_size=batch_size,
            cache_actions=cache_actions
        )

    def _get_logdir(self):
        name = ['SplitAward', self.payment_rule, str(self.n_players) + 'players_' +
                str(self.n_units) + 'units']
        return os.path.join(*name)

    def _plot(self, fig, plot_data, writer: SummaryWriter or None, epoch=None,
                xlim: list=None, ylim: list=None, labels: list=None,
                x_label="valuation", y_label="bid", fmts=['o'],
                figure_name: str='bid_function', plot_points=100):

        super()._plot(fig, plot_data, writer, epoch, xlim, ylim, labels,
                    x_label, y_label, fmts, figure_name, plot_points)