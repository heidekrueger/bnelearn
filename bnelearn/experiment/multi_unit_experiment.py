"""
In this file multi-unit experiments ´MultiUnitExperiment´ and their analytical
BNEs (if known) are defiened. Also, the ´SplitAwardExperiment´ is implemented as well,
as it shares most its properties.
"""

import os
from abc import ABC
import warnings

import torch
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from scipy import integrate, interpolate

from bnelearn.bidder import Bidder, ReverseBidder
from bnelearn.environment import AuctionEnvironment
from .experiment import  Experiment
from bnelearn.experiment.configurations import ExperimentConfig
from bnelearn.mechanism import (
    MultiUnitVickreyAuction, MultiUnitUniformPriceAuction,
    MultiUnitDiscriminatoryAuction, FPSBSplitAwardAuction
)
from bnelearn.strategy import ClosureStrategy
from bnelearn.correlation_device import (
    IndependentValuationDevice, MultiUnitDevice
)


###############################################################################
###                             BNE STRATEGIES                              ###
###############################################################################

def _multiunit_bne(setting, payment_rule):
    """
    Method that returns the known BNE strategy for the standard multi-unit auctions
    (split-award is NOT one of the) as callable if available and None otherwise.
    """

    if  float(setting.risk) != 1:
        return None  # Only know BNE for risk neutral bidders

    if payment_rule in ('vcg', 'vickrey'):
        def truthful(valuation, player_position=None):  # pylint: disable=unused-argument
            return valuation
        return truthful

    if (setting.correlation_types not in [None, 'independent'] or
            setting.risk != 1):
        return None

    if payment_rule in ('first_price', 'discriminatory'):
        if setting.n_units == 2 and setting.n_players == 2:
            if not setting.constant_marginal_values:
                print('BNE is only approximated roughly!')
                return _optimal_bid_multidiscriminatory2x2
            else:
                # TODO get valuation_cdf from experiment_config
                # return _optimal_bid_multidiscriminatory2x2CMV(valuation_cdf)
                return None

    if payment_rule == 'uniform':
        if setting.n_units == 2 and setting.n_players == 2:
            return _optimal_bid_multiuniform2x2()
        if (setting.n_units == 3 and setting.n_players == 2
                and setting.item_interest_limit == 2):
            return _optimal_bid_multiuniform3x2limit2

    return None

def _optimal_bid_multidiscriminatory2x2(valuation, player_position=None):
    """BNE strategy in the multi-unit discriminatory price auction 2 players and 2 units"""

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

    opt_bid = torch.clone(valuation)
    opt_bid[:, 0] = b1(opt_bid[:, 0])
    opt_bid[:, 1] = b2(opt_bid[:, 1])
    opt_bid = opt_bid.sort(dim=1, descending=True)[0]
    return opt_bid

def _optimal_bid_multidiscriminatory2x2CMV(valuation_cdf):
    """ BNE strategy in the multi-unit discriminatory price auction 2 players and 2 units
        with constant marginal valuations
    """
    n_players = 2

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

        bidding = muda_tb_cmv_bne(lambda x: torch.exp(valuation_cdf.log_prob(x)).cpu().numpy(),
                                  lambda x: valuation_cdf.cdf(x).cpu().numpy())

        def _optimal_bid(valuation, player_position=None):
            opt_bid = np.zeros_like(valuation.cpu().numpy())
            for agent in range(n_players):
                opt_bid[agent] = bidding(valuation[agent, :])
            return torch.tensor(opt_bid)

    return _optimal_bid

def _optimal_bid_multiuniform2x2():
    """ Returns two BNE strategies List[callable] in the multi-unit uniform price auction
        with 2 players and 2 units.
    """

    def opt_bid_1(valuation, player_position=None):
        opt_bid = torch.clone(valuation)
        opt_bid[:,1] = 0
        return opt_bid

    def opt_bid_2(valuation, player_position=None):
        opt_bid = torch.ones_like(valuation)
        opt_bid[:,1] = 0
        return opt_bid

    return [opt_bid_1, opt_bid_2]

def _optimal_bid_multiuniform3x2limit2(valuation, player_position=None):
    """ BNE strategy in the multi-unit uniform price auction with 3 units and
        2 palyers that are both only interested in 2 units
    """
    opt_bid = torch.clone(valuation)
    opt_bid[:, 1] = opt_bid[:, 1] ** 2
    opt_bid[:, 2] = 0
    return opt_bid

def _optimal_bid_splitaward2x2_1(experiment_config):
    """ BNE pooling equilibrium in the split-award auction with 2 players and
        2 lots (as in Anton and Yao, 1992)

        Returns callable.
    """

    efficiency_parameter = experiment_config.efficiency_parameter
    u_lo = experiment_config.u_lo
    u_hi = experiment_config.u_hi

    # cut off bids at top
    cut_off = 4 * u_hi[0]

    value_cdf = lambda theta: (theta - u_lo[0]) / (u_hi[0] - u_lo[0])

    def _optimal_bid(valuation, player_position=None, return_payoff_dominant=True):
        sigma_bounds = torch.ones_like(valuation, device=valuation.device)
        sigma_bounds[:, 0] = efficiency_parameter * u_hi[0]
        sigma_bounds[:, 1] = (1 - efficiency_parameter) * u_lo[0]
        # [:,0]: lower bound and [:,1] upper

        _p_sigma = (1 - efficiency_parameter) * u_lo[0]  # highest possible p_sigma

        def G(theta):
            return _p_sigma + (_p_sigma - u_hi[0] * efficiency_parameter * value_cdf(theta)) \
                   / (1 - value_cdf(theta))

        wta_bounds = 2 * sigma_bounds
        wta_bounds[:, 1] = G(valuation[:, 0])

        payoff_dominant_bid = torch.cat(
            (wta_bounds[:, 1].unsqueeze(0).t_(), sigma_bounds[:, 1].unsqueeze(0).t_()),
            axis=1
        )
        payoff_dominant_bid[payoff_dominant_bid > cut_off] = cut_off

        if return_payoff_dominant:
            return payoff_dominant_bid
        return {'sigma_bounds': sigma_bounds, 'wta_bounds': wta_bounds}

    return _optimal_bid

def _optimal_bid_splitaward2x2_2(experiment_config):
    """ BNE WTA equilibrium in the split-award auction with 2 players and
        2 lots (as in Anton and Yao Proposition 4, 1992).

        Returns callable.
    """
    print('Warning: BNE is approximated on CPU.')

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
    val_lin = np.linspace(u_lo[0], u_hi[0] - eps, opt_bid_batch_size)

    def integral(theta):
        return np.array(
            [integrate.quad(
                lambda x: (1 - value_cdf(x)) ** (n_players - 1), v, u_hi[0],
                epsabs=eps
            )[0] for v in theta]
        )

    def opt_bid_100(theta):
        return theta + (integral(theta) / (
            (1 - value_cdf(theta)) ** (n_players - 1)))

    opt_bid[:, 0] = opt_bid_100(val_lin)
    opt_bid[:, 1] = opt_bid_100(val_lin) - efficiency_parameter * u_lo[0]  # or more

    opt_bid_function = [
        interpolate.interp1d(val_lin, opt_bid[:, 0], fill_value='extrapolate'),
        interpolate.interp1d(val_lin, opt_bid[:, 1], fill_value='extrapolate')
    ]

    # use interpolation of opt_bid done on first batch
    def _optimal_bid(valuation, player_position=None):
        bid = torch.tensor(
            [opt_bid_function[0](valuation[:,0].cpu().numpy()),
             opt_bid_function[1](valuation[:,0].cpu().numpy())
            ],
            device = valuation.device,
            dtype = valuation.dtype
        ).t_()
        bid[bid < 0] = 0
        return bid

    return _optimal_bid

###############################################################################


class MultiUnitExperiment(Experiment, ABC):
    """
    Experiment class for the standard multi-unit auctions.
    """

    def __init__(self, config: ExperimentConfig):
        self.config = config

        self.n_units = self.n_items = self.config.setting.n_units
        self.n_players = self.config.setting.n_players
        self.payment_rule = self.config.setting.payment_rule
        self.risk = float(self.config.setting.risk)

        if len(self.config.setting.u_lo) == 1:
            self.u_lo = self.config.setting.u_lo * self.n_players

        if len(self.config.setting.u_hi) == 1:
            self.u_hi = self.config.setting.u_hi * self.n_players

        # Correlated setting?
        if config.setting.correlation_types == 'additive':
            self.CorrelationDevice = MultiUnitDevice
            self.gamma = self.correlation = float(config.setting.gamma)
        elif config.setting.correlation_types in ['independent', None]:
            self.gamma = self.correlation = 0.
            if config.setting.gamma is not None and float(config.setting.gamma) > 0:
                warnings.warn('No correlation selected.')
        else:
            raise NotImplementedError('Correlation not implemented.')

        if self.gamma > 0.0:
            self.correlation_groups = [list(range(self.n_players))]
            self.correlation_coefficients = [self.gamma]
            self.correlation_devices = [
                self.CorrelationDevice(
                    common_component_dist=torch.distributions.Uniform(
                        config.setting.u_lo[0], config.setting.u_hi[0]),
                    batch_size=config.learning.batch_size,
                    n_common_components=self.n_items,
                    correlation=self.gamma),
                ]
            # Can't sample cond values here
            self.config.logging.log_metrics['util_loss'] = False

        self.model_sharing = self.config.learning.model_sharing
        if self.model_sharing:
            self.n_models = 1
            self._bidder2model = [0] * self.n_players
        else:
            self.n_models = self.n_players
            self._bidder2model = list(range(self.n_players))

        if not hasattr(self, 'positive_output_point'):
            self.positive_output_point = torch.tensor(
                [self.u_hi[0]] * self.n_items, dtype=torch.float)

        self.constant_marginal_values = self.config.setting.constant_marginal_values
        self.item_interest_limit = self.config.setting.item_interest_limit

        if self.config.setting.pretrain_transform is not None:
            self.pretrain_transform = self.config.setting.pretrain_transform
        else:
            self.pretrain_transform = self.default_pretrain_transform

        self.input_length = self.config.setting.n_units

        self.plot_xmin = self.plot_ymin = min(self.u_lo)
        self.plot_xmax = self.plot_ymax = max(self.u_hi)

        super().__init__(config=config)

    def _strat_to_bidder(self, strategy, batch_size, player_position=0, cache_actions=False):
        """
        Standard strat_to_bidder method.
        """
        correlation_type = 'additive' if hasattr(self, 'correlation_groups') else None
        return Bidder.uniform(
            lower=self.u_lo[player_position], upper=self.u_hi[player_position],
            strategy=strategy,
            n_items=self.n_units,
            risk=self.risk,
            item_interest_limit=self.item_interest_limit,
            descending_valuations=True,
            constant_marginal_values=self.constant_marginal_values,
            player_position=player_position,
            batch_size=batch_size,
            cache_actions=cache_actions,
            correlation_type=correlation_type
        )

    def _setup_mechanism(self):
        """Setup the mechanism"""
        if self.payment_rule in ('discriminatory', 'first_price'):
            self.mechanism_type = MultiUnitDiscriminatoryAuction
        elif self.payment_rule in ('vcg', 'second_price'):
            self.mechanism_type = MultiUnitVickreyAuction
        elif self.payment_rule == 'uniform':
            self.mechanism_type = MultiUnitUniformPriceAuction
        else:
            raise ValueError('payment rule unknown')

        self.mechanism = self.mechanism_type(cuda=self.hardware.cuda)

    def _check_and_set_known_bne(self):
        """check for available BNE strategy"""
        if self.correlation in [0.0, None] and self.config.setting.correlation_types in ['independent', None]:
            self._optimal_bid = _multiunit_bne(self.config.setting, self.config.setting.payment_rule)
            return self._optimal_bid is not None
        return False

    def _setup_eval_environment(self):
        """Setup the BNE envierment for later evaluation of the learned strategies"""

        assert self.known_bne
        assert hasattr(self, '_optimal_bid')

        if not isinstance(self._optimal_bid, list):
            self._optimal_bid = [self._optimal_bid]

        # set up list for multiple bne
        self.bne_env = [None] * len(self._optimal_bid)
        self.bne_utilities = [None] * len(self._optimal_bid)

        for i, strat in enumerate(self._optimal_bid):
            bne_strategies = [ClosureStrategy(strat) for _ in range(self.n_players)]

            self.bne_env[i] = AuctionEnvironment(
                mechanism=self.mechanism,
                agents=[
                    self._strat_to_bidder(bne_strategy, self.logging.eval_batch_size, j,
                                          cache_actions=self.config.logging.cache_eval_actions)
                    for j, bne_strategy in enumerate(bne_strategies)
                ],
                n_players=self.n_players,
                batch_size=self.logging.eval_batch_size,
                strategy_to_player_closure=self._strat_to_bidder
            )

            self.bne_utilities[i] = [self.bne_env[i].get_reward(agent, draw_valuations=True)
                                     for agent in self.bne_env[i].agents]

        print('BNE envs have been set up.')

    def _get_logdir_hierarchy(self):
        name = ['multi_unit', self.payment_rule, str(self.risk) + 'risk',
                str(self.n_players) + 'players_' + str(self.n_units) + 'units']
        if self.gamma > 0:
            name += [self.config.setting.correlation_types, f"gamma_{self.gamma:.3}"]
        return os.path.join(*name)

    def _plot(self, **kwargs):
        super()._plot(**kwargs)

        if self.n_units == 2 and not isinstance(self, SplitAwardExperiment):
            super()._plot_3d(**kwargs)

    @staticmethod
    def default_pretrain_transform(input_tensor):
        """Default pretrain transformation: truthful bidding"""
        return torch.clone(input_tensor)


class SplitAwardExperiment(MultiUnitExperiment):
    """
    Experiment class of the first-price sealed bid split-award auction.
    """

    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.efficiency_parameter = self.config.setting.efficiency_parameter

        super().__init__(config=config)

        assert all(u_lo > 0 for u_lo in self.config.setting.u_lo), \
            '100% Unit must be valued > 0'

        self.positive_output_point = torch.tensor(
            [1.2, self.efficiency_parameter * 1.2], dtype=torch.float)

        # ToDO Implicit type conversion, OK?
        self.plot_xmin = [self.u_lo[0], self.u_hi[0]]
        self.plot_xmax = [self.setting.efficiency_parameter * self.u_lo[0],
                          self.setting.efficiency_parameter * self.u_hi[0]]
        self.plot_ymin = [0, 2 * self.u_hi[0]]
        self.plot_ymax = [0, 2 * self.u_hi[0]]

    def _setup_mechanism(self):
        if self.payment_rule == 'first_price':
            self.mechanism = FPSBSplitAwardAuction(cuda=self.hardware.cuda)
        else:
            raise NotImplementedError('for the split-award auction only the ' \
                + 'first-price payment rule is supported')

    def _check_and_set_known_bne(self):
        """check for available BNE strategy"""
        if self.config.setting.n_units == 2 and self.config.setting.n_players == 2:
            self._optimal_bid = [
                _optimal_bid_splitaward2x2_1(self.config.setting),
                _optimal_bid_splitaward2x2_2(self.config.setting)
            ]
        return True

    def _strat_to_bidder(self, strategy, batch_size, player_position=None, cache_actions=False):
        """Standard strat_to_bidder method, but with ReverseBidder"""
        return ReverseBidder.uniform(
            lower=self.u_lo[0], upper=self.u_hi[0],
            strategy=strategy,
            n_items=self.n_units,
            item_interest_limit=self.item_interest_limit,
            descending_valuations=False,
            constant_marginal_values=self.constant_marginal_values,
            player_position=player_position,
            efficiency_parameter=self.efficiency_parameter,
            batch_size=batch_size,
            cache_actions=cache_actions
        )

    def _get_logdir_hierarchy(self):
        name = ['SplitAward', self.payment_rule, str(self.n_players) + 'players_' +
                str(self.n_units) + 'units']
        return os.path.join(*name)
