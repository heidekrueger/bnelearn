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

    def _plot(self, plot_data, writer: SummaryWriter or None, epoch=None,
              xlim: list = None, ylim: list = None, labels: list = None,
              x_label="valuation", y_label="bid", fmts=['o'],
              figure_name: str = 'bid_function', plot_points=100):

        super()._plot(plot_data=plot_data, writer=writer, epoch=epoch,
                      xlim=xlim, ylim=ylim, labels=labels, x_label=x_label,
                      y_label=y_label, fmts=fmts, figure_name=figure_name,
                      plot_points=plot_points)

        if self.n_units == 2:
            super()._plot_3d(plot_data, writer, epoch, figure_name)

    @staticmethod
    def default_pretrain_transform(input_tensor):
        """Default pretrain transformation: truthful bidding"""
        return torch.clone(input_tensor)
