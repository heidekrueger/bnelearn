"""
In this file multi-unit experiments ´MultiUnitExperiment´ and their analytical
BNEs (if known) are defiened. Also, the ´SplitAwardExperiment´ is implemented as well,
as it shares most its properties.
"""

import os
import warnings
from abc import ABC
from typing import Callable

import torch


from scipy import integrate, interpolate
from torch.utils.tensorboard import SummaryWriter

from bnelearn.bidder import Bidder, ReverseBidder
from bnelearn.environment import AuctionEnvironment
from bnelearn.experiment.configurations import ExperimentConfig
from bnelearn.experiment.equilibria import (
    _multiunit_bne, _optimal_bid_multidiscriminatory2x2,
    _optimal_bid_multidiscriminatory2x2CMV, _optimal_bid_multiuniform2x2,
    _optimal_bid_multiuniform3x2limit2, _optimal_bid_splitaward2x2_1,
    _optimal_bid_splitaward2x2_2)
from bnelearn.mechanism import (FPSBSplitAwardAuction,
                                MultiUnitDiscriminatoryAuction,
                                MultiUnitUniformPriceAuction,
                                MultiUnitVickreyAuction)
from bnelearn.sampler import (MultiUnitValuationObservationSampler,
                              CompositeValuationObservationSampler)
from bnelearn.strategy import ClosureStrategy


from .experiment import Experiment


class MultiUnitExperiment(Experiment, ABC):
    """
    Experiment class for the standard multi-unit auctions.
    """

    def __init__(self, config: ExperimentConfig):
        self.config = config

        self.n_units = self.n_items = self.config.setting.n_units
        self.observation_size = self.valuation_size = self.action_size = self.n_units
        self.n_players = self.config.setting.n_players
        self.payment_rule = self.config.setting.payment_rule
        self.risk = float(self.config.setting.risk)

        # Transfrom bounds to list in case of symmetry
        if len(self.config.setting.u_lo) == 1:
            self.u_lo = self.config.setting.u_lo * self.n_players
        if len(self.config.setting.u_hi) == 1:
            self.u_hi = self.config.setting.u_hi * self.n_players

        # Handle correlation
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

        # Handle model sharing in case of symmetry
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

        # Set properties of prior
        self.constant_marginal_values = self.config.setting.constant_marginal_values
        self.item_interest_limit = self.config.setting.item_interest_limit

        if self.config.setting.pretrain_transform is not None:
            self.pretrain_transform = self.config.setting.pretrain_transform
        else:
            self.pretrain_transform = self.default_pretrain_transform

        self.plot_xmin = self.plot_ymin = min(self.u_lo)
        self.plot_xmax = self.plot_ymax = max(self.u_hi)

        super().__init__(config=config)

    def _strat_to_bidder(self, strategy, batch_size, player_position=0, enable_action_caching=False):
        """
        Standard strat_to_bidder method.
        """
        return Bidder(strategy, player_position, batch_size, bid_size=self.n_units,
                      enable_action_caching=enable_action_caching, risk=self.risk)

    def _setup_sampler(self):
        """
        `bidder_samplers` could be combined for symmetric priot bounds.
        """
        default_batch_size = self.learning.batch_size
        device = self.hardware.device
        
        # if TODO
        
        # setup individual samplers for each bidder
        bidder_samplers = [
            MultiUnitValuationObservationSampler(
                n_players=1, n_items=self.n_units,
                max_demand=self.item_interest_limit,
                u_lo=self.u_lo[i], u_hi=self.u_hi[i],
                default_batch_size=self.learning.batch_size,
                default_device=self.hardware.device
            )
            for i in range(self.n_players)
        ]

        self.sampler = CompositeValuationObservationSampler(
            self.n_players, self.valuation_size, self.observation_size,
            bidder_samplers, default_batch_size, device
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
                                          enable_action_caching=self.config.logging.cache_eval_actions)
                    for j, bne_strategy in enumerate(bne_strategies)
                ],
                valuation_observation_sampler=self.sampler,
                n_players=self.n_players,
                batch_size=self.logging.eval_batch_size,
                strategy_to_player_closure=self._strat_to_bidder
            )

            self.bne_utilities[i] = [self.bne_env[i].get_reward(agent, redraw_valuations=True)
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
              x_label="valuation", y_label="bid", colors=None, fmts=['o'],
              figure_name: str='bid_function', plot_points=100):

        super()._plot(plot_data=plot_data, writer=writer, epoch=epoch,
                      xlim=xlim, ylim=ylim, labels=labels, x_label=x_label,
                      y_label=y_label, colors=colors, fmts=fmts,
                      figure_name=figure_name, plot_points=plot_points)

        # 3D plot if available
        if self.n_units == 2 and not isinstance(self, SplitAwardExperiment):
            # Discard BNEs as they're making 3d plots more complicated
            if self.known_bne and plot_data[0].shape[1] > len(self.models):
                plot_data = [d[:, :len(self.models), :] for d in plot_data]
            super()._plot_3d(plot_data=plot_data, writer=writer, epoch=epoch,
                             figure_name=figure_name, labels=labels)

    @staticmethod
    def default_pretrain_transform(input_tensor):
        """Default pretrain transformation: truthful bidding"""
        return torch.clone(input_tensor)


class SplitAwardExperiment(MultiUnitExperiment):
    """
    Experiment class of the first-price sealed bid split-award auction.
    """

    def __init__(self, config: ExperimentConfig):
        raise NotImplementedError("Implementation does not yet work after #188.")
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

    def _setup_sampler(self):
        return NotImplementedError

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

    # def default_pretrain_transform(self, input_tensor):
    #     """Pretrain transformation for this setting"""
    #     temp = torch.clone(input_tensor)
    #     if input_tensor.shape[1] == 1:
    #         output_tensor = torch.cat((
    #             temp,
    #             self.efficiency_parameter * temp
    #         ), 1)
    #     else:
    #         output_tensor = temp
    #     return output_tensor

    def _strat_to_bidder(self, strategy, batch_size, player_position=None, enable_action_caching=False):
        """Standard strat_to_bidder method, but with ReverseBidder"""
        return ReverseBidder(
            strategy=strategy,
            n_items=self.n_units,
            item_interest_limit=self.item_interest_limit,
            descending_valuations=False,
            constant_marginal_values=self.constant_marginal_values,
            player_position=player_position,
            efficiency_parameter=self.efficiency_parameter,
            batch_size=batch_size,
            enable_action_caching=enable_action_caching
        )

    def _get_logdir_hierarchy(self):
        name = ['SplitAward', self.payment_rule, str(self.n_players) + 'players_' +
                str(self.n_units) + 'units']
        return os.path.join(*name)
