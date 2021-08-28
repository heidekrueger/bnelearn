"""In this file multi-unit experiments ´MultiUnitExperiment´ are defined and
their analytical BNEs (if known) are assigned. Also, the ´SplitAwardExperiment´
is implemented as well, as it shares most its properties.
"""

import os
import warnings
from abc import ABC

import torch
from torch.utils.tensorboard import SummaryWriter

from bnelearn.experiment.logger import Logger

from bnelearn.bidder import Bidder, ReverseBidder
from bnelearn.environment import AuctionEnvironment
from bnelearn.experiment import Experiment
from bnelearn.experiment.configurations import ExperimentConfig
from bnelearn.experiment.equilibria import (multiunit_bne_factory,
                                            bne_splitaward_2x2_1,
                                            bne_splitaward_2x2_2)
from bnelearn.mechanism import (FPSBSplitAwardAuction,
                                MultiUnitDiscriminatoryAuction,
                                MultiUnitUniformPriceAuction,
                                MultiUnitVickreyAuction)
from bnelearn.sampler import (MultiUnitValuationObservationSampler,
                              CompositeValuationObservationSampler,
                              SplitAwardtValuationObservationSampler)
from bnelearn.strategy import ClosureStrategy


class _MultiUnitSetupEvalMixin(ABC):
    r"""Mixinthat provides a common implementation of `_setup_eval_environment`
       for both `MultiUnitExperiment` and `SplitAwardExperiment`.
    """
    # Mixin class --> false positives in pylint.
    # pylint: disable=no-member,attribute-defined-outside-init,access-member-before-definition

    def _setup_eval_environment(self):
        """Setup the BNE envierment for later evaluation of the learned strategies."""

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


class MultiUnitExperiment(_MultiUnitSetupEvalMixin, Experiment):
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

        # TODO remove comment
        #else:
        #    self.pretrain_transform = self.default_pretrain_transform
        #self.input_length = self.config.setting.n_units
        
        plot_bounds = {}
        plot_bounds['plot_xmin'] =  min(self.u_lo)
        plot_bounds['plot_xmax'] =  max(self.u_hi)
        plot_bounds['plot_ymin'] =  min(self.u_lo)
        plot_bounds['plot_ymax'] =  max(self.u_hi)

        super().__init__(config=config)
                
        if self.logging.enable_logging:
            self.logger = Logger(config=self.config, known_bne=self.known_bne, plot_bounds=plot_bounds,  
                                evaluation_env=self.bne_env, _model2bidder=self._model2bidder, n_models=self.n_models, 
                                model_names=self._model_names, logdir_hierarchy=self._get_logdir_hierarchy(), 
                                sampler=self.sampler, plotter=self._plot, optimal_bid=self._optimal_bid)

    def _strat_to_bidder(self, strategy, batch_size, player_position=0, enable_action_caching=False):
        """Standard `strat_to_bidder` method."""
        return Bidder(strategy, player_position, batch_size, bid_size=self.n_units,
                      enable_action_caching=enable_action_caching, risk=self.risk)

    def _setup_sampler(self):
        default_batch_size = self.learning.batch_size
        device = self.hardware.device

        # Handle correlation
        if self.config.setting.correlation_types in ['independent', None]:
            self.gamma = self.correlation = 0.0
            if self.config.setting.gamma is not None \
                and float(self.config.setting.gamma) > 0:
                warnings.warn('No correlation selected.')
        else:
            raise NotImplementedError('Correlation not implemented for MultiUnit settings.')

        # Check for symmetric priors
        if len(set(self.u_lo)) == 1 and len(set(self.u_hi)) == 1:
            # Case: Symmetric Priors
            self.sampler = MultiUnitValuationObservationSampler(
                n_players=self.n_players, n_items=self.n_units,
                max_demand=self.item_interest_limit,
                u_lo=self.u_lo[0], u_hi=self.u_hi[0],
                default_batch_size=default_batch_size,
                default_device=device)
        else:
            # Case: asymmetric bidders with individual samplers
            bidder_samplers = [
                MultiUnitValuationObservationSampler(
                    n_players=1, n_items=self.n_units,
                    max_demand=self.item_interest_limit,
                    u_lo=self.u_lo[i], u_hi=self.u_hi[i],
                    default_batch_size=default_batch_size,
                    default_device=device)
                for i in range(self.n_players)
            ]
            self.sampler = CompositeValuationObservationSampler(
                self.n_players, self.valuation_size, self.observation_size,
                bidder_samplers, default_batch_size, device)

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
            self._optimal_bid = multiunit_bne_factory(self.config.setting, self.config.setting.payment_rule)
            return self._optimal_bid is not None
        return super()._check_and_set_known_bne()

    def _get_logdir_hierarchy(self):
        name = ['multi_unit', self.payment_rule, str(self.risk) + 'risk',
                str(self.n_players) + 'players_' + str(self.n_units) + 'units']
        # if self.gamma > 0:
        #     name += [self.config.setting.correlation_types, f"gamma_{self.gamma:.3}"]
        return os.path.join(*name)

    def _plot(self, plot_data, writer: SummaryWriter or None, epoch=None,
              xlim: list = None, ylim: list = None, labels: list = None,
              x_label="valuation", y_label="bid", fmts=['o'],
              colors: list = None, figure_name: str = 'bid_function',
              plot_points: int = 100):
        """Plotting of multi-unit experiment with possible 3D plot for two unit
        case.
        """
        super()._plot(plot_data=plot_data, writer=writer, epoch=epoch,
                      xlim=xlim, ylim=ylim, labels=labels, x_label=x_label,
                      y_label=y_label, colors=colors, fmts=fmts,
                      figure_name=figure_name, plot_points=plot_points)

        # 3D plot if available
        if self.n_units == 2:
            # Discard BNEs as they're making 3d plots more complicated
            if self.known_bne and plot_data[0].shape[1] > len(self.models):
                plot_data = [d[:, :len(self.models), :] for d in plot_data]
            super()._plot_3d(plot_data=plot_data, writer=writer, epoch=epoch,
                             figure_name=figure_name, labels=labels)


class SplitAwardExperiment(_MultiUnitSetupEvalMixin, Experiment):
    """Experiment class of the first-price sealed bid split-award auction."""

    def __init__(self, config: ExperimentConfig):
        self.config = config

        assert self.config.setting.u_lo is not None, "Prior boundaries not specified!"
        assert self.config.setting.u_hi is not None, "Prior boundaries not specified!"

        assert len(set(self.config.setting.u_lo)) == 1, "Only symmetric priors supported!"
        assert len(set(self.config.setting.u_hi)) == 1, "Only symmetric priors supported!"

        assert self.config.setting.n_units == 2, 'Only two units (lots) supported!'
        assert self.config.setting.n_players == 2, 'Only two players are supported!'
        assert all(u_lo > 0 for u_lo in self.config.setting.u_lo), \
            '100% Unit must be valued > 0'

        # Split-award specific parameters
        self.n_units = self.n_items = self.action_size = \
            self.observation_size = self.valuation_size = self.config.setting.n_units
        self.n_players = self.config.setting.n_players
        self.payment_rule = self.config.setting.payment_rule
        self.risk = float(self.config.setting.risk)
        self.efficiency_parameter = self.config.setting.efficiency_parameter

        # Handle model sharing in case of symmetry
        self.model_sharing = self.config.learning.model_sharing
        if self.model_sharing:
            self.n_models = 1
            self._bidder2model = [0] * self.n_players
        else:
            self.n_models = self.n_players
            self._bidder2model = list(range(self.n_players))

        # Setup valuation prior
        self.valuation_prior = 'uniform'
        self.u_lo = torch.tensor(self.config.setting.u_lo[0], dtype=torch.float32,
            device=self.config.hardware.device)
        self.u_hi = torch.tensor(self.config.setting.u_hi[0], dtype=torch.float32,
            device=self.config.hardware.device)
        self.config.setting.common_prior = \
            torch.distributions.uniform.Uniform(low=self.u_lo, high=self.u_hi)
        self.common_prior = self.config.setting.common_prior

        self.positive_output_point = torch.tensor(
            [1.2, self.efficiency_parameter * 1.2], dtype=torch.float)

        # Plotting bounds
        _lo = self.config.setting.u_lo[0]
        _hi = self.config.setting.u_hi[0]
        plot_bounds = {}
        plot_bounds['plot_xmin'] = [self.efficiency_parameter * _lo, _lo]
        plot_bounds['plot_xmax'] = [self.efficiency_parameter * _hi, _hi]
        plot_bounds['plot_ymin'] = [0, 0]
        plot_bounds['plot_ymax'] = [2 * _hi, 2 * _hi]

        super().__init__(config=config)
                
        if self.logging.enable_logging:
            self.logger = Logger(config=self.config, known_bne=self.known_bne, plot_bounds=plot_bounds, 
                                evaluation_env=self.bne_env, _model2bidder=self._model2bidder, n_models=self.n_models, 
                                model_names=self._model_names, logdir_hierarchy=self._get_logdir_hierarchy(), 
                                sampler=self.sampler, plotter=self._plot, optimal_bid=self._optimal_bid)

        


    def _setup_sampler(self):

        # Handle correlation
        if self.config.setting.correlation_types in ['independent', None]:
            self.gamma = self.correlation = 0.0
            if self.config.setting.gamma is not None \
                and float(self.config.setting.gamma) > 0:
                warnings.warn('No correlation selected.')
        else:
            raise NotImplementedError('Correlation not implemented.')

        # Setup sampler
        self.sampler = SplitAwardtValuationObservationSampler(
            lo=self.u_lo, hi=self.u_hi, efficiency_parameter=self.efficiency_parameter,
            valuation_size=self.valuation_size,
            default_batch_size=self.config.learning.batch_size,
            default_device=self.config.hardware.device
        )

    def _setup_mechanism(self):
        if self.payment_rule == 'first_price':
            self.mechanism = FPSBSplitAwardAuction(cuda=self.hardware.cuda)
        else:
            raise NotImplementedError('for the split-award auction only the ' \
                + 'first-price payment rule is supported')

    def _check_and_set_known_bne(self):
        """check for available BNE strategy"""
        if self.config.setting.n_units == 2 and self.config.setting.n_players == 2 \
            and self.risk == 1 and self.correlation == 0:
            self._optimal_bid = [
                bne_splitaward_2x2_1(self.config.setting, True),
                bne_splitaward_2x2_1(self.config.setting, False),
                bne_splitaward_2x2_2(self.config.setting)
            ]
            return True
        return super()._check_and_set_known_bne()

    def pretrain_transform(self, input_tensor):
        """Pretrain transformation for this setting"""
        temp = torch.clone(input_tensor)
        if input_tensor.shape[1] == 1:
            output_tensor = torch.cat((
                temp,
                self.efficiency_parameter * temp
            ), 1)
        else:
            output_tensor = temp
        return output_tensor

    def _strat_to_bidder(self, strategy, batch_size, player_position=None, enable_action_caching=False):
        """Standard strat_to_bidder method, but with ReverseBidder"""
        return ReverseBidder(strategy=strategy, player_position=player_position,
                             batch_size=batch_size, bid_size=self.n_units,
                             enable_action_caching=enable_action_caching,
                             efficiency_parameter=self.efficiency_parameter,
                             risk=self.risk)

    def _get_logdir_hierarchy(self):
        name = ['SplitAward', self.payment_rule, str(self.n_players) + 'players_' +
                str(self.n_units) + 'units']
        return os.path.join(*name)
