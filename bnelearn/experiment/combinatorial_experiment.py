"""
This module implements combinatorial experiments. Currently, this is only Local Global experiments as
considered by Bosshard et al. (2018).

Limitations and comments:
    - Currently implemented for only uniform valuations
    - Strictly speaking Split Award might belong here (however, implmentation closer to multi-unit)

TODO:
    - Check if truthful bidding is BNE in LLLLGG with VCG
"""
import os
from abc import ABC
from functools import partial
from typing import Iterable, List
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import torch

from bnelearn.mechanism.auctions_combinatorial import LLGAuction, LLLLGGAuction
from bnelearn.bidder import Bidder
from bnelearn.environment import AuctionEnvironment
from bnelearn.experiment.configurations import ExperimentConfig
from bnelearn.experiment import Experiment
from bnelearn.strategy import ClosureStrategy
from bnelearn.correlation_device import (
    IndependentValuationDevice,
    BernoulliWeightsCorrelationDevice,
    ConstantWeightsCorrelationDevice
)


class LocalGlobalExperiment(Experiment, ABC):
    """
    This class represents Local Global experiments in general as considered by Bosshard et al. (2018).
    It serves only to provide common logic and parameters for LLG and LLLLGG.
    """

    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.n_players = self.config.setting.n_players
        self.n_local = self.config.setting.n_local
        self.n_items = self.config.setting.n_items

        assert self.config.setting.u_lo is not None, """Missing prior information!"""
        assert self.config.setting.u_hi is not None, """Missing prior information!"""
        u_lo = self.config.setting.u_lo
        # Frontend could either provide single number u_lo that is shared or a list for each player.
        if isinstance(u_lo, Iterable):
            assert len(u_lo) == self.n_players
            u_lo = [float(l) for l in u_lo]
        else:
            u_lo = [float(u_lo)] * self.n_players
        self.u_lo = u_lo

        u_hi = self.config.setting.u_hi
        assert isinstance(u_hi, Iterable)
        assert len(u_hi) == self.n_players
        assert u_hi[1:self.config.setting.n_local] == \
               u_hi[:self.config.setting.n_local - 1], "local bidders should be identical"
        assert u_hi[0] < \
               u_hi[self.config.setting.n_local], "local bidders must be weaker than global bidder"
        self.u_hi = [float(h) for h in u_hi]

        self.positive_output_point = torch.tensor([min(self.u_hi)] * self.n_items)

        self.model_sharing = self.config.learning.model_sharing
        if self.model_sharing:
            self.n_models = 2
            self._bidder2model: List[int] = [0] * self.config.setting.n_local \
                                            + [1] * (self.n_players - self.config.setting.n_local)
        else:
            self.n_models = self.n_players
            self._bidder2model: List[int] = list(range(self.n_players))

        super().__init__(config=config)

        self.plot_xmin = min(u_lo)
        self.plot_xmax = max(u_hi)
        self.plot_ymin = self.plot_xmin
        self.plot_ymax = self.plot_xmax * 1.05

    def _get_model_names(self):
        if self.model_sharing:
            global_name = 'global' if self.n_players - self.n_local == 1 else 'globals'
            return ['locals', global_name]
        else:
            return super()._get_model_names()


    def _strat_to_bidder(self, strategy, batch_size, player_position=0, cache_actions=False):
        correlation_type = 'additive' if hasattr(self, 'correlation_groups') else None
        return Bidder.uniform(self.u_lo[player_position], self.u_hi[player_position], strategy, player_position=player_position,
                              batch_size=batch_size, n_items = self.n_items, correlation_type=correlation_type)


class LLGExperiment(LocalGlobalExperiment):
    """
    A combinatorial experiment with 2 local and 1 global bidder and 2 items; but each bidders bids on 1 bundle only.
    Local bidder 1 bids only on the first item, the second only on the second and global only on both.
    Ausubel and Baranov (2018) provide closed form solutions for the 3 core selecting rules.
    """

    def __init__(self, config: ExperimentConfig):
        self.config = config
        assert self.config.setting.n_players == 3, "Incorrect number of players specified."

        self.gamma = config.setting.gamma

        if self.gamma > 0.0:
            if config.setting.correlation_types == 'Bernoulli_weights':
                self.correlation_groups = [[0, 1], [2]]
                self.correlation_coefficients = [self.gamma, 0.0]
                self.correlation_devices = [
                    BernoulliWeightsCorrelationDevice(
                        common_component_dist = torch.distributions.Uniform(config.setting.u_lo[0],
                                                                            config.setting.u_hi[0]),
                        batch_size=config.learning.batch_size,
                        n_items=1,
                        correlation = self.gamma),
                    IndependentValuationDevice()]
            elif config.setting.correlation_types == 'constant_weights':
                self.correlation_groups = [[0, 1], [2]]
                self.correlation_coefficients = [self.gamma, 0.0]
                self.correlation_devices = [
                    ConstantWeightsCorrelationDevice(
                        common_component_dist = torch.distributions.Uniform(config.setting.u_lo[0],
                                                                            config.setting.u_hi[0]),
                        batch_size=config.learning.batch_size,
                        n_items=1,
                        correlation = self.gamma),
                    IndependentValuationDevice()]
            else:
                raise NotImplementedError('other correlation not implemented.')

        self.input_length = 1
        #self.config.setting.n_players = 3
        self.config.setting.n_local = 2
        self.config.setting.n_items = 1
        super().__init__(config=config)

    def _setup_mechanism(self):
        self.mechanism = LLGAuction(rule=self.payment_rule)

    def _optimal_bid(self, valuation, player_position):
        """Core selecting and vcg equilibria for the Bernoulli weigths model in Ausubel & Baranov (2019)

           Note: for gamma=0 or gamma=1, these are identical to the constant weights model.
        """
        if not isinstance(valuation, torch.Tensor):
            valuation = torch.tensor(valuation)

        # all core-selecting rules are strategy proof for global player:
        if self.payment_rule in ['vcg', 'proxy', 'nearest_zero', 'nearest_bid',
                                 'nearest_vcg'] and player_position == 2:
            return valuation
        ##### Local bidders:

        ## perfect correlation
        if not self.config.setting.correlation_types == 'constant_weights':
            if self.gamma == 1.0: #limit case, others not well defined
                sigma = 1.0 # TODO: implement for other valuation profiles!
                bid = valuation
                if self.payment_rule == 'nearest_vcg':
                    bid.mul_(sigma / (1 + sigma - 2**(-sigma)))
                elif self.payment_rule == 'nearest_bid':
                    bid.mul_(sigma / (1 + sigma))
                # truthful for vcg and proxy/nearest-zero
                return bid
            ## no or imperfect correlation
            if self.payment_rule == 'vcg':
                return valuation
            if self.payment_rule in ['proxy', 'nearest_zero']:
                bid_if_positive = 1 + torch.log(valuation * (1.0 - self.gamma) + self.gamma) / (1.0 - self.gamma)
                return torch.max(torch.zeros_like(valuation), bid_if_positive)
            if self.payment_rule == 'nearest_bid':
                return (np.log(2) - torch.log(2.0 - (1. - self.gamma) * valuation)) / (1. - self.gamma)
            if self.payment_rule == 'nearest_vcg':
                bid_if_positive = 2. / (2. + self.gamma) * (
                    valuation - (3. - np.sqrt(9 - (1. - self.gamma) ** 2)) / (1. - self.gamma))
                return torch.max(torch.zeros_like(valuation), bid_if_positive)
            raise ValueError('optimal bid not implemented for this payment rule')
        else:
            raise ValueError('optimal bid not implemented for this correlation type')

    def _check_and_set_known_bne(self):
        # TODO: This is not exhaustive, other criteria must be fulfilled for the bne to be known!
        #  (i.e. uniformity, bounds, etc)
        if self.config.setting.payment_rule in \
            ['vcg', 'nearest_bid', 'nearest_zero', 'proxy', 'nearest_vcg']:
            return True
        return super()._check_and_set_known_bne()

    def _setup_eval_environment(self):

        assert self.known_bne
        assert hasattr(self, '_optimal_bid')

        bne_strategies = [
            ClosureStrategy(partial(self._optimal_bid, player_position=i))  # pylint: disable=no-member
            for i in range(self.n_players)]

        if not self.config.setting.correlation_types == 'constant_weights':
            bne_env_corr_devices = None
            if self.correlation_groups:
                bne_env_corr_devices = [
                    BernoulliWeightsCorrelationDevice(
                        common_component_dist=torch.distributions.Uniform(self.config.setting.u_lo[0],
                                                                          self.config.setting.u_hi[0]),
                        batch_size=self.config.logging.eval_batch_size,
                        n_items=1,
                        correlation=self.gamma),
                    IndependentValuationDevice()]

        else:
            self.known_bne = False
            return

        bne_env = AuctionEnvironment(
            mechanism=self.mechanism,
            agents=[self._strat_to_bidder(bne_strategies[i], player_position=i,
                                          batch_size=self.config.logging.eval_batch_size)
                    for i in range(self.n_players)],
            n_players=self.n_players,
            batch_size=self.config.logging.eval_batch_size,
            strategy_to_player_closure=self._strat_to_bidder,
            correlation_groups=self.correlation_groups,
            correlation_devices=bne_env_corr_devices
            )

        self.bne_env = bne_env

        bne_utilities_sampled = torch.tensor(
            [bne_env.get_reward(a, draw_valuations=True) for a in bne_env.agents])
        print(f'Setting up BNE env with batch size 2**{np.log2(self.config.logging.eval_batch_size)}.')
        print(('Utilities in BNE (sampled):' + '\t{:.5f}' * self.n_players + '.').format(*bne_utilities_sampled))
        print(
            "No closed form solution for BNE utilities available in this setting. Using sampled value as baseline.")
        self.bne_utilities = bne_utilities_sampled

    def _get_logdir_hierarchy(self):
        name = ['LLG', self.payment_rule, self.config.setting.correlation_types, f"gamma_{self.gamma:.3}"]
        return os.path.join(*name)


class LLLLGGExperiment(LocalGlobalExperiment):
    """
    A combinatorial experiment with 4 local and 2 global bidder and 6 items; but each bidders bids on 2 bundles only.
        Local bidder 1 bids on the bundles {(item_1,item_2),(item_2,item_3)}
        Local bidder 2 bids on the bundles {(item_3,item_4),(item_4,item_5)}
        ...
        Gloabl bidder 1 bids on the bundles {(item_1,item_2,item_3,item_4), (item_5,item_6,item_7,item_8)}
        Gloabl bidder 1 bids on the bundles {(item_3,item_4,item_5,item_6), (item_1,item_2,item_7,item_8)}
    No BNE are known (but VCG).
    Bosshard et al. (2018) consider this setting with nearest-vcg and first-price payments.

    TODO:
        - Implement eval_env for VCG
    """

    def __init__(self, config: ExperimentConfig):
        self.config = config
        assert self.config.setting.n_players == 6, "not right number of players for setting"
        self.input_length = 2

        self.config.running.n_players = 6
        self.config.setting.n_local = 4
        self.config.setting.n_items = 2
        super().__init__(config=config)

    def _setup_mechanism(self):
        self.mechanism = LLLLGGAuction(rule=self.payment_rule, core_solver=self.setting.core_solver,
                                       parallel=self.hardware.max_cpu_threads, cuda=self.hardware.cuda)

    def _get_logdir_hierarchy(self):
        name = ['LLLLGG', self.payment_rule, str(self.n_players) + 'p']
        return os.path.join(*name)

    def _plot(self, plot_data, writer: SummaryWriter or None, epoch=None,
              xlim: list = None, ylim: list = None, labels: list = None,
              x_label="valuation", y_label="bid", fmts=['o'],
              figure_name: str = 'bid_function', plot_points=100):
        super()._plot(plot_data, writer, epoch, xlim, ylim, labels,
                      x_label, y_label, fmts, figure_name, plot_points)
        super()._plot_3d(plot_data, writer, epoch, figure_name)
