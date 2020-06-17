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

from bnelearn.mechanism import (LLGAuction, LLLLGGAuction, FirstPriceSealedBidAuction, VickreyAuction)
from bnelearn.bidder import Bidder, CombinatorialItemBidder
from bnelearn.environment import AuctionEnvironment
from bnelearn.experiment import Experiment, GPUController
from bnelearn.experiment.configurations import ExperimentConfiguration, LearningConfiguration, LoggingConfiguration
from bnelearn.strategy import ClosureStrategy

class LocalGlobalExperiment(Experiment, ABC):
    """
    This class represents Local Global experiments in general as considered by Bosshard et al. (2018).
    It serves only to provide common logic and parameters for LLG and LLLLGG.
    """
    def __init__(self, n_players, n_local, n_items, experiment_config, learning_config,
                 logging_config, gpu_config, known_bne):
        self.n_players = n_players
        self.n_local = n_local
        self.n_items = n_items

        assert experiment_config.u_lo is not None, """Missing prior information!"""
        assert experiment_config.u_hi is not None, """Missing prior information!"""
        u_lo = experiment_config.u_lo
        # Frontend could either provide single number u_lo that is shared or a list for each player.
        if isinstance(u_lo, Iterable):
            assert len(u_lo) == self.n_players
            u_lo = [float(l) for l in u_lo]
        else:
            u_lo = [float(u_lo)] * self.n_players
        self.u_lo = u_lo

        u_hi = experiment_config.u_hi
        assert isinstance(u_hi, Iterable)
        assert len(u_hi) == self.n_players
        assert u_hi[1:n_local] == u_hi[:n_local-1], "local bidders should be identical"
        assert u_hi[0] < u_hi[n_local], "local bidders must be weaker than global bidder"
        self.u_hi = [float(h) for h in u_hi]

        self.positive_output_point = torch.tensor([min(self.u_hi)]*self.n_items)

        self.model_sharing = experiment_config.model_sharing
        if self.model_sharing:
            self.n_models = 2
            self._bidder2model: List[int] = [0] * n_local + [1] * (self.n_players - n_local)
        else:
            self.n_models = self.n_players
            self._bidder2model: List[int] = list(range(self.n_players))

        super().__init__(experiment_config,  learning_config, logging_config, gpu_config, known_bne)

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

    def _strat_to_bidder(self, strategy, batch_size, player_position=0):
        return Bidder.uniform(self.u_lo[player_position], self.u_hi[player_position], strategy, player_position=player_position,
                              batch_size=batch_size, n_items = self.n_items)

class LLGExperiment(LocalGlobalExperiment):
    """
    A combinatorial experiment with 2 local and 1 global bidder and 2 items; but each bidders bids on 1 bundle only.
    Local bidder 1 bids only on the first item, the second only on the second and global only on both.
    Ausubel and Baranov (2018) provide closed form solutions for the 3 core selecting rules.
    """
    def __init__(self, experiment_config: ExperimentConfiguration, learning_config: LearningConfiguration,
                 logging_config: LoggingConfiguration, gpu_config: GPUController):

        assert experiment_config.n_players == 3, "Incorrect number of players specified."

        self.gamma = experiment_config.gamma
        assert self.gamma == 0, "Gamma > 0 implemented yet."

        # TODO: This is not exhaustive, other criteria must be fulfilled for the bne to be known! (i.e. uniformity, bounds, etc)
        known_bne = experiment_config.payment_rule in \
            ['vcg', 'nearest_bid','nearest_zero', 'proxy', 'nearest_vcg']
        self.input_length = 1
        super().__init__(3,2, 1, experiment_config, learning_config, logging_config, gpu_config, known_bne)

    def _setup_mechanism(self):
        self.mechanism = LLGAuction(rule = self.payment_rule)

    def _optimal_bid(self, valuation, player_position):
        if not isinstance(valuation, torch.Tensor):
            valuation = torch.tensor(valuation)

        # all core-selecting rules are strategy proof for global player:
        if self.payment_rule in ['vcg', 'proxy', 'nearest_zero', 'nearest_bid',
                                   'nearest_vcg'] and player_position == 2:
            return valuation
        # local bidders:
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
        raise ValueError('optimal bid not implemented for other rules')

    def _setup_eval_environment(self):
        bne_strategies = [
            ClosureStrategy(partial(self._optimal_bid, player_position=i)) # pylint: disable=no-member
            for i in range(self.n_players)
        ]

        bne_env = AuctionEnvironment(
            mechanism=self.mechanism,
            agents=[self._strat_to_bidder(bne_strategies[i], player_position=i, batch_size=self.logging_config.eval_batch_size)
                    for i in range(self.n_players)],
            n_players=self.n_players,
            batch_size=self.logging_config.eval_batch_size,
            strategy_to_player_closure=self._strat_to_bidder
        )

        self.bne_env = bne_env

        bne_utilities_sampled = torch.tensor(
            [bne_env.get_reward(a, draw_valuations=True) for a in bne_env.agents])

        print(('Utilities in BNE (sampled):' + '\t{:.5f}' * self.n_players + '.').format(*bne_utilities_sampled))
        print("No closed form solution for BNE utilities available in this setting. Using sampled value as baseline.")
        self.bne_utilities = bne_utilities_sampled

    def _get_logdir_hierarchy(self):
        name = ['LLG', self.payment_rule]
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
    def __init__(self, experiment_config: ExperimentConfiguration, learning_config: LearningConfiguration,
                 logging_config: LoggingConfiguration, gpu_config: GPUController):

        assert experiment_config.n_players == 6, "not right number of players for setting"
        self.input_length = 2

        known_bne = False
        super().__init__(6, 4, 2, experiment_config, learning_config, logging_config, gpu_config, known_bne)

    def _setup_mechanism(self):
        self.mechanism = LLLLGGAuction(rule=self.payment_rule, core_solver=self.experiment_config.core_solver, 
                                       parallel=self.experiment_config.parallel, cuda=self.gpu_config.cuda)

    def _get_logdir_hierarchy(self):
        name = ['LLLLGG', self.payment_rule, str(self.n_players) + 'p']
        return os.path.join(*name)

    def _plot(self, plot_data, writer: SummaryWriter or None, epoch=None,
              xlim: list=None, ylim: list=None, labels: list=None,
              x_label="valuation", y_label="bid", fmts=['o'],
              figure_name: str='bid_function', plot_points=100):

        super()._plot(plot_data, writer, epoch, xlim, ylim, labels,
                      x_label, y_label, fmts, figure_name, plot_points)
        super()._plot_3d(plot_data, writer, epoch, figure_name)

class CAItemBiddingExperiment(Experiment):
    """
    A combinatorial Experiment where items are sold simultaneously but bidder might have
    complex preferences.
    """
    def __init__(self, experiment_config: ExperimentConfiguration, learning_config: LearningConfiguration,
                 logging_config: LoggingConfiguration, gpu_config: GPUController):

        self.n_items = experiment_config.n_units
        self.n_bundles = (2 ** self.n_items) - 1
        self.n_players = experiment_config.n_players
        self.payment_rule = experiment_config.payment_rule

        self.u_lo = experiment_config.u_lo
        self.u_hi = experiment_config.u_hi
        self.n_collections = experiment_config.n_collections

        self.model_sharing = experiment_config.model_sharing
        if self.model_sharing:
            self.n_models = 1
            self._bidder2model = [0] * self.n_players
        else:
            self.n_models = self.n_players
            self._bidder2model = list(range(self.n_players))

        if not hasattr(self, 'positive_output_point'):
            self.positive_output_point = torch.tensor([[self.u_hi[0]] * self.n_bundles], dtype=torch.float)

        known_bne = False

        if experiment_config.pretrain_transform is not None:
            self.pretrain_transform = experiment_config.pretrain_transform
        else:
            self.pretrain_transform = self.default_pretrain_transform

        self.input_length = self.n_bundles

        self.plot_xmin = self.plot_ymin = min(self.u_lo)
        self.plot_xmax = self.plot_ymax = max(self.u_hi)

        super().__init__(experiment_config, learning_config, logging_config, gpu_config, known_bne)
        self.using_bid_language = True

        print('\n=== Hyperparameters ===')
        for k in learning_config.learner_hyperparams.keys():
            print('{}: {}'.format(k, learning_config.learner_hyperparams[k]))
        print('=======================\n')

    def _strat_to_bidder(self, strategy, batch_size, player_position=0, cache_actions=False):
        """
        Standard strat_to_bidder method.
        """
        return CombinatorialItemBidder.uniform(
            lower=self.u_lo[player_position], upper=self.u_hi[player_position],
            strategy=strategy,
            n_items=self.n_bundles,
            n_collections=self.n_collections,
            player_position=player_position,
            batch_size=batch_size,
            cache_actions=cache_actions
        )

    def _setup_mechanism(self):
        """Setup the mechanism"""
        if self.payment_rule == 'first_price':
            self.mechanism_type = FirstPriceSealedBidAuction
        elif self.payment_rule in ('vcg', 'second_price'):
            self.mechanism_type = VickreyAuction
        else:
            raise ValueError('payment rule unknown')

        self.mechanism = self.mechanism_type(cuda=self.gpu_config.cuda)

    def _setup_eval_environment(self):
        """Setup the BNE envierment for later evaluation of the learned strategies"""
        print('no BNE known.')

    def _get_logdir_hierarchy(self):
        name = ['CAItemBidding', self.payment_rule,
                str(self.n_players) + 'players_' +
                str(self.n_collections) + 'collections_' +
                str(self.n_items) + 'units']
        return os.path.join(*name)

    def _plot(self, plot_data, writer: SummaryWriter or None, epoch=None,
              xlim: list=None, ylim: list=None, labels: list=None,
              x_label="valuation", y_label="bid", fmts=['o'],
              figure_name: str='bid_function', plot_points=100):

        # subselection of single-item valuations
        plot_data = list(plot_data)
        single_item_bundles = self.env.agents[0].transformation[:self.n_items,:].sum(0) == 1
        plot_data[0] = plot_data[0][..., single_item_bundles]

        super()._plot(plot_data, writer, epoch, xlim, ylim, labels,
                      x_label, y_label, fmts, figure_name, plot_points)

        if self.n_bundles == 2:
            super()._plot_3d(plot_data, writer, epoch, figure_name)

    def default_pretrain_transform(self, input_tensor):
        """Default pretrain transformation: truthful bidding"""
        return torch.clone(input_tensor[..., :self.n_items])
