"""This module implements Experiments on single items in double auctions"""

import os
import warnings
from abc import ABC
from typing import Callable, List
from functools import partial
import torch
import numpy as np
from scipy import integrate, interpolate
from scipy import optimize

from bnelearn.bidder import Bidder
from bnelearn.environment import AuctionEnvironment
from bnelearn.experiment import Experiment
from bnelearn.experiment.configurations import ExperimentConfig
from bnelearn.sampler import (SymmetricIPVSampler, UniformSymmetricIPVSampler)
from bnelearn.strategy import ClosureStrategy
from bnelearn.mechanism import kDoubleAuction, VickreyDoubleAuction
from bnelearn.strategy import ClosureStrategy
from bnelearn.experiment.equilibria import (
    truthful_bid,
    bne_bilateral_bargaining_uniform_linear,
    bne_bilateral_bargaining_uniform_symmetric)


class DoubleAuctionSingleItemExperiment(Experiment, ABC):

    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.n_items = 1

        self.n_buyers = self.config.setting.n_buyers
        self.n_sellers = self.config.setting.n_sellers
        self.n_players = self.n_buyers + self.n_sellers
        assert self.n_players == self.config.setting.n_players

        self.k = self.config.setting.k
        self.observation_size = self.valuation_size = self.action_size = 1

        if not hasattr(self, 'payment_rule'):
            self.payment_rule = self.config.setting.payment_rule
        if not hasattr(self, 'valuation_prior'):
            self.valuation_prior = 'unknown'

        if self.config.logging.eval_batch_size < 2 ** 20:
            print(f"Using small eval_batch_size of {self.config.logging.eval_batch_size}. Use at least 2**22 for proper experiment runs!")

        self.model_sharing = self.config.learning.model_sharing

        if self.model_sharing:
            self.n_models = 2
            self._bidder2model: List[int] = [0] * self.n_buyers \
                                            + [1] * (self.n_sellers)
        else:
            self.n_models = self.n_buyers + self.n_sellers
            self._bidder2model: List[int] = list(range(self.n_buyers + self.n_sellers))

        super().__init__(config=config)

    def _setup_mechanism(self):
        if self.payment_rule == 'k_price':
            self.mechanism = kDoubleAuction(
                cuda=self.hardware.cuda, k_value = self.k,
                n_buyers=self.n_buyers, n_sellers=self.n_sellers
                )
        elif self.payment_rule == 'vcg':
            self.mechanism = VickreyDoubleAuction(
                cuda=self.hardware.cuda, n_buyers=self.n_buyers,
                n_sellers=self.n_sellers
                )
        else:
            raise ValueError('Invalid Mechanism type!')

    @staticmethod
    def get_risk_profile(risk) -> str:
        """Used for logging and checking existence of bne"""
        return f'risk_{risk}'

    def _get_model_names(self):
        if self.model_sharing or self.n_players == 2:
            return ['buyer'  + ('s' if self.n_buyers  > 1 else ''),
                    'seller' + ('s' if self.n_sellers > 1 else '')]
        else:
            return [f'buyer {i}'  for i in range(self.n_buyers) ] \
                 + [f'seller {i}' for i in range(self.n_sellers)]

    def _strat_to_bidder(self, strategy, batch_size, player_position=0,
                         enable_action_caching=False) -> Bidder:
        seller = player_position > self.n_buyers - 1
        return Bidder(strategy, player_position, batch_size,
                      enable_action_caching=enable_action_caching,
                      risk=self.risk, seller=seller)


class DoubleAuctionSymmetricPriorSingleItemExperiment(DoubleAuctionSingleItemExperiment):
    """A Single Item Experiment that has the same valuation prior for all
    participating bidders.
    """

    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.n_items = 1

        self.n_buyers = self.config.setting.n_buyers
        self.n_sellers = self.config.setting.n_sellers
        self.n_players = self.n_buyers + self.n_sellers
        assert self.n_players == self.config.setting.n_players

        self.k = self.config.setting.k

        self.common_prior = self.config.setting.common_prior
        self.positive_output_point = torch.stack([self.common_prior.mean] * self.n_items)

        self.risk = float(self.config.setting.risk)
        self.risk_profile = self.get_risk_profile(self.risk)

        super().__init__(config=config)

    def _check_and_set_known_bne(self):

        if self.payment_rule in ['second_price', 'vcg', 'vickrey_price']:
            self._optimal_bid =  [truthful_bid]
            return True

        else:
            warnings.warn('optimal bid not implemented for this type')
            return False

    def _setup_eval_environment(self):
        """Determines whether a bne exists and sets up eval environment."""
        assert self.known_bne
        assert hasattr(self, '_optimal_bid')

        bne_strategies = [None] * len(self._optimal_bid)
        self.bne_env = [None] * len(self._optimal_bid)
        self.bne_utilities = [None] * len(self._optimal_bid)
        for i, bid_function in enumerate(self._optimal_bid):
            bne_strategies[i] = [ClosureStrategy(partial(bid_function, player_position=j))
                                 for j in range(self.n_players)]

            self.bne_env[i] = AuctionEnvironment(
                mechanism=self.mechanism,
                agents=[self._strat_to_bidder(bne_strategies[i][p], player_position=p,
                                              batch_size=self.logging.eval_batch_size,
                                              enable_action_caching=self.config.logging.cache_eval_actions)
                        for p in range(self.n_players)],
                valuation_observation_sampler=self.sampler,
                n_players=self.n_players,
                batch_size=self.logging.eval_batch_size,
                strategy_to_player_closure=self._strat_to_bidder
            )

            self.bne_utilities[i] = torch.tensor(
                [self.bne_env[i].get_reward(a, redraw_valuations=True) for a in self.bne_env[i].agents])

            print(('Utilities in BNE{} (sampled):' + '\t{:.5f}' * self.n_players + '.') \
                .format(i + 1,*self.bne_utilities[i]))


class DoubleAuctionUniformSymmetricPriorSingleItemExperiment(DoubleAuctionSymmetricPriorSingleItemExperiment):
    """Double Auction Uniform Symmetric Prior Experiment for unit demand: Each
    seller has one item and each buyer can buy max. one item.
    """
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.n_buyers = self.config.setting.n_buyers
        self.n_sellers = self.config.setting.n_sellers
        self.k = self.config.setting.k

        assert self.config.setting.u_lo is not None, """Prior boundaries not specified!"""
        assert self.config.setting.u_hi is not None, """Prior boundaries not specified!"""

        self.valuation_prior = 'uniform'
        self.u_lo = torch.tensor(self.config.setting.u_lo[0], dtype=torch.float32,
            device=self.config.hardware.device)
        self.u_hi = torch.tensor(self.config.setting.u_hi[0], dtype=torch.float32,
            device=self.config.hardware.device)
        self.config.setting.common_prior = \
            torch.distributions.uniform.Uniform(low=self.u_lo, high=self.u_hi)

        self.plot_xmin = self.u_lo.cpu().numpy()
        self.plot_xmax = self.u_hi.cpu().numpy()
        self.plot_ymin = 0
        self.plot_ymax = self.u_hi.cpu().numpy() * 1.05

        super().__init__(config=config)

    def _setup_sampler(self):
        self.sampler = SymmetricIPVSampler(
            self.common_prior, self.n_players, self.valuation_size,
            self.config.learning.batch_size, self.config.hardware.device
        )

    def _check_and_set_known_bne(self):

        if self.payment_rule == 'k_price' and self.risk == 1.0:
            self._optimal_bid = [bne_bilateral_bargaining_uniform_linear(
                self.config, self.u_lo, self.u_hi)]
            if self.k == 0.5 and self.u_lo == 0 and self.u_hi == 1:
                symmetric_bnes_to_plot = [0.25, 0.45]
                if self.config.setting.pretrain_transform is not None:
                    symmetric_bnes_to_plot += [self.config.setting.pretrain_transform]
                self._optimal_bid += bne_bilateral_bargaining_uniform_symmetric(self.config, symmetric_bnes_to_plot)
            return True

        return super()._check_and_set_known_bne()

    def pretrain_transform(self, player_position: int) -> callable:
        """Transformation during pretraining"""
        g_05 = self.setting.pretrain_transform

        if g_05 is None:
            return None

        return lambda v: bne_bilateral_bargaining_uniform_symmetric(
            self.config, [g_05])[0](v, player_position)

    def _get_logdir_hierarchy(self):

        if self.payment_rule == 'k_price':
            name = ['double_auction','single_item', self.payment_rule, str(self.k), self.valuation_prior,
                    'symmetric', self.risk_profile, str(self.n_buyers) + 'b' + str(self.n_sellers) + 's']
        else:
            name = ['double_auction','single_item', self.payment_rule, self.valuation_prior,
                    'symmetric', self.risk_profile, str(self.n_buyers) + 'b' + str(self.n_sellers) + 's']
        return os.path.join(*name)
