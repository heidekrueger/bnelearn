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

from bnelearn.mechanism import AverageAuction, VickreyDoubleAuction
from bnelearn.strategy import ClosureStrategy


###############################################################################
#######   Known equilibrium bid functions                                ######
###############################################################################
# Define known BNE functions top level, so they may be pickled for parallelization
# These are called millions of times, so each implementation should be
# setting specific, i.e. there should be NO setting checks at runtime.

def _optimal_bid_buyer_average(valuation: torch.Tensor, **kwargs) -> torch.Tensor:
    return (2/3 * valuation) + 1/12 

def _optimal_bid_seller_average(valuation: torch.Tensor, **kwargs) -> torch.Tensor:
    return (2/3 * valuation) + 1/4 

def _truthful_bid(valuation: torch.Tensor, **kwargs) -> torch.Tensor:
    return valuation


class DoubleAuctionSingleItemExperiment(Experiment, ABC):

    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.n_players = self.config.setting.n_players
        self.n_items = 1
        self.n_buyers = self.config.setting.n_buyers
        self.n_sellers = self.config.setting.n_sellers
        self.input_length = 1

        if not hasattr(self, 'payment_rule'):
            self.payment_rule = self.config.setting.payment_rule
        if not hasattr(self, 'valuation_prior'):
            self.valuation_prior = 'unknown'

        if self.config.logging.eval_batch_size < 2 ** 20:
            print(f"Using small eval_batch_size of {self.config.logging.eval_batch_size}. Use at least 2**22 for proper experiment runs!")
        
        
        self.model_sharing = self.config.learning.model_sharing

        assert self.model_sharing == False

        self.n_models = 2
        
        self._bidder2model: List[int] = [0] * self.n_buyers \
                                            + [1] * (self.n_sellers)
        
        super().__init__(config=config)

    def _setup_mechanism(self):
        if self.payment_rule == 'first_price':
            self.mechanism = AverageAuction(cuda=self.hardware.cuda, 
                                            n_buyers=self.n_buyers, n_sellers=self.n_sellers)
        elif self.payment_rule == 'second_price':
            self.mechanism = VickreyDoubleAuction(cuda=self.hardware.cuda, 
                                                  n_buyers=self.n_buyers, n_sellers=self.n_sellers)
        else:
            raise ValueError('Invalid Mechanism type!')

    @staticmethod
    def get_risk_profile(risk) -> str:
        """Used for logging and checking existence of bne"""
        if risk == 1.0:
            return 'risk_neutral'
        elif risk == 0.5:
            return 'risk_averse'
        else:
            return 'other'
    
    def _get_model_names(self):
        return ['buyers', 'sellers']
    
    def _strat_to_bidder(self, strategy, batch_size, player_position=0, cache_actions=False):

        seller = player_position > self.n_buyers - 1

        return Bidder(self.common_prior, strategy, player_position, batch_size, cache_actions=cache_actions,
                      risk=self.risk, seller=seller)



class DoubleAuctionSymmetricPriorSingleItemExperiment(DoubleAuctionSingleItemExperiment):
    """A Single Item Experiment that has the same valuation prior for all participating bidders.
    """

    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.n_players = self.config.setting.n_players
        self.n_items = 1
        self.n_buyers = self.config.setting.n_buyers
        self.n_sellers = self.config.setting.n_sellers

        self.common_prior = self.config.setting.common_prior
        self.positive_output_point = torch.stack([self.common_prior.mean] * self.n_items)

        self.risk = float(self.config.setting.risk)
        self.risk_profile = self.get_risk_profile(self.risk)

        super().__init__(config=config)
    
    def _optimal_bid(self, valuation, player_position):
        
        if not isinstance(valuation, torch.Tensor):
            valuation = torch.tensor(valuation)
        
        if self.payment_rule == 'first_price':
            if player_position > self.n_buyers - 1:
                return _optimal_bid_seller_average(valuation)
            else:
                return _optimal_bid_buyer_average(valuation)
        
        elif self.payment_rule == 'second_price':
            return _truthful_bid(valuation)
        
        else:
            warnings.warn('optimal bid not implemented for this type')
            self.known_bne = False


    def _check_and_set_known_bne(self):
        if self.payment_rule in \
            ['first_price', 'second_price']:
            return True
        else:
            # no bne found, defer to parent
            return super()._check_and_set_known_bne()


    def _setup_eval_environment(self):
        """Determines whether a bne exists and sets up eval environment."""

        assert  self.known_bne
        assert  hasattr(self, '_optimal_bid')

        # TODO: parallelism should be taken from elsewhere. Should be moved to config. Assigned @Stefan
        # n_processes_optimal_strategy = 44 if self.valuation_prior != 'uniform' and \
        #                                         self.payment_rule != 'second_price' else 0
      
        bne_strategies = [
            ClosureStrategy(partial(self._optimal_bid, player_position=i))
            for i in range(self.n_players)]

        agents = [self._strat_to_bidder(bne_strategies[i], player_position=i, 
                                        batch_size=self.config.logging.eval_batch_size)
                 for i in range(self.n_players)]

        self.known_bne = True

        self.bne_env = AuctionEnvironment(
            self.mechanism,
            agents=agents,
            n_players=self.n_players,
            batch_size=self.logging.eval_batch_size,
            strategy_to_player_closure=self._strat_to_bidder
        )

        self.bne_utilities = torch.tensor(
            [self.bne_env.get_reward(a, draw_valuations=True) for a in self.bne_env.agents])
        
        print(('Utilities in BNE (sampled):' + '\t{:.5f}' * self.n_players + '.').format(*self.bne_utilities))
        print("No closed form solution for BNE utilities available in this setting. Using sampled value as baseline.")


    def _get_logdir_hierarchy(self):
        name = ['double_auction','single_item', self.payment_rule, self.valuation_prior,
                'symmetric', self.risk_profile, str(self.n_players) + 'p']
        return os.path.join(*name)


class DoubleAuctionUniformSymmetricPriorSingleItemExperiment(DoubleAuctionSymmetricPriorSingleItemExperiment):

    def __init__(self, config: ExperimentConfig):
        self.config = config

        assert self.config.setting.u_lo is not None, """Prior boundaries not specified!"""
        assert self.config.setting.u_hi is not None, """Prior boundaries not specified!"""

        self.valuation_prior = 'uniform'
        self.u_lo = self.config.setting.u_lo
        self.u_hi = self.config.setting.u_hi
        self.config.setting.common_prior = \
            torch.distributions.uniform.Uniform(low=self.u_lo, high=self.u_hi)

        # ToDO Implicit list to float type conversion
        self.plot_xmin = self.u_lo
        self.plot_xmax = self.u_hi
        self.plot_ymin = 0
        self.plot_ymax = self.u_hi * 1.05

        super().__init__(config=config)