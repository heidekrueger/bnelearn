import os
import warnings
from abc import ABC
from functools import partial
from typing import List

import torch
from scipy import integrate
from bnelearn.bidder import Bidder
from bnelearn.environment import AuctionEnvironment
from bnelearn.experiment import Experiment
from bnelearn.experiment.configurations import ExperimentConfig
from bnelearn.experiment.equilibria import (
    symmetric_war_of_attrition_uniform,
    common_value_lottery_contest)
from bnelearn.mechanism import Contest
from bnelearn.sampler import (AffiliatedValuationObservationSampler,
                              CompositeValuationObservationSampler,
                              MineralRightsValuationObservationSampler,
                              SymmetricIPVSampler, UniformSymmetricIPVSampler, CommonValueSampler)
from bnelearn.strategy import ClosureStrategy
from bnelearn.util.distribution_util import copy_dist_to_device

class ContestExperiment(Experiment, ABC):
    
    def __init__(self, config: ExperimentConfig):
        
        self.config = config

        self.n_players = self.config.setting.n_players

        if self.config.logging.eval_batch_size < 2 ** 20:
            print(f"Using small eval_batch_size of {self.config.logging.eval_batch_size}. Use at least 2**22 for proper experiment runs!")

        self.payment_rule = self.config.setting.payment_rule
        self.csf = self.config.setting.csf

        # model sharing for symmetric strategies
        self.model_sharing = config.learning.model_sharing
        if self.model_sharing:
            self.n_models = 1
            self._bidder2model = [0] * self.n_players
        else:
            self.n_models = self.n_players
            self._bidder2model = list(range(self.n_players))

        super().__init__(config=config)

    # @staticmethod
    # def get_risk_profile(risk) -> str:
    #     """Used for logging and checking existence of bne"""
    #     if risk == 1.0:
    #         return 'risk_neutral'
    #     elif risk == 0.5:
    #         return 'risk_averse'
    #     else:
    #         return 'other'

class SingleItemContestExperiment(ContestExperiment, ABC):

    def __init__(self, config: Experiment):

        # Modelling settings
        # TODO Markus - model sharing einbinden hier
        self.n_models = self.n_players
        self._bidder2model = list(range(self.n_players))

        self.observation_size = self.valuation_size = self.action_size = 1

        super().__init__(config)

    def _setup_mechanism(self):
        self.mechanism = Contest(cuda=self.hardware.cuda, payment_rule=self.payment_rule, csf=self.csf)

class SingleItemConstantPriorContest(SingleItemContestExperiment):

    def __init__(self, config: Experiment):

        # Setting 
        self.n_items = 1
        self.observation_size = self.valuation_size = self.action_size = 1


        self.common_value = config.setting.common_value

        self.n_players = config.setting.n_players
        self.positive_output_point = torch.tensor([self.common_value] * self.n_items)

        super().__init__(config)

    def _get_logdir_hierarchy(self):
        name = [f'contest/symmetric/single_item/constant/{self.csf}/{self.payment_rule}/']
        return os.path.join(*name)

    def _setup_sampler(self):

        supp_bounds = torch.ones([1, self.valuation_size, 2]) * self.common_value

        bidder_samplers = [
            CommonValueSampler(1, self.valuation_size, supp_bounds, self.learning.batch_size, self.hardware.device)
            for i in range(self.n_players)
        ]

        self.sampler = CompositeValuationObservationSampler(
            self.n_players, self.valuation_size, self.observation_size, bidder_samplers,
            self.learning.batch_size, self.hardware.device
        )

    def _strat_to_bidder(self, strategy, batch_size, player_position=0, enable_action_caching=False):
        # Do we need the prior limits here?
        return Bidder(strategy, player_position, batch_size, enable_action_caching=enable_action_caching)

    def _check_and_set_known_bne(self):
        self._optimal_bid = partial(common_value_lottery_contest, n_players=self.n_players)
        return True

    def _get_analytical_bne_utility(self) -> torch.Tensor:
        """Calculates utility in BNE from known closed-form solution (possibly using numerical integration)"""
        bne_utility = self.common_value/(self.n_players**2)

        return torch.tensor(bne_utility, device=self.hardware.device)

    def _setup_eval_environment(self):
        """Determines whether a bne exists and sets up eval environment."""

        assert self.known_bne
        assert  hasattr(self, '_optimal_bid')
        print("Setting up the evaluation environment..." + \
            "\tDepending on your and hardware and the eval_batch_size, this may take a while," +\
                "-- sequential numeric integration on the cpu is required in this environment.")

        bne_strategy = ClosureStrategy(self._optimal_bid, parallel=0, mute=True)


        # define bne agents once then use them in all runs
        self.bne_env = AuctionEnvironment(
            mechanism = self.mechanism,
            agents=[self._strat_to_bidder(bne_strategy,
                                          player_position=i,
                                          batch_size=self.logging.eval_batch_size,
                                          enable_action_caching=self.logging.cache_eval_actions)
                    for i in range(self.n_players)],
            valuation_observation_sampler = self.sampler,
            batch_size=self.logging.eval_batch_size,
            n_players=self.n_players,
            strategy_to_player_closure=self._strat_to_bidder
        )

        bne_utility_analytical = self._get_analytical_bne_utility()

        self.bne_utility = bne_utility_analytical
        self.bne_utilities = [self.bne_utility] * self.n_models


class SingleItemUniFormPriorContest(SingleItemContestExperiment):

    def __init__(self, config: Experiment):

        # Type information
        if not isinstance(self.config.setting.u_lo, list):
            self.u_lo = [float(self.config.setting.u_lo)] * self.n_players
        else:
            self.u_lo: List[float] = [float(self.config.setting.u_lo[i]) for i in range(self.n_players)]        
        
        self.u_hi: List[float] = [float(self.config.setting.u_hi[i]) for i in range(self.n_players)]

        super().__init__(config)

    def _get_logdir_hierarchy(self):
        name = [f'contest/symmetric/single_item/uniform/{self.csf}/{self.payment_rule}/']
        return os.path.join(*name)

    def _setup_sampler(self):

        default_batch_size = self.learning.batch_size
        device = self.hardware.device
        # setup individual samplers for each bidder
        bidder_samplers = [
            UniformSymmetricIPVSampler(
                self.u_lo[i], self.u_hi[i], 1,
                self.valuation_size, default_batch_size, device)
            for i in range(self.n_players)]

        self.sampler = CompositeValuationObservationSampler(
            self.n_players, self.valuation_size, self.observation_size, bidder_samplers,
            default_batch_size, device
            )

    def _strat_to_bidder(self, strategy, batch_size, player_position=0, enable_action_caching=False):
        return Bidder(strategy, player_position, batch_size, enable_action_caching=enable_action_caching,
                      n_players=self.n_players, prior_limits=[self.u_lo, self.u_hi])

    def _check_and_set_known_bne(self):
        # TODO Markus - add equilibria here
        return False