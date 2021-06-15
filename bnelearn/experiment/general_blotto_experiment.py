"""
This file implements the experimental setting for the General Blotto game and its variations. 
"""

import os
import torch

from bnelearn.experiment.experiment import Experiment
from bnelearn.experiment.configurations import (ExperimentConfig)
from bnelearn.mechanism.general_blotto import GeneralBlotto
from bnelearn.bidder import BlottoBidder

class GeneralBlottoExperiment(Experiment):

    """
    Experiment class for the General Blotto games.
    """

    def __init__(self, config: ExperimentConfig):

        self.config = config
        self.n_players = self.config.setting.n_players
        self.n_items = 3
        self.input_length = 3
        self.positive_output_point = None
        self.u_lo = float(config.setting.u_lo)
        self.u_hi = float(config.setting.u_hi)
        self.common_prior = torch.distributions.uniform.Uniform(low=self.u_lo, high=self.u_hi) # TODO: check meaningful prior
        self.budget_ratio = config.setting.budget_ratio
        self.normalize_valuations = config.setting.normalize_valuations

        self.model_sharing = self.config.learning.model_sharing

        if self.model_sharing:
            self.n_models = 1
            self._bidder2model = [0] * self.n_players
        else:
            self.n_models = self.n_players
            self._bidder2model = list(range(self.n_players))

        # initialize bidders budgets
        self.budgets = [3, 3 * self.budget_ratio]

        super().__init__(config=config)

    def _get_logdir_hierarchy(self):
        name = ['general_blotto']
        return os.path.join(*name)


    def _setup_mechanism(self):
        print('Using General Blotto mechanism')
        self.mechanism = GeneralBlotto(cuda=self.hardware.cuda)


    def _strat_to_bidder(self, strategy, batch_size, player_position=None):

        # Assign each bidder a budget
        budget = self.budgets[player_position]

        return BlottoBidder(self.common_prior, strategy, player_position, batch_size, self.n_items, budget = budget,
                            normalize_valuations = self.normalize_valuations)

