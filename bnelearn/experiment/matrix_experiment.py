"""This module implements Experiments on matrix games"""

import os
import torch

from bnelearn.experiment import Experiment
from bnelearn.experiment.configurations import ExperimentConfig
from bnelearn.environment import MatrixGameEnvironment
from bnelearn.mechanism.matrix_games import JordanGame
from bnelearn.bidder import MatrixGamePlayer
from bnelearn.strategy import MatrixGameStrategy


class JordanExperiment(Experiment):
    """Experiment setup for the Jordan game."""
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.n_actions = 2

        assert config.setting.n_players == 3
        self.n_players = config.setting.n_players
        self.valuation_prior = 'uniform'

        self.model_sharing = self.config.learning.model_sharing
        if self.model_sharing:
            self.n_models = 1
            self._bidder2model = [0] * self.n_players
        else:
            self.n_models = self.n_players
            self._bidder2model = list(range(self.n_players))

        super().__init__(config=config)

    def _setup_mechanism(self):
        self.mechanism = JordanGame(cuda=self.hardware.cuda)

    def _setup_learning_environment(self):
        self.env = MatrixGameEnvironment(game=self.mechanism,
                                         agents=self.bidders,
                                         n_players=self.n_players,
                                         batch_size=self.learning.batch_size,
                                         strategy_to_player_closure=self._strat_to_bidder
                                         )

    def _check_and_set_known_bne(self):
        return False # TODO Nils

    def _get_analytical_bne_utility(self):
        pass

    def _setup_eval_environment(self):
        pass

    def _setup_bidders(self):
        """
        1. Create and save the models and bidders
        2. Save the model parameters
        """
        print('Setting up bidders...')
        self.models = [None] * self.n_models

        for i in range(len(self.models)):
            self.models[i] = MatrixGameStrategy(n_actions=self.n_actions) \
                .to(self.hardware.device)

        self.bidders = [
            self._strat_to_bidder(self.models[m_id], batch_size=self.learning.batch_size,
                                  player_position=i)
            for i, m_id in enumerate(self._bidder2model)
        ]

        self.n_parameters = [sum([p.numel() for p in model.parameters()]) for model in
                             self.models]

        if self.learning.pretrain_iters > 0:
            print('\tno pretraining for matrix games.')

    def _strat_to_bidder(self, strategy, batch_size, player_position=0, cache_actions=False):
        if cache_actions:
            print('`cache_actions` not supported!')
        player = MatrixGamePlayer(strategy=strategy, player_position=player_position,
                                  batch_size=batch_size, cuda=True)
        return player

    def _get_logdir_hierarchy(self):
        name = ['jordan']
        return os.path.join(*name)
