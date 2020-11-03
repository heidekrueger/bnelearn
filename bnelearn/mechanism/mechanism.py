# -*- coding: utf-8 -*-

"""
This module implements games such as matrix games and auctions.
"""

from abc import ABC, abstractmethod

# pylint: disable=E1102
import torch


class Game(ABC):
    """
    Base class for any kind of games
    """

    def __init__(self, cuda: bool = True):
        self.cuda = cuda and torch.cuda.is_available()
        self.device = 'cuda' if self.cuda else 'cpu'

    @abstractmethod
    def play(self, action_profile):
        """Play the game!"""
        # get actions from players and define outcome
        raise NotImplementedError()


class Mechanism(Game, ABC):
    """
    Auction Mechanism - Interpreted as a Bayesian game.
    A Mechanism collects bids from all players, then allocates available
    items as well as payments for each of the players.
    """

    def play(self, action_profile):
        return self.run(bids=action_profile)

    @abstractmethod
    def run(self, bids):
        """Alias for play for auction mechanisms"""
        raise NotImplementedError()

    def get_revenue(self, env):
        """
        Returns the average seller revenue over a batch.
        """
        bid_profile = torch.zeros(env.batch_size, env.n_players,
                                  env.agents[0].n_items, device=self.device)
        for pos, bid in env._generate_agent_actions(): # pylint: disable=protected-access
            bid_profile[:, pos, :] = bid
        _, payments = self.play(bid_profile)

        return payments.sum(axis=1).float().mean()
