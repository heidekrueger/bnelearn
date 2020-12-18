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

    def get_efficiency(self, env, draw_valuations: bool = False) -> float:
        """Returns the percentage of efficiently allocated outcomes over a
        batch.

        Args:
            env (:obj:`Environment`).
            draw_valuations (bool): whether or not to redraw the valuations of
                the agents.

        Returns:
            efficiency (float): precentage of efficiently allocated outcomes.

        """
        if draw_valuations:
            env.draw_valuations_()

        action_length = env.agents[0].n_items

        bid_profile = torch.zeros(env.batch_size, env.n_players, action_length,
                                  device=self.device)
        for pos, bid in env._generate_agent_actions():  # pylint: disable=protected-access
            bid_profile[:, pos, :] = bid
        actual_allocations, _ = self.play(bid_profile)

        valuation_profile = torch.zeros(env.batch_size, env.n_players, action_length,
                                        device=self.device)
        for agent in env.agents:
            valuation_profile[:, agent.player_position, :] = agent.valuations
        fair_allocations, _ = self.play(valuation_profile)

        # Count no. of batches where all items are equally distributed over all agents
        equal_allocations = actual_allocations == fair_allocations
        efficiency = torch.all(torch.all(equal_allocations, axis=2), axis=1)
        return efficiency.float().mean()
