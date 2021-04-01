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

    def get_revenue(self, env, draw_valuations: bool = False) -> float:
        """Returns the average seller revenue over a batch.

        Args:
            env (:obj:`Environment`).
            draw_valuations (bool): whether or not to redraw the valuations of
                the agents.

        Returns:
            revenue (float): average of seller revenue over a batch of games.

        """
        if draw_valuations:
            env.draw_valuations_()

        action_length = env.agents[0].n_items

        bid_profile = torch.zeros(env.batch_size, env.n_players, action_length,
                                  device=self.device)
        for pos, bid in env._generate_agent_actions():  # pylint: disable=protected-access
            bid_profile[:, pos, :] = bid
        _, payments = self.play(bid_profile)

        return payments.sum(axis=1).float().mean()

    def get_efficiency(self, env, draw_valuations: bool = False) -> float:
        """Average percentage that the actual welfare reaches of the maximal
        possible welfare over a batch.

        Args:
            env (:obj:`Environment`).
            draw_valuations (:bool:) whether or not to redraw the valuations of
                the agents.

        Returns:
            efficiency (:float:) Percentage that the actual welfare reaches of
                the maximale possible welfare. Averaged over batch.

        """
        batch_size = min(env.agents[0].valuations.shape[0], 2 ** 12)

        if draw_valuations:
            env.draw_valuations_()

        action_length = env.agents[0].n_items

        bid_profile = torch.zeros(batch_size, env.n_players, action_length,
                                  device=self.device)
        for pos, bid in env._generate_agent_actions():  # pylint: disable=protected-access
            bid_profile[:, pos, :] = bid[:batch_size, ...]
        actual_allocations, _ = self.play(bid_profile)
        actual_welfare = torch.zeros(batch_size, device=self.device)
        for a in env.agents:
            actual_welfare += a.get_welfare(
                actual_allocations[:batch_size, a.player_position],
                a.valuations[:batch_size, ...]
            )

        valuation_profile = torch.zeros(env.batch_size, env.n_players, action_length,
                                        device=self.device)
        for agent in env.agents:
            valuation_profile[:, agent.player_position, :] = agent.valuations
        maximum_allocations, _ = self.play(valuation_profile)
        maximum_welfare = torch.zeros(batch_size, device=self.device)
        for a in env.agents:
            maximum_welfare += a.get_welfare(
                maximum_allocations[:batch_size, a.player_position],
                a.valuations[:batch_size, ...]
            )

        efficiency = (actual_welfare / maximum_welfare).mean().float()
        return efficiency
