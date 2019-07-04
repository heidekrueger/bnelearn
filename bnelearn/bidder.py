# -*- coding: utf-8 -*-
"""Bidder module

This module implements players / bidders / agents in games.

"""

from abc import ABC, abstractmethod
import torch
from torch.distributions import Distribution

class Player(ABC):
    """
        A player in a game, determined by her
        - strategy
        - utility function over outcomes
    """

    def __init__(self, strategy, player_position=None, batch_size=1, cuda=True):
        self.cuda = cuda and torch.cuda.is_available()
        self.device = 'cuda' if self.cuda else 'cpu'
        self.player_position :int = player_position # None in dynamic environments!
        self.strategy = strategy
        self.batch_size = batch_size

    def get_action(self):
        """Chooses an action according to the player's strategy."""
        return self.strategy.play(batch_size=self.batch_size)

    def prepare_iteration(self):
        """ Prepares one iteration of environment-observation."""
        pass #pylint: disable=unnecessary-pass

    @abstractmethod
    def get_utility(self, **kwargs):
        """Calculates player's utility based on outcome of a game."""
        pass #pylint: disable=unnecessary-pass

class MatrixGamePlayer(Player):
    """ A player playing a matrix game"""
    def __init__(self, strategy, player_position=None, batch_size=1, cuda=True):
        super().__init__(strategy, player_position=player_position,
                         batch_size=batch_size, cuda=cuda)


    def get_utility(self, *outcome): #pylint: disable=arguments-differ
        """ get player's utility for a batch of outcomes"""
        # for now, outcome is (allocation, payment)
        _, payments = outcome
        return -payments


class Bidder(Player):
    """
        A player in an auction game. Has a distribution over valuations/types that is common knowledge.
    """
    def __init__(self,
                 value_distribution: Distribution,
                 strategy,
                 player_position=None,
                 batch_size=1,
                 n_items = 1,
                 cuda=True
                 ):
        super().__init__(strategy, player_position, batch_size, cuda)

        self.value_distribution = value_distribution
        self.n_items = n_items
        self.valuations = torch.zeros(batch_size, n_items, device=self.device)
        self.utility = torch.zeros(batch_size, device=self.device)

    ### Alternative Constructors #############
    @classmethod
    def uniform(cls, lower, upper, strategy, **kwargs):
        """Constructs a bidder with uniform valuation prior."""
        dist = torch.distributions.uniform.Uniform(low = lower, high=upper)
        return cls(dist, strategy, **kwargs)

    @classmethod
    def normal(cls, mean, stddev, strategy, **kwargs):
        """Constructs a bidder with Gaussian valuation prior."""
        dist = torch.distributions.normal.Normal(loc = mean, scale = stddev)
        return cls(dist, strategy, **kwargs)

    ### Members ####################
    def prepare_iteration(self):
        self.draw_valuations_()

    def draw_valuations_(self):
        """Sample a new batch of valuations from the Bidder's prior.
           Negative draws will be clipped at 0.0!
        """
        # If in place sampling is available for our distribution, use it!
        # This will save time for memory allocation and/or copying between devices
        # As sampling from general torch.distribution is only available on CPU.
        # (might mean adding more boilerplate code here if specific distributions are desired

        # uniform
        if isinstance(self.value_distribution, torch.distributions.uniform.Uniform):
            self.valuations.uniform_(self.value_distribution.low, self.value_distribution.high)
        # gaussian
        elif isinstance(self.value_distribution, torch.distributions.normal.Normal):
            self.valuations.normal_(mean = self.value_distribution.loc, std = self.value_distribution.scale).relu_()
        # add additional internal in-place samplers as needed!
        else:
            # slow! (sampling on cpu then copying to GPU)
            self.valuations = self.value_distribution.rsample(self.valuations.size()).to(self.device).relu()

        return self.valuations

    def get_utility(self, allocations, payments): #pylint: disable=arguments-differ
        """
        For a batch of allocations and payments return the player's utilities.
        """

        assert allocations.dim() == 2 # batch_size x items
        assert payments.dim() == 1 # batch_size

        self.utility = (self.valuations * allocations).sum(dim=1) - payments
        return self.utility

    def get_action(self):
        inputs = self.valuations.view(self.batch_size, -1)
        return self.strategy.play(inputs)
