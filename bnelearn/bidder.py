# -*- coding: utf-8 -*-
"""Bidder module

This module implements players / bidders / agents in games.

"""

from abc import ABC, abstractmethod
import warnings
import torch
from bnelearn.strategy import (Strategy, MatrixGameStrategy,
                               FictitiousPlayStrategy, FictitiousNeuralPlayStrategy)


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

    @abstractmethod
    def get_action(self):
        """Chooses an action according to the player's strategy."""

    @abstractmethod
    def get_utility(self, *args, **kwargs):
        """Calculates player's utility based on outcome of a game."""

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

    def get_action(self):
        if (isinstance(self.strategy, MatrixGameStrategy) or isinstance(self.strategy, FictitiousNeuralPlayStrategy)):
            return self.strategy.play(batch_size=self.batch_size)
        if isinstance(self.strategy, FictitiousPlayStrategy):
            return self.strategy.play(self.player_position)

        raise ValueError("Invalid Strategy Type for Matrix game: {}".format(type(self.strategy)))

class Bidder(Player):
    """A player in an auction game. Has a distribution over valuations/types
    that is common knowledge. These valuations correspond to the ´n_items´
    available.

    Attributes:
        batch_size: corresponds to the number of individual auctions.
        descending_valuations: if is true, the valuations will be returned
            in decreasing order.
        enable_action_caching: determines whether actions should be cached and
            retrieved from memory, rather than recomputed as long as valuations
            haven't changed.
        TODO ...

    """

    def __init__(self,
                 strategy: Strategy,
                 player_position: torch.Tensor = None,
                 batch_size: int = 1,
                 valuation_size: int = 1,
                 observation_size: int =1,
                 bid_size: int = 1,
                 cuda: str = True,
                 enable_action_caching: bool = False,
                 risk: float = 1.0
                 ):

        super().__init__(strategy, player_position, batch_size, cuda)

        self.valuation_size = valuation_size
        self.observation_size = observation_size
        self.bid_size = bid_size

        self.risk = risk
        self._enable_action_caching = enable_action_caching
        self._cached_observations_changed = False # true if new observations drawn since actions calculated
        self._cached_observations = None
        self._cached_valuations_changed = False # true if new observations drawn since actions calculated
        self._cached_valuations = None

        if self._enable_action_caching:
            self._cached_valuations = torch.zeros(batch_size, valuation_size, device=self.device)
            self._cached_observations = torch.zeros(batch_size, observation_size, device=self.device)
            self._cached_actions = torch.zeros(batch_size, bid_size, device=self.device)

    @property
    def cached_observations(self):
        return self._cached_observations

    @cached_observations.setter
    def cached_observations(self, new_value: torch.Tensor):
        """When manually setting observations, make sure that the _observations_changed flag is set correctly."""
        if new_value.shape != self._cached_observations.shape:
            warnings.warn("New observations have different shape than specified in Bidder object!")
        if (new_value.dtype, new_value.device) != (self._cached_observations.dtype, self._cached_observations.device):
            warnings.warn(
                "New observations have different dtype and/or device than bidder. Converting to {},{}".format(
                    self._cached_observations.device, self._cached_observations.dtype)
                )

        if not new_value.equal(self._cached_observations):
            self._cached_observations = new_value.to(self._cached_observations.device, self._cached_observations.dtype)
            self._cached_observations_changed = True

    @property
    def cached_valuations(self):
        return self._cached_valuations

    @cached_valuations.setter
    def cached_valuations(self, new_value: torch.Tensor):
        """When manually setting valuations, make sure that the _valuations_changed flag is set correctly."""
        if new_value.shape != self._cached_valuations.shape:
            warnings.warn("New valuations have different shape than specified in Bidder object!")
        if (new_value.dtype, new_value.device) != (self._cached_valuations.dtype, self._cached_valuations.device):
            warnings.warn(
                "New valuations have different dtype and/or device than bidder. Converting to {},{}".format(
                    self._cached_valuations.device, self._cached_valuations.dtype)
                )

        if not new_value.equal(self._cached_valuations):
            self._cached_valuations = new_value.to(self._cached_valuations.device, self._cached_valuations.dtype)
            self._cached_valuations_changed = True

    def get_utility(self, allocations, payments, valuations=None):
        """
        For a batch of valuations, allocations, and payments of the bidder,
        return their utility.

        Can handle multiple batch dimensions, e.g. for allocations a shape of
        ( outer_batch_size, inner_batch_size, n_items). These batch dimensions are kept in returned
        payoff.
        """

        if valuations is None:
            valuations = self._cached_valuations

        welfare = self.get_welfare(allocations, valuations)
        payoff = welfare - payments

        if self.risk == 1.0:
            return payoff

        # payoff^alpha not well defined in negative domain for risk averse agents
        # the following is a memory-saving implementation of
        #return payoff.relu()**self.risk - (-payoff).relu()**self.risk
        return payoff.relu().pow_(self.risk).sub_(payoff.neg_().relu_().pow_(self.risk))

    def get_welfare(self, allocations, valuations=None):
        """For a batch of allocations return the player's welfare.

        If valuations are not specified, welfare is calculated for
        `self.valuations`.

        Can handle multiple batch dimensions, e.g. for valuations a shape of
        (..., batch_size, n_items). These batch dimensions are kept in returned
        welfare.
        """
        assert allocations.dim() >= 2 # [batch_sizes] x items
        if valuations is None:
            valuations = self._cached_valuations

        item_dimension = valuations.dim() - 1
        welfare = (valuations * allocations).sum(dim=item_dimension)

        return welfare

    def get_action(self, observations = None, deterministic: bool = False):
        """Calculate action from given observations, or retrieve from cache"""

        if self._enable_action_caching and not self._cached_observations_changed and \
            (observations is None or torch.equal(observations, self._cached_observations)):

            return self._cached_actions

        if observations is None:
            assert self._enable_action_caching, \
                "Action caching is disabled but no observation argument was provided to get_actions."
            # No observations have been given, but _cached_observations_changed
            # use cached observations but recompute actions
            observations = self._cached_observations

        #TODO: there was a reshaping here added by Nils (to self.batch_size, -1). This is problematic, should be done
        # in strategy, not here. Strategy should always map complete obs to complete actions.
        inputs = observations
        # for cases when n_observations != input_length (e.g. Split-Award Auctions, combinatorial auctions with bid languages)
        # TODO: generalize this, see #82. https://gitlab.lrz.de/heidekrueger/bnelearn/issues/82
        if hasattr(self.strategy, 'input_length') and self.strategy.input_length != self.observation_size:
            warnings.warn("Strategy expects shorter input_length than n_items. Truncating observations...")
            dim = self.strategy.input_length
            inputs = inputs[:,:dim]

        actions = self.strategy.play(inputs, deterministic=deterministic)

        if self._enable_action_caching:
            self.cached_observations = observations
            self._cached_actions = actions
            # we have updated the cached actions, so we can disable the
            # flag that they need to be recomputed.
            self._cached_observations_changed = False

        return actions


class ReverseBidder(Bidder):
    """
    Bidder that has reversed utility (*(-1)) as valuations correspond to
    their costs and payments to what they get payed.
    """
    def __init__(self, efficiency_parameter=None, **kwargs):
        self.efficiency_parameter = efficiency_parameter
        super().__init__(**kwargs)


    def get_utility(self, allocations, payments, valuations = None):
        """For reverse bidders, returns are inverted.
        """
        return - super().get_utility(allocations, payments, valuations)


class CombinatorialBidder(Bidder):
    """Bidder in combinatrorial auctions.

    Note: Currently only set up for full LLG setting.

    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if hasattr(self.strategy, 'input_length'):  # `ClosureStrategy` doesn't have `input_length`
            self.input_length = self.strategy.input_length
            self.output_length = self.strategy.output_length
        else:
            self.input_length = self.valuation_size
            self.output_length = self.bid_size

    def get_welfare(self, allocations, valuations: torch.Tensor=None) -> torch.Tensor:
        assert allocations.dim() >= 2  # *batch_sizes x items
        if valuations is None:
            valuations = self._cached_valuations

        item_dimension = valuations.dim() - 1
        # 0: item A | 1: item B | 2: bundle {A, B}
        # `player_position` == index of valued item for this agent
        if self.player_position != 2:  # locals also value bundle
            allocations_reduced_dim = allocations[..., [self.player_position, 2]] \
                .sum(axis=item_dimension, keepdim=True)
        else:  # global only values bundle
            allocations_reduced_dim = torch.logical_or(
                # won bundle of both
                allocations[..., [2]] == 1,
                # won both separately
                allocations[..., [0, 1]].sum(axis=item_dimension, keepdim=True) > 1
            )

        welfare = (valuations * allocations_reduced_dim).sum(dim=item_dimension)
        return welfare
