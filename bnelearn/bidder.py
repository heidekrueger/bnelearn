# -*- coding: utf-8 -*-
"""Bidder module

This module implements players / bidders / agents in games.

"""

from abc import ABC, abstractmethod
import warnings
import math
import torch
from bnelearn.strategy import Strategy, MatrixGameStrategy, FictitiousPlayStrategy, FictitiousNeuralPlayStrategy


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
        raise NotImplementedError

    # def prepare_iteration(self):
    #     """ Prepares one iteration of environment-observation."""
    #     pass #pylint: disable=unnecessary-pass

    @abstractmethod
    def get_utility(self, **kwargs):
        """Calculates player's utility based on outcome of a game."""
        raise NotImplementedError

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
        cache_actions: determines whether actions should be cached and
            retrieved from memory, rather than recomputed as long as valuations
            haven't changed.
        TODO ...

    """
    # TODO Nils: clearly distinguish observation and type! (Nedded for
    # correlation, splt-award, etc.)
    def __init__(self,
                 strategy: Strategy,
                 player_position: torch.Tensor = None,
                 batch_size: int = 1,
                 valuation_size: int = 1,
                 observation_size: int =1,
                 bid_size: int = 1,
                 cuda: str = True,
                 cache_actions: bool = False,
                 risk: float = 1.0,
                 regret: float = 0.0,
                 ):

        super().__init__(strategy, player_position, batch_size, cuda)

        self.valuation_size = valuation_size
        self.observation_size = observation_size
        self.bid_size = bid_size,
        
        self.risk = risk
        self._cache_actions = cache_actions
        self._valuations_changed = False # true if new valuation drawn since actions calculated
        self._valuations = torch.zeros(batch_size, valuation_size, device=self.device)
        self._observations_changed = False # true if new observations drawn since actions calculated
        self._observations = torch.zeros(batch_size, observation_size, device=self.device)
        if self._cache_actions:
            self.actions = torch.zeros(batch_size, bid_size, device=self.device)

        # TODO: what are these, should these really be parts of bidder state?
        self.welfare_reference = None
        self.payments_reference = None
        self.regret = regret


    @property
    def valuations(self):
        return self._valuations

    @property
    def observations(self):
        return self._observations

    @valuations.setter
    def valuations(self, new_value: torch.Tensor):
        """When manually setting valuations, make sure that the _valuations_changed flag is set correctly."""
        if new_value.shape != self._valuations.shape:
            warnings.warn("New valuations have different shape than specified in Bidder object!")
        if (new_value.dtype, new_value.device) != (self._valuations.dtype, self._valuations.device):
            warnings.warn(
                "New valuations have different dtype and/or device than bidder. Converting to {},{}".format(
                    self._valuations.device, self._valuations.dtype)
                )

        if not new_value.equal(self._valuations):
            self._valuations = new_value.to(self._valuations.device, self._valuations.dtype)
            self._valuations_changed = True

    @observations.setter
    def observations(self, new_value: torch.Tensor):
        """When manually setting observations, make sure that the _observations_changed flag is set correctly."""
        if new_value.shape != self._observations.shape:
            warnings.warn("New observations have different shape than specified in Bidder object!")
        if (new_value.dtype, new_value.device) != (self._observations.dtype, self._observations.device):
            warnings.warn(
                "New observations have different dtype and/or device than bidder. Converting to {},{}".format(
                    self._observations.device, self._observations.dtype)
                )

        if not new_value.equal(self._observations):
            self._observations = new_value.to(self._observations.device, self._observations.dtype)
            self._observations_changed = True 



    def get_utility(self, allocations, payments, bids): #pylint: disable=arguments-differ
        """
        For a batch of allocations and payments return the player's utilities at
        current valuations.
        """
        valuations = self.valuations

        if self.regret > 0:
            welfare = self.get_welfare(allocations[:, self.player_position, :], valuations)
            payoff = welfare - payments[:, self.player_position]

            # TODO: Only implemented for LLG
            if self.valuation_size > 1:
                raise NotImplementedError('Regret not implemented for this setting.')

            alpha = beta = self.regret
            win_mask = torch.any(allocations[:, self.player_position, :].bool(), axis=1)
            if self.player_position < bids.shape[1] - 1:  # local
                highest_opponent_bids = bids[:, -1, 0]

                # winner's regret
                #   how much have the locals overbidden the global
                #   -> distribute that regret among them according to their relative
                #   payment attribution
                #   should be zero in core-selecting payment rule
                total_payments = payments[:, :-1].sum(axis=1)[win_mask]
                relative_payment = (
                    payments[:, self.player_position][win_mask] / total_payments
                )
                relative_payment[relative_payment != relative_payment] = 0  # set nan to 0

                payoff[win_mask] -= alpha * (
                    total_payments - highest_opponent_bids[win_mask]
                ).relu() * relative_payment

                # loser's regret
                #   when the sum of the bids of the opponents of my coalition and
                #   my own valuation exceeds the global bid, the local has that
                #   difference as regret

                # option a: use own value and other local's bid
                coalition_bids = bids[~win_mask, :-1, 0].sum(axis=1) - bids[~win_mask, self.player_position, 0]

                # option b: use all local's valuations and distribute attribution
                # coaltion_value = torch.zeros_like(highest_opponent_bids[~win_mask])
                # for a in env.agents[:-1]:
                #     coaltion_value += a.valuations[~win_mask, 0]
                # relative_value = (
                #     valuations[~win_mask, 0] / coaltion_value
                # )
                # relative_value[relative_value != relative_value] = 0  # set nan to 0

                payoff[~win_mask] -= beta * (
                    valuations[~win_mask, 0] + coalition_bids - highest_opponent_bids[~win_mask]
                ).relu()

            else:  # global
                highest_opponent_bids = bids[:, :-1, 0].sum(axis=1)

                # winner's regret
                payoff[win_mask] -= alpha * (
                    payments[win_mask, -1] - highest_opponent_bids[win_mask]
                ).relu()

                # loser's regret
                payoff[~win_mask] -= beta * (
                    valuations[~win_mask, 0] - highest_opponent_bids[~win_mask]
                ).relu()

            return payoff

        return self.get_counterfactual_utility(
            allocations[:, self.player_position, :],
            payments[:, self.player_position],
            valuations
        )

    def get_counterfactual_utility(self, allocations, payments, counterfactual_valuations):
        """
        For a batch of allocations, payments and counterfactual valuations return the
        player's utilities.

        Can handle multiple batch dimensions, e.g. for allocations a shape of
        (..., batch_size, n_items). These batch dimensions are kept in returned
        payoff.
        """
        welfare = self.get_welfare(allocations, counterfactual_valuations)
        payoff = welfare - payments

        if self.risk == 1.0:
            return payoff
        else:
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
        assert allocations.dim() == 2 # batch_size x items
        if valuations is None:
            valuations = self.valuations

        item_dimension = valuations.dim() - 1
        welfare = (valuations * allocations).sum(dim=item_dimension)

        return welfare

    def get_action(self):
        """Calculate action from current valuations, or retrieve from cache"""
        if self._cache_actions and not self._valuations_changed:
            return self.actions
        inputs = self.valuations.view(self.batch_size, -1)
        # for cases when n_observations != input_length (e.g. Split-Award Auctions, combinatorial auctions with bid languages)
        # TODO: generalize this, see #82. https://gitlab.lrz.de/heidekrueger/bnelearn/issues/82
        if hasattr(self.strategy, 'input_length') and self.strategy.input_length != self.observation_size:
            warnings.warn("Strategy expects shorter input_length than n_items. Truncating valuations...")
            dim = self.strategy.input_length
            inputs = inputs[:,:dim]

        actions = self.strategy.play(inputs)
        self._valuations_changed = False

        if self._cache_actions:
            self.actions = actions

        return actions


class ReverseBidder(Bidder):
    """
    Bidder that has reversed utility (*(-1)) as valuations correspond to
    their costs and payments to what they get payed.
    """
    def __init__(self, efficiency_parameter=None, **kwargs):
        self.efficiency_parameter = efficiency_parameter
        super().__init__(**kwargs)

    # pylint: disable=arguments-differ
    # def get_valuation_grid(self, n_points, extended_valuation_grid=False):
    #     """ Extends `Bidder.draw_values_grid` with efficiency parameter

    #     Args
    #     ----
    #         extended_valuation_grid: bool, if True returns legitimate valuations, otherwise it returns
    #             a larger grid, which can be used as ``all reasonable bids`` as needed for
    #             estiamtion of regret.
    #     """

    #     grid_values = torch.zeros(n_points, self.valuation_size, device=self.device)

    #     if extended_valuation_grid:
    #         grid_values = super().get_valuation_grid(n_points, extended_valuation_grid)
    #     else:
    #         grid_values[:, 0] = torch.linspace(self._grid_lb, self._grid_ub, n_points,
    #                                            device=self.device)
    #         grid_values[:, 1] = self.efficiency_parameter * grid_values[:, 0]

    #     return grid_values

    ### TODO: what's efficiency parameter? what do we do with this?
    # def draw_valuations_(self, common_component = None, weights: torch.Tensor or float = 0.0):
    #     """ Extends `Bidder.draw_valuations_` with efiiciency parameter
    #     """
    #     _ = super().draw_valuations_(common_component, weights)

    #     assert self.valuations.shape[1] == 2, \
    #         'linear valuations are only defined for two items.'
    #     self.valuations[:, 1] = self.efficiency_parameter * self.valuations[:, 0]

    #     return self.valuations

    def get_counterfactual_utility(self, allocations, payments, counterfactual_valuations):
        """For reverse bidders, returns are inverted.
        """
        return - super().get_counterfactual_utility(allocations, payments, counterfactual_valuations)


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

    # def get_valuation_grid(self, **kwargs):  # pylint: disable=arguments-differ
    #     return super().get_valuation_grid(
    #         dimension=self.output_length,
    #         **kwargs
    #     )

    def get_welfare(self, allocations, valuations: torch.Tensor=None) -> torch.Tensor:
        assert allocations.dim() == 2  # batch_size x items
        if valuations is None:
            valuations = self.valuations

        item_dimension = valuations.dim() - 1
        # 0: item A | 1: item B | 2: bundle {A, B}
        # `player_position` == index of valued item for this agent
        if self.player_position != 2:  # locals also value bundle
            allocations = allocations[:, [self.player_position, 2]] \
                .sum(axis=item_dimension) \
                .view(-1, 1)
        else:  # global only values bundle
            allocations = torch.logical_or(
                # won bundle of both
                allocations[:, 2] == 1,
                # won both separately
                allocations[:, [0, 1]].sum(axis=item_dimension) > 1
            ).view(-1, 1)

        welfare = (valuations * allocations).sum(dim=item_dimension)
        return welfare
