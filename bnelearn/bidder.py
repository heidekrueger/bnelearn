# -*- coding: utf-8 -*-
"""Bidder module

This module implements players / bidders / agents in games.

"""

from abc import ABC, abstractmethod
import warnings
import torch
from torch.distributions import Distribution
from bnelearn.strategy import MatrixGameStrategy, FictitiousPlayStrategy, FictitiousNeuralPlayStrategy

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

    def prepare_iteration(self):
        """ Prepares one iteration of environment-observation."""
        pass #pylint: disable=unnecessary-pass

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
    """ A player in an auction game. Has a distribution over valuations/types that is
        common knowledge. These valuations correspond to the ´n_items´ available.

        ´batch_size´ corresponds to the number of individual auctions.
        If ´descending_valuations´ is true, the valuations will be returned
        in decreasing order.
        `cache_actions` determines whether actions should be cached and retrieved from memory,
            rather than recomputed as long as valuations haven't changed.
    """
    def __init__(self,
                 value_distribution: Distribution,
                 strategy,
                 player_position = None,
                 batch_size = 1,
                 n_items = 1,
                 cuda = True,
                 cache_actions: bool = False,
                 descending_valuations = False,
                 risk: float = 1.0,
                 item_interest_limit = None,
                 constant_marginal_values = False
                 ):

        super().__init__(strategy, player_position, batch_size, cuda)

        self.value_distribution = value_distribution
        self.n_items = n_items
        self.descending_valuations = descending_valuations
        self.item_interest_limit = item_interest_limit
        self.constant_marginal_values = constant_marginal_values
        self.risk = risk
        self._cache_actions = cache_actions
        self._valuations_changed = False # true if new valuation drawn since actions calculated
        self._valuations = torch.zeros(batch_size, n_items, device=self.device)
        if self._cache_actions:
            self.actions = torch.zeros(batch_size, n_items, device=self.device)
        self.draw_valuations_()
        self.grid_lb = max(0,self.value_distribution.icdf(torch.tensor(0.001)))
        self.grid_ub = self.value_distribution.icdf(torch.tensor(0.999))

    ### Alternative Constructors #############
    @classmethod
    def uniform(cls, lower, upper, strategy, **kwargs):
        """Constructs a bidder with uniform valuation prior."""
        dist = torch.distributions.uniform.Uniform(low=lower, high=upper)
        return cls(dist, strategy, **kwargs)

    @classmethod
    def normal(cls, mean, stddev, strategy, **kwargs):
        """Constructs a bidder with Gaussian valuation prior."""
        dist = torch.distributions.normal.Normal(loc = mean, scale = stddev)
        return cls(dist, strategy, **kwargs)

    ### Members ####################
    def prepare_iteration(self):
        self.draw_valuations_()

    @property
    def valuations(self):
        return self._valuations

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
            self._valuations_changed =True

    def draw_valuations_new_batch_(self, batch_size):
        # Lets you sample valuations of a new batch size (e.g. for regret)
        # TODO: this will break action caching if it is being used! Consider redesigning this interface
        # TODO Nils: doesn't work for MultiUnit nor SplitAward
        self.valuations = self.value_distribution.rsample(torch.Size([batch_size, self.valuations.size()[1]])).to(self.device).relu()

    def draw_valuations_grid_(self, batch_size):
        """ Returns a batch of valuations equally distributed within the actual range
            and NOT accordiing to the actual destribution.

            Args
            ----
                batch_size: int, upper bound of returned batch size
        """
        batch_size_per_dim = int(batch_size ** (1/self.n_items))
        lin = torch.linspace(self.grid_lb, self.grid_ub,
                             batch_size_per_dim, device=self.device)
        grid_valuations = torch.stack([x.flatten() for x in torch.meshgrid([lin] * self.n_items)]).t()
        if isinstance(self.item_interest_limit, int):
            grid_valuations[:,self.item_interest_limit:] = 0
        if self.constant_marginal_values:
           grid_valuations.index_copy_(1, torch.arange(1, self.n_items, device=self.device),
                                       grid_valuations[:,0:1].repeat(1, self.n_items-1))
        if self.descending_valuations:
            grid_valuations = grid_valuations.sort(dim=1, descending=True)[0].unique(dim=0)

        return grid_valuations

    def draw_valuations_(self):
        """ Sample a new batch of valuations from the Bidder's prior. Negative
            draws will be clipped at 0.0!

            If ´descending_valuations´ is true, the valuations will be returned
            in decreasing order.
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

        if isinstance(self.item_interest_limit, int):
            self.valuations[:,self.item_interest_limit:] = 0

        if self.constant_marginal_values:
            self.valuations.index_copy_(1, torch.arange(1, self.n_items, device=self.device),
                                        self.valuations[:,0:1].repeat(1, self.n_items-1))

        if self.descending_valuations:
            # for uniform vals and 2 items <=> F1(v)=v**2, F2(v)=2v-v**2
            self.valuations, _ = self.valuations.sort(dim=1, descending=True)

        self._valuations_changed = True
        return self.valuations

    def get_utility(self, allocations, payments): #pylint: disable=arguments-differ
        """
        For a batch of allocations and payments return the player's utilities at
        current valuations.
        """
        return self.get_counterfactual_utility(allocations, payments, self.valuations)

    def get_counterfactual_utility(self, allocations, payments, counterfactual_valuations):
        """For a batch of allocations, payments and counterfactual valuations
            return the player's utilities
        """
        assert allocations.dim() == 2 # batch_size x items
        assert payments.dim() == 1 # batch_size

        payoff = self.get_welfare(allocations, counterfactual_valuations) - payments

        if self.risk == 1.0:
            return payoff
        else:
            # payoff^alpha not well defined in negative domain for risk averse agents
            return payoff.relu()**self.risk - (-payoff).relu()**self.risk

    def get_welfare(self, allocations, valuations = None):
        """
        For a batch of allocations and payments return the player's welfare.
        If valuations are not specified, welfare is calculated for `self.valuations`.
        """

        assert allocations.dim() == 2 # batch_size x items
        if valuations is None:
            valuations = self.valuations

        welfare = (valuations * allocations).sum(dim=1)

        return welfare

    def get_action(self):
        """Calculate action from current valuations, or retrieve from cache"""
        if self._cache_actions and not self._valuations_changed:
            return self.actions
        inputs = self.valuations.view(self.batch_size, -1)
        # for cases when n_itmes != input_length (e.g. Split-Award Auctions, combinatorial auctions with bid languages)
        # TODO: generalize this, see #82. https://gitlab.lrz.de/heidekrueger/bnelearn/issues/82
        if hasattr(self.strategy, 'input_length') and self.strategy.input_length != self.n_items:
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
    def __init__(self,
                 value_distribution: Distribution,
                 strategy,
                 player_position = None,
                 batch_size = 1,
                 n_units = 1,
                 cuda = True,
                 cache_actions: bool = False,
                 descending_valuations = False,
                 risk: float = 1.0,
                 item_interest_limit = None,
                 constant_marginal_values = False,
                 efficiency_parameter = None,
                 ):

        self.efficiency_parameter = efficiency_parameter
        super().__init__(
            value_distribution,
            strategy,
            player_position,
            batch_size,
            n_units,
            cuda,
            cache_actions,
            descending_valuations,
            risk,
            item_interest_limit,
            constant_marginal_values
        )
        self.grid_lb_regret = 0
        self.grid_ub_regret = float(2*self.grid_ub)

    @classmethod
    def uniform(cls, lower, upper, strategy, **kwargs):
        """Constructs a bidder with uniform valuation prior."""
        dist = torch.distributions.uniform.Uniform(low=lower, high=upper)
        return cls(dist, strategy, **kwargs)

    @classmethod
    def normal(cls, mean, stddev, strategy, **kwargs):
        """Constructs a bidder with Gaussian valuation prior."""
        dist = torch.distributions.normal.Normal(loc = mean, scale = stddev)
        return cls(dist, strategy, **kwargs)

    def draw_valuations_grid_(self, batch_size):
        """ Extends `Bidder.draw_valuations_grid_` with efiiciency parameter
        """
        grid_valuations = super().draw_valuations_grid_(batch_size)
        grid_valuations[:,1] = self.efficiency_parameter * grid_valuations[:,0]

        return grid_valuations

    def draw_valuations_(self):
        """ Extends `Bidder.draw_valuations_` with efiiciency parameter
        """
        _ = super().draw_valuations_()

        assert self.valuations.shape[1] == 2, \
            'linear valuations are only defined for two items.'
        self.valuations[:,1] = self.efficiency_parameter * self.valuations[:,0]

        return self.valuations

    def get_counterfactual_utility(self, allocations, payments, counterfactual_valuations):
        """For reverse bidders, returns are inverted.
        """
        return - super().get_counterfactual_utility(allocations,payments,counterfactual_valuations)
