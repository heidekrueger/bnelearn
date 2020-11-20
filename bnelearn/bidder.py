# -*- coding: utf-8 -*-
"""Bidder module

This module implements players / bidders / agents in games.

"""

from abc import ABC, abstractmethod
import warnings
import math
import torch
from torch.distributions import Distribution
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
    """ A player in an auction game. Has a distribution over valuations/types that is
        common knowledge. These valuations correspond to the ´n_items´ available.

        ´batch_size´ corresponds to the number of individual auctions.
        If ´descending_valuations´ is true, the valuations will be returned
        in decreasing order.
        `cache_actions` determines whether actions should be cached and retrieved from memory,
            rather than recomputed as long as valuations haven't changed.

        # TODO Nils: clearly distinguish observation and type! (Nedded for correlation, splt-award, etc.)
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
                 constant_marginal_values = False,
                 correlation_type = None,
                 ):

        super().__init__(strategy, player_position, batch_size, cuda)

        self.value_distribution = value_distribution
        self.n_items = n_items
        self.descending_valuations = descending_valuations
        self.item_interest_limit = item_interest_limit
        self.constant_marginal_values = constant_marginal_values
        self.correlation_type = correlation_type
        self.risk = risk
        self._cache_actions = cache_actions
        self._valuations_changed = False # true if new valuation drawn since actions calculated
        self._valuations = torch.zeros(batch_size, n_items, device=self.device)
        if self._cache_actions:
            self.actions = torch.zeros(batch_size, n_items, device=self.device)
        self.draw_valuations_()

        # Compute lower and upper bounds for grid computation
        self._grid_lb = self.value_distribution.support.lower_bound \
            if hasattr(self.value_distribution.support, 'lower_bound') \
            else self.value_distribution.icdf(torch.tensor(0.001))
        self._grid_lb = max(0, self._grid_lb)
        self._grid_ub = self.value_distribution.support.upper_bound \
            if hasattr(self.value_distribution.support, 'upper_bound') \
            else self.value_distribution.icdf(torch.tensor(0.999))

    ### Alternative Constructors #############
    @classmethod
    def uniform(cls, lower: float, upper: float, strategy: Strategy, **kwargs):
        """Constructs a bidder with uniform valuation prior."""
        dist = torch.distributions.uniform.Uniform(low=lower, high=upper)
        return cls(value_distribution=dist, strategy=strategy, **kwargs)

    @classmethod
    def normal(cls, mean: float, stddev: float, strategy: Strategy, **kwargs):
        """Constructs a bidder with Gaussian valuation prior."""
        dist = torch.distributions.normal.Normal(loc = mean, scale = stddev)
        return cls(dist, strategy, **kwargs)

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
            self._valuations_changed = True

    def get_valuation_grid(self, n_points=None, extended_valuation_grid=False,
                           dtype=torch.float32, step=None, dimension=None):
        """ Returns a grid of approximately `n_points` valuations that are
            equidistant (in each dimension) on the support of self.value_distribution.
            If the support is unbounded, the 0.1th and 99.9th percentiles are used instead.

            For most distributions, the actual total size of the grid returned will be min(n^self.n_items)
            s.t. n^self.n_items >= n_points. E.g. for n_items=2 the grid will be square, for 3 it will be a cupe, etc.

            (For descending_valuations, the logic is somewhat different)

            Args:
                n_points: int, minimum number of total points in the grid
                extended_valuation_grid: bool, switch for bounds of interval
                step: float, step length. Only used when `n_points` is None
                dimension: int (otional), provide if `n_items`,
                    `strategy.input_lenght`, and `strategy.output_lenght`
                    are not all equal.
            returns:
                grid_values (dim: [ceil(n_points^(1/n_items)]*n_items)

            # TODO: - update this tu support different number of points per dimension
                    - with descending_valuations, this currently draws many more points than needed
                      then throws most of them away
        """

        assert n_points is None or step is None, \
            'Use only one of `n_points` or `step`'

        if extended_valuation_grid and hasattr(self, '_grid_lb_util_loss'):
            # pylint: disable=no-member
            lb = self._grid_lb_util_loss
            ub = self._grid_ub_util_loss
        else:
            lb = self._grid_lb
            ub = self._grid_ub

        if dimension is None:
            dimension = self.n_items

        if n_points is None:
            batch_size_per_dim = math.ceil((ub - lb) / step + 1)
        else:
            # change batch_size s.t. it'll approx. end up at intended n_points in the end
            adapted_size = n_points
            if self.descending_valuations:
                adapted_size = n_points * math.factorial(dimension)

            batch_size_per_dim = math.ceil(adapted_size ** (1/dimension))

        lin = torch.linspace(lb, ub, batch_size_per_dim, device=self.device, dtype=dtype)

        grid_values = torch.stack([
            x.flatten() for x in torch.meshgrid([lin] * dimension)]).t()

        if isinstance(self.item_interest_limit, int):
            grid_values[:,self.item_interest_limit:] = 0
        if self.constant_marginal_values:
            grid_values.index_copy_(1, torch.arange(1, dimension, device=self.device),
                                    grid_values[:,0:1].repeat(1, dimension-1))
        if self.descending_valuations:
            grid_values = grid_values.sort(dim=1, descending=True)[0].unique(dim=0)

        # assert grid_values.shape[0] >= n_points, "grid_size is lower than expected!"
        return grid_values

    def draw_valuations_(self, common_component = None, weights: torch.Tensor or float = 0.0):
        """ Sample a new batch of valuations from the Bidder's prior. Negative
            draws will be clipped at 0.0!

            When correlation info is given, valuations are drawn according to a mixture of the
            individually drawn component and the provided common component according to the provided weights.

            If ´descending_valuations´ is true, the valuations will be returned
            in decreasing order.

            Args:
                common_component (optional): torch.tensor (batch_size x n_items)
                    Tensor of (hidden) common component, same dimension as self.valuation.
                weights: (float, [0,1]) or tensor (batch_size x n_items) with values in [0,1]
                    defines how much to weigh the common component. If float, weighs entire tensor. If tensor
                    weighs component-wise.

            # TODO Stefan: Does correlation interere with Nils' implementations of descending valuations
            #              Or Item interest limits? --> Test!
        """
        if isinstance(weights, float):
            weights = torch.tensor(weights)

        assert weights.shape in {torch.Size([]), torch.Size([self.batch_size, self.n_items])}, \
            "Weights have invalid shape!"

        # Note: do NOT force-move weights and common_component to self.device until required!

        ### 1. For perfect correlation, no need to calculate individual component
        if torch.all(weights == 1.0):
            self.valuations = common_component.to(self.device).relu()
            return self.valuations

        ### 2. Otherwise determine individual component

        # If in place sampling is available for our distribution, use it!
        # This will save time for memory allocation and/or copying between devices
        # As sampling from general torch.distribution is only available on CPU.
        # (might mean adding more boilerplate code here if specific distributions are desired

        # uniform
        if isinstance(self.value_distribution, torch.distributions.uniform.Uniform):
            self.valuations.uniform_(self.value_distribution.low, self.value_distribution.high)
        # Gaussian
        elif isinstance(self.value_distribution, torch.distributions.normal.Normal):
            self.valuations.normal_(mean = self.value_distribution.loc, std = self.value_distribution.scale)
        else:
            # This is slow! (sampling on cpu then copying to GPU)
            # add additional internal in-place samplers above as needed!
            self.valuations = self.value_distribution.rsample(self.valuations.size()).to(self.device)

        ### 3. Determine mixture of individual and common component
        if torch.any(weights > 0):
            if self.correlation_type == 'additive':
                weights = weights.to(self.device)
                self.valuations = weights * common_component.to(self.device) + (1-weights) * self.valuations
            elif self.correlation_type == 'multiplicative':
                self.valuations = 2 * common_component.to(self.device) * self.valuations
                self._unkown_valuation = common_component.to(self.device)
            elif self.correlation_type == 'affiliated':
                self.valuations = (
                    common_component.to(self.device)[:, self.player_position]
                    + common_component.to(self.device)[:, 2]
                ).view(self.batch_size, -1)
                self._unkown_valuation = 0.5 * \
                    (common_component.to(self.device) * torch.tensor([1, 1, 2], device=self.device)) \
                        .sum(axis=1, keepdim=True)
            else:
                raise NotImplementedError('correlation type unknown')

        ### 4. Finishing up
        self.valuations.relu_() #ensure nonnegativity for unbounded-support distributions

        if isinstance(self.item_interest_limit, int):
            self.valuations[:,self.item_interest_limit:] = 0

        if self.constant_marginal_values:
            self.valuations.index_copy_(1, torch.arange(1, self.n_items, device=self.device),
                                        self.valuations[:,0:1].repeat(1, self.n_items-1))

        if self.descending_valuations:
            # for uniform vals and 2 items <=> F1(v)=v**2, F2(v)=2v-v**2
            self.valuations, _ = self.valuations.sort(dim=1, descending=True)

        self._valuations_changed = True # torch in-place operations do not trigger check in setter-method!
        return self.valuations

    def get_utility(self, allocations, payments): #pylint: disable=arguments-differ
        """
        For a batch of allocations and payments return the player's utilities at
        current valuations.
        """
        if hasattr(self, '_unkown_valuation'):
            valuations = self._unkown_valuation # case: signal != valuation
        else:
            valuations = self.valuations

        return self.get_counterfactual_utility(allocations, payments, valuations)

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
        # for cases when n_items != input_length (e.g. Split-Award Auctions, combinatorial auctions with bid languages)
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
    def __init__(self, efficiency_parameter=None, **kwargs):
        self.efficiency_parameter = efficiency_parameter
        super().__init__(**kwargs)
        self._grid_lb_util_loss = 0
        self._grid_ub_util_loss = float(2 * self._grid_ub)

    def get_valuation_grid(self, n_points, extended_valuation_grid=False):
        """ Extends `Bidder.draw_values_grid` with efficiency parameter

        Args
        ----
            extended_valuation_grid: bool, if True returns legitimate valuations, otherwise it returns
                a larger grid, which can be used as ``all reasonable bids`` as needed for
                estiamtion of regret.
        """

        grid_values = torch.zeros(n_points, self.n_items, device=self.device)

        if extended_valuation_grid:
            grid_values = super().get_valuation_grid(n_points, extended_valuation_grid)
        else:
            grid_values[:, 0] = torch.linspace(self._grid_lb, self._grid_ub, n_points,
                                               device=self.device)
            grid_values[:, 1] = self.efficiency_parameter * grid_values[:, 0]

        return grid_values

    def draw_valuations_(self, common_component = None, weights: torch.Tensor or float = 0.0):
        """ Extends `Bidder.draw_valuations_` with efiiciency parameter
        """
        _ = super().draw_valuations_(common_component, weights)

        assert self.valuations.shape[1] == 2, \
            'linear valuations are only defined for two items.'
        self.valuations[:, 1] = self.efficiency_parameter * self.valuations[:, 0]

        return self.valuations

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
        self.input_length = self.strategy.input_length
        self.output_length = self.strategy.output_length

    def get_valuation_grid(self, **kwargs):
        return super().get_valuation_grid(
            dimension=self.output_length,
            **kwargs
        )

    def get_welfare(self, allocations, valuations=None):
        assert allocations.dim() == 2 # batch_size x items
        if valuations is None:
            valuations = self.valuations

        item_dimension = valuations.dim() - 1
        # 0: item A | 1: item B | 2: bundle {A, B}
        welfare = (valuations * allocations[:, self.player_position].view(-1, 1)) \
            .sum(dim=item_dimension)

        return welfare
