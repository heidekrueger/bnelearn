"""This module implements samplers for independent-private value auction settings."""

from typing import List, Tuple
from math import ceil, log
from functools import reduce

import torch
from torch.cuda import _device_t as Device
from torch.distributions import Distribution
from .base import PVSampler
from bnelearn.util.tensor_util import item2bundle

class FixedManualIPVSampler(PVSampler):
    """For testing purposes:
    A sampler that returns a fixed tensor as valuations/observations.
    """
    def __init__(self, valuation_tensor: torch.Tensor):

        assert valuation_tensor.dim() == 3, "invalid input tensor"

        self._profile = valuation_tensor

        batch_size, n_players, valuation_size = valuation_tensor.shape
        device = valuation_tensor.device

        support_low = valuation_tensor.min(dim=0).values
        support_hi = valuation_tensor.max(dim=0).values

        support_bounds = torch.stack([support_low, support_hi], dim=-1)


        super.__init__(n_players, valuation_size, support_bounds,
                       batch_size, device)

        self.draw_conditional_profiles = SymmetricIPVSampler.draw_conditional_profiles

    def _sample(self, batch_sizes, device):
        return self._profile, self._profile


class SymmetricIPVSampler(PVSampler):
    """A Valuation Oracle that draws valuations independently and symmetrically
    for all players and each entry of their valuation vector according to a specified
    distribution.

    This base class works with all torch.distributions but requires sampling on
    cpu then moving to the device. When using cuda, use the faster,
    distribution-specific subclasses instead where provided.

    """

    UPPER_BOUND_QUARTILE_IF_UNBOUNDED = .999

    def __init__(self, distribution: Distribution,
                 n_players: int, valuation_size: int = 1,
                 default_batch_size: int = 1, default_device: Device = None
                ):
        """
        Args:
            distribution: a single-dimensional torch.distributions.Distribution.
            n_players: the number of players
            valuation_size: the length of each valuation vector
            default_batch_size: the default batch size for sampling from this instance
            default_device: the default device to draw valuations. If none given,
                uses 'cuda' if available, 'cpu' otherwise
        """
        self.base_distribution = distribution
        self.distribution = self.base_distribution.expand([n_players, valuation_size])

        # bounds: use real support, unless unbounded:
        support = self.base_distribution.support
        if isinstance(support, torch.distributions.constraints._Real):
            upper_bound = self.base_distribution.icdf(
                torch.tensor(self.UPPER_BOUND_QUARTILE_IF_UNBOUNDED))
            lower_bound = torch.tensor(0, device=upper_bound.device)
        else:
            lower_bound = torch.tensor(support.lower_bound).relu()
            upper_bound = torch.tensor(support.upper_bound)

        assert upper_bound >= lower_bound

        # repeat support bounds across all players and valuation dimensions
        support_bounds = torch.stack([lower_bound, upper_bound]).repeat([n_players, valuation_size, 1])

        super().__init__(n_players, valuation_size, support_bounds, default_batch_size, #pylint: disable=arguments-out-of-order
                         default_device)

    def _sample(self, batch_sizes: int or List[int], device: Device) -> torch.Tensor:
        """Draws a batch of observation/valuation profiles (equivalent in PV)"""
        batch_sizes = self._parse_batch_sizes_arg(batch_sizes)
        device = device or self.default_device

        return self.distribution.sample(batch_sizes).to(device)

    def draw_conditional_profiles(self,
                                  conditioned_player: int,
                                  conditioned_observation: torch.Tensor,
                                  inner_batch_size: int,
                                  device: Device = None) -> Tuple[torch.Tensor, torch.Tensor]:
        # Due to independence, we can simply draw a full profile and replace the
        # conditioned_observations by their specified value.
        # Due to PV, valuations are equal to the observations

        device = device or self.default_device
        inner_batch_size = inner_batch_size or self.default_batch_size
        *outer_batch_sizes, observation_size = conditioned_observation.shape

        profile = self._sample([*outer_batch_sizes, inner_batch_size], device)

        profile[..., conditioned_player, :] = \
            conditioned_observation \
                .view(*outer_batch_sizes, 1, observation_size) \
                .repeat(*([1]*len(outer_batch_sizes)), inner_batch_size, 1)

        return profile, profile


class UniformSymmetricIPVSampler(SymmetricIPVSampler):
    """An IPV sampler with symmetric Uniform priors."""
    def __init__(self, lo, hi,
                 n_players, valuation_size,
                 default_batch_size, default_device: Device = None):
        distribution = torch.distributions.uniform.Uniform(low=lo, high=hi)
        super().__init__(distribution,
                         n_players, valuation_size,
                         default_batch_size, default_device)

    def _sample(self, batch_sizes, device) -> torch.Tensor:
        batch_sizes = self._parse_batch_sizes_arg(batch_sizes)

        # create an empty tensor on the output device, then sample in-place
        return torch.empty(
            [*batch_sizes, self.n_players, self.valuation_size],
            device=device).uniform_(self.base_distribution.low, self.base_distribution.high)


class GaussianSymmetricIPVSampler(SymmetricIPVSampler):
    """An IPV sampler with symmetric Gaussian priors."""
    def __init__(self, mean, stddev,
                 n_players, valuation_size,
                 default_batch_size, default_device: Device = None):
        distribution = torch.distributions.normal.Normal(loc=mean, scale=stddev)
        super().__init__(distribution,
                         n_players, valuation_size,
                         default_batch_size, default_device)

    def _sample(self, batch_sizes, device) -> torch.Tensor:
        batch_sizes = self._parse_batch_sizes_arg(batch_sizes)
        # create empty tensor, sample in-place, clip
        return torch.empty([*batch_sizes, self.n_players, self.valuation_size],
                           device=device) \
            .normal_(self.base_distribution.loc, self.base_distribution.scale) \
            .relu_()

class BetaSymmetricIPVSampler(SymmetricIPVSampler):
    """An IPV sampler with symmetric Beta priors."""
    def __init__(self, alpha: float, beta: float,
                 n_players: int, valuation_size: int,
                 default_batch_size: int, default_device: Device = None):
        distribution = torch.distributions.Beta(concentration1=alpha,
                                                concentration0=beta)
        super().__init__(distribution,
                         n_players, valuation_size,
                         default_batch_size, default_device)

    def _sample(self, batch_sizes, device) -> torch.Tensor:
        batch_sizes = self._parse_batch_sizes_arg(batch_sizes)
        # create empty tensor, sample in-place, clip
        size = [self.n_players, self.valuation_size]
        return self.base_distribution.expand(size) \
            .rsample([*batch_sizes]).to(device)

class MultiUnitValuationObservationSampler(UniformSymmetricIPVSampler):
    """Sampler for Multi-Unit, private value settings.
    Sampler for valuations and signals in Multi-Unit Auctions, following
    Krishna, Chapter 13.

    These are symmetric private value settings, where
    (a) valuations are descending along the valuation_size dimension and
        represent the marginal utility of winning an additional item.
    (b) bidders may be limited to be interested in at most a certain number of
        items.

    """

    def __init__(self, n_players: int, n_items: int = 1,
                 max_demand: int = None, constant_marginal_values: bool = False,
                 u_lo: float = 0.0, u_hi: float = 1.0,
                 default_batch_size: int = 1, default_device = None):
        """
        Args:
            n_players
            n_items: the number of items
            max_demand: the maximal number of items a bidder is interested in winning.
            constant_marginal_values: whether or not all values should be
                constant (i.e. to enforce additive valuations on homogenous goods.)
            u_lo: lower bound for uniform distribution
            u_hi: upper bound for uniform distribtuion
            default_batch_size
            default_device
        """

        # if no demand limit is given, assume it is the total number of items
        self.max_demand: int = max_demand or n_items
        assert isinstance(self.max_demand, int), "maximum demand must be integer or none."
        assert self.max_demand > 0, "invalid max demand"
        assert self.max_demand <= n_items, "invalid max demand"

        assert u_lo >= 0, "valuations must be nonnegative"
        assert u_hi > u_lo, "upper bound must larger than lower bound"

        self._constant_marginal_values = constant_marginal_values
        self._u_lo = u_lo
        self._u_hi = u_hi

        super().__init__(u_lo, u_hi,
                         n_players, n_items,
                         default_batch_size=default_batch_size,
                         default_device=default_device)

        # define alias AFTER super init such that both point to same memory address
        self.n_items = self.valuation_size

    def _sample(self, batch_sizes, device) -> torch.Tensor:
        """Draws a batch of uniform valuations, sorts them,
        and masks them out if necessary"""
        batch_sizes = self._parse_batch_sizes_arg(batch_sizes)
        # profile is batch x player x items
        profile =  super()._sample(batch_sizes, device)
        # sort by item
        profile, _ = profile.sort(dim=-1, descending=True)
        # valuations beyond the limit are 0
        profile[..., self.max_demand:self.n_items] = 0.0
        # valuations are constant
        if self._constant_marginal_values:
            profile[..., :, 1:] = profile[..., :, [0]] \
                .repeat(*[1]*(len(profile.shape) - 1), profile.shape[-1] - 1)
        return profile

    def generate_valuation_grid(self, player_position: int, minimum_number_of_points: int,
                                dtype=torch.float, device = None,
                                support_bounds: torch.Tensor = None) -> torch.Tensor:
        rectangular_grid = super().generate_valuation_grid(
            player_position, minimum_number_of_points, dtype, device, support_bounds)

        # transform to triangular grid (valuations are marginally descending)
        return rectangular_grid.sort(dim=1, descending=True)[0].unique(dim=0)


class SplitAwardtValuationObservationSampler(UniformSymmetricIPVSampler):
    """Sampler for Split-Award, private value settings. Here bidders have two
    valuations of which one is a linear combination of the other.
    """
    def __init__(self, efficiency_parameter: float, **kwargs):
        super().__init__(n_players=2, **kwargs)
        self.efficiency_parameter = efficiency_parameter
        assert self.support_bounds[0, 0, 0] == self.support_bounds[1, 0, 0], \
            'Bounds not suppoted in this setting.'
        assert self.support_bounds[0, 0, 1] == self.support_bounds[1, 0, 1], \
            'Bounds not suppoted in this setting.'

    def _sample(self, batch_sizes, device) -> torch.Tensor:
        batch_sizes = self._parse_batch_sizes_arg(batch_sizes)

        # create an empty tensor on the output device, then sample in-place
        sample = torch.empty([*batch_sizes, self.n_players, 2], device=device) \
            .uniform_(self.base_distribution.low, self.base_distribution.high)
        sample[..., 0] = sample[..., 1] * self.efficiency_parameter
        return sample

    def generate_valuation_grid(self, player_position: int, minimum_number_of_points: int,
                                dtype=torch.float, device=None,
                                support_bounds: torch.Tensor = None) -> torch.Tensor:
        device = device or self.default_device

        if support_bounds is None:
            support_bounds = self.support_bounds

        bounds = support_bounds[player_position]

        # dimensionality
        dims = 2
        n_points_per_dim = ceil(minimum_number_of_points/dims)

        # create equidistant line along the support
        line = torch.linspace(bounds[0][0], bounds[0][1], n_points_per_dim,
                              device=device, dtype=dtype)

        grid = torch.stack((self.efficiency_parameter * line, line), dim=-1) \
            .view(-1, dims)

        return grid


class CombinatorialItemSampler(MultiUnitValuationObservationSampler):
    """Combinatorial Auction with Item Bidding Sampler. Supports uniform
    distributions only.
    """
    def __init__(self, n_players: int, n_items: int, valuation_type: str='XOS',
                 valuation_dict: dict={}, **kwargs):
        u_lo = 0
        u_hi = 1
        self.n_bundles = (2 ** n_items) - 1

        self.valuation_type = valuation_type
        self.transformation = item2bundle(n_items)

        if valuation_type == 'XOS':
            if 'n_collections' in valuation_dict.keys():
                self.n_collections = valuation_dict['n_collections']
            else:
                self.n_collections = 1  # fall back to additive valuations

            if 'one_player_with_unit_demand' in valuation_dict.keys():
                self.unit_demand = valuation_dict['one_player_with_unit_demand']
            else:
                self.unit_demand = False

        elif valuation_type == 'submodular':
            if 'submodular_factor' not in valuation_dict.keys():
                valuation_dict['submodular_factor'] = .5  # TODO: default value?
            self.submodular_factor = valuation_dict['submodular_factor']

        super().__init__(n_players=n_players, n_items=self.n_bundles,
                         u_lo=u_lo, u_hi=u_hi, **kwargs)

    def _sample(self, batch_sizes, device) -> torch.Tensor:
        """Sample a new batch of valuations from the bidder's prior."""
        batch_sizes = self._parse_batch_sizes_arg(batch_sizes)

        # TODO: Quick-fix of flatteing all batch dims for now
        batch_size = reduce(lambda x, y: x*y, batch_sizes)

        n_bundles = self.n_bundles
        n_items = int(log(n_bundles + 1, 2))
        transformation = self.transformation.to(device)

        sample = torch.empty([batch_size, self.n_players, n_bundles], device=device)

        for player_position in range(self.n_players):
        
            if self.valuation_type == 'XOS':
                """XOS valuations: There are `n_collections` additive valuation
                functions (e.g. bundle valuations are the sum of all individual
                items). And the final bundle valuation is then the max over all
                collections.
                """
                collection_valuations = torch.empty([batch_size, self.n_collections, n_bundles],
                                                    device=device)

                # uniform samples
                if isinstance(self.base_distribution, torch.distributions.uniform.Uniform):
                    collection_valuations.uniform_(self.base_distribution.low, self.base_distribution.high)
                    collection_valuations[:, :, n_items:] = 0
                else:
                    raise NotImplementedError('Unknown valuation distribution.')

                if self.unit_demand and player_position == 0:
                    # repeat each item valuation
                    vals = collection_valuations[:, :, :n_items] \
                        .repeat_interleave(n_bundles, 2) \
                        .view(batch_size, n_items, n_bundles)
                    # select the most valuable item in each bundle
                    sample[:, player_position, :] = \
                        torch.einsum('bij,ij->bij', vals, transformation[:n_items, :]) \
                            .max(dim=1)[0]

                else:
                    sample[:, player_position, :] = \
                        torch.matmul(collection_valuations, transformation) \
                            .max(dim=1)[0]

            elif self.valuation_type == 'submodular':
                """Submodularity is a form of diminishing returns: the more
                stuff on already has got, the less additional value one gets
                from new stuff.
                
                Idea: U[0, 1] valuations for single item bundles. For all other
                bundles, the lowest valued item is only adding a fraction
                (hyper-param) of its value to the final bundle value. Problem:
                Not representative for all submodular functions.
                """
                inf = 1e16
                valuations = torch.empty([batch_size, n_items], device=device) \
                    .uniform_(self.base_distribution.low, self.base_distribution.high) \
                    .repeat_interleave(n_bundles) \
                    .view(batch_size, n_items, n_bundles)

                valuations *= transformation[:n_items, :]

                valuations[valuations == 0] = inf  # have to ignore 0s as not part of bundle
                _, idx = valuations.min(dim=1, keepdim=True)
                valuations[valuations == inf] = 0
                all_lowest_values = torch.arange(n_items, device=device) \
                    .reshape(1, n_items, 1) == idx
                single_items = transformation.sum(dim=0) == 1
                lowest_values_in_bundles = torch.logical_and(all_lowest_values, single_items)
                valuations[lowest_values_in_bundles] *= self.submodular_factor

                sample[:, player_position, :] = valuations.sum(dim=1)

        return sample.view(*batch_sizes, self.n_players, n_bundles)

    def generate_valuation_grid(self, player_position: int, minimum_number_of_points: int,
                                dtype=torch.float, device = None,
                                support_bounds: torch.Tensor = None) -> torch.Tensor:
        """Generates an evenly spaced grid of (approximately and at least)
        minimum_number_of_points valuations covering the support of the
        valuation space for the given player. These are meant as rational actions
        for the player to evaluate, e.g. in the util_loss estimator.

        The default reference implementation returns a rectangular grid on
        [0, upper_bound] x valuation_size.
        """
        bundle_grid = super().generate_valuation_grid(
            player_position=player_position, minimum_number_of_points=minimum_number_of_points,
            dtype=dtype, device=device, support_bounds=support_bounds)

        return bundle_grid[:, :int(log(self.n_bundles + 1, 2))]
    
    def generate_action_grid(self, player_position: int, minimum_number_of_points: int,
                             dtype=torch.float, device = None) -> torch.Tensor:
        
        device = device or self.default_device
        n_items = int(log(self.n_bundles + 1, 2))

        bounds = self.support_bounds[player_position, :n_items]

        n_points_per_dim = ceil(minimum_number_of_points**(1/n_items))

        # create equidistant lines along the support in each dimension
        lines = [torch.linspace(bounds[d][0], bounds[d][1], n_points_per_dim,
                                device=device, dtype=dtype)
                 for d in range(n_items)]
        grid = torch.stack(torch.meshgrid(lines), dim=-1).view(-1, n_items)

        return grid

    def generate_reduced_grid(self, player_position: int, minimum_number_of_points: int,
                              dtype=torch.float, device = None) -> torch.Tensor:
        return super().generate_valuation_grid(
            player_position=player_position, minimum_number_of_points=minimum_number_of_points,
            dtype=dtype, device=device)