"""This class implements drawing of valuation and observation profiles."""

from abc import ABC, abstractmethod
from math import sqrt, ceil
from typing import List, Tuple
import warnings

import torch
from torch.cuda import _device_t as Device, device_of
from torch.distributions import Distribution
from torch.overrides import is_tensor_like


class ValuationObservationSampler(ABC):
    """Provides functionality to draw valuation and observation profiles."""

    

    def __init__(self, n_players, valuation_size, observation_size,
                 support_bounds,
                 default_batch_size = 1, default_device = None):
        self.n_players: int = n_players # The number of players in the valuation profile
        self.valuation_size: int = valuation_size # The dimensionality / length of a single valuation vector
        self.observation_size: int = observation_size # The dimensionality / length of a single observation vector
        self.default_batch_size: int = default_batch_size # a default batch size
        self.default_device: Device = (default_device or 'cuda') if torch.cuda.is_available() else 'cpu'
        
        assert support_bounds.size() == torch.Size([n_players, valuation_size, 2]), \
            "invalid support bounds."
        self.support_bounds: torch.FloatTensor = support_bounds.to(self.default_device)

    def _parse_batch_sizes_arg(self, batch_sizes_argument: int or List[int] or None) -> List[int]:
        """Parses an integer batch_size_argument into a list. If none given,
           defaults to the list containing the default_batch_size of the instance.
        """
        batch_sizes = batch_sizes_argument or self.default_batch_size
        if isinstance(batch_sizes, int):
            batch_sizes = [batch_sizes]
        return batch_sizes

    @abstractmethod
    def draw_profiles(self, batch_sizes: int or List[int] = None, device=None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Draws and returns a batch of valuation and observation profiles.

        Kwargs:
            batch_size (optional): int, the batch_size to draw. If none provided,
            `self.default_batch_size` will be used.
            device (optional): torch.cuda.Device, the device to draw profiles on

        Returns:
            valuations: torch.Tensor (batch_size x n_players x valuation_size): a valuation profile
            observations: torch.Tensor (batch_size x n_players x observation_size): an observation profile
        """

    @abstractmethod
    def draw_conditional_profiles(self,
                                  conditioned_player: int,
                                  conditioned_observation: torch.Tensor,
                                  inner_batch_size: int, device: Device = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Draws and returns batches conditional valuation and corresponding observation profile.
        For each entry of `conditioned_observation`, `batch_size` samples will be drawn!

        Note that here, we are returning full profiles instead (including
        `conditioned_player`'s observation and others' valuations.)

        Args:
            conditioned_player: int
                Index of the player whose observation we are conditioning on.
            conditioned_observation: torch.Tensor (`outer_batch_size` (implicit), `observation_size`)
                A (batch of) observations of player `conditioned_player`.

        Kwargs:
            batch_size (optional): int, the "inner"batch_size to draw - i.e.
            how many conditional samples to draw for each provided `conditional_observation`.
            If none provided, will use `self.default_batch_size` of the class.

        Returns:
            valuations: torch.Tensor (outer_batch_size x inner_batch_size x n_players x valuation_size):
                a conditional valuation profile
            observations: torch.Tensor (`outer_batch_size`, inner_batch_size x n_players x observation_size):
                a corresponding conditional observation profile.
                observations[:,conditioned_observation,:] will be equal to
                `conditioned_observation` repeated `batch_size` times
        """
        pass

    def generate_valuation_grid(self, player_position: int, n_grid_points: int,
                                dtype=torch.float, device = None) -> torch.Tensor:
        """Generates an evenly spaced grid of (approximately) n_grid_points
        valuations covering the support of the valuation space for the given
        player. These are meant as rational actions for the player to evaluate,
        e.g. in the util_loss estimator.

        The default reference implementation returns a rectangular grid on
        [0, upper_bound] x valuation_size.

        TODO: There are settings (e.g. ReverseBidder), where it may be rational
        to overbid, this still needs to be implemented.
        """

        device = device or self.default_device

        bounds = self.support_bounds[player_position]

        # dimensionality
        dims = self.valuation_size

        # use equal density in each dimension of the valuation, such that
        # the total number of points is at least as high as the specified one
        n_points_per_dim = ceil(n_grid_points**(1/dims))

        # create equidistant lines along the support in each dimension
        lines = [torch.linspace(bounds[d][0], bounds[d][1], n_points_per_dim,
                                device=device, dtype=dtype)
                 for d in range(dims)]
        grid = torch.stack(torch.meshgrid(lines), dim=-1).view(-1, dims)

        return grid

class PVSampler(ValuationObservationSampler, ABC):
    """A sampler for Private Value settings, i.e. when observations and
     valuations are identical
    """

    def __init__(self, n_players: int, valuation_size: int, support_bounds,
                 default_batch_size: int = 1, default_device: Device = None):
        super().__init__(n_players, valuation_size, valuation_size, support_bounds,
                         default_batch_size, default_device)

    @abstractmethod
    def _sample(self, batch_sizes: int or List[int], device: Device) -> torch.Tensor:
        """Returns a batch of profiles (which are both valuations and observations)"""

    def draw_profiles(self, batch_sizes: int or List[int] = None,
                      device: Device = None) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_sizes = self._parse_batch_sizes_arg(batch_sizes)
        device = device or self.default_device
        # In the PV setting, valuations and observations are identical.
        profile = self._sample(batch_sizes, device)
        return profile, profile


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


#TODO: change name to express that this is symmetric also along items, not just players?
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
        lower_bound = self.base_distribution.icdf(torch.tensor(0.)).relu()
        upper_bound = self.base_distribution.icdf(torch.tensor(1.))
        if upper_bound.isinf().item():
            upper_bound = self.base_distribution.icdf(
                torch.tensor(self.UPPER_BOUND_QUARTILE_IF_UNBOUNDED))
        
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
        *outer_batch_sizes , observation_size = conditioned_observation.shape

        profile = self._sample([*outer_batch_sizes, inner_batch_size], device)

        profile[...,conditioned_player, :] = \
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


class CorrelatedSymmetricUniformPVSampler(PVSampler, ABC):
    r"""
    Draws (non-independent) symmetric uniform valuation profiles that are
    pairwise correlated between the bidders according to the
    models from Ausubel & Baranov (2019) (https://doi.org/10.1007/s00182-019-00691-3).

    Use subclasses `BernoulliWeightsCorrelatedSymmetricUniformPVSampler` or
    `ConstantWeightsCorrelatedSymmetricUniformPVSampler` to initialize.

    Individual valuations v_i are composed additively of an individual
    component :math:`z_i` and a common component :math:`s`,
    where :math:`z_i, s` are each uniform.

    .. math::
        v_i = (1-w) z_i + w s

    weights are set according to either the 'Bernoulli Weights Model' or
    'Constant Weights Model'.

    In this scheme, a Bernoulli (0 or 1) weight determines that either
    :math:`v_i = z_i` or :math:`v_i = s`, with weights/probabilities being set
    such that correlation :math:`\gamma` is achieved between bidders.
    """

    def __init__(self, u_lo: float or torch.Tensor, u_hi: float or torch.Tensor,
                 n_players: int, valuation_size: int, correlation: float,
                 weight_method,
                 default_batch_size = 1, default_device: Device = None):
        r"""
        Args:
            u_lo, u_hi: lower and upper bounds of the distribution
            n_players: the number of players in the profile
            valuation_size (int): length of observation vector,
            correlation (float): correlation strength :math:`0\leq\gamma \leq 1`
            weight_method (str): the type of correlation model,
                one of 'Bernoulli' or 'constant'
            default_batch_size: the default batch size for sampling from this instance
            default_device: the default device to draw valuations. If none given,
                uses 'cuda' if available, 'cpu' otherwise
        """

        assert u_lo >= 0, "Negative valuations currently not supported!"
        self.u_lo = u_lo
        self.u_hi = u_hi
        self.gamma = correlation
        if weight_method in ['Bernoulli', 'constant']:
            self.method = weight_method
        else:
            raise ValueError('Unknown method, must be one of "Bernoulli", "constant"')
        
        support_bounds = torch.tensor([u_lo, u_hi]).repeat([n_players, valuation_size, 1])

        super().__init__(n_players, valuation_size, support_bounds,
                         default_batch_size, default_device)

    @abstractmethod
    def _get_weights(self, batch_sizes: List[int], device: Device) -> torch.Tensor:
        """Returns a batch of weights according to the model
           or a constant weight for use in the entire batch.
        """

    def _sample(self, batch_sizes: List[int], device: Device) -> torch.Tensor:
        """Draws a batch of observation/valuation profiles (equivalent in PV)"""
        batch_sizes = self._parse_batch_sizes_arg(batch_sizes)
        device = device or self.default_device

        individual_components = torch.empty(
            [*batch_sizes, self.n_players, self.valuation_size],
            device = device) \
            .uniform_(self.u_lo, self.u_hi)

        common_component = torch.empty(
            [*batch_sizes, 1, self.valuation_size],
            device = device) \
            .uniform_(self.u_lo, self.u_hi)

        w = self._get_weights(batch_sizes, device)

        return (1-w) * individual_components + w * common_component

class BernoulliWeightsCorrelatedSymmetricUniformPVSampler(CorrelatedSymmetricUniformPVSampler):

    def __init__(self, n_players: int, valuation_size: int, correlation: float,
                 u_lo: float or torch.Tensor = 0.0, u_hi: float or torch.Tensor = 1.0,
                 default_batch_size = 1, default_device: Device = None):
        super().__init__(u_lo, u_hi, n_players, valuation_size, correlation,
                         'Bernoulli', default_batch_size, default_device)

    def _get_weights(self, batch_sizes: List[int], device: Device) -> torch.Tensor:
        """Draws Bernoulli distributed weights along the batch size.

        Returns:
            w: Tensor of shape (batch_size, 1, 1)
        """
        # TODO: do we want to handle weights differently in different batch dimensions?
        return (torch.empty(batch_sizes, device=device)
                .bernoulli_(self.gamma) # different weight per batch
                .view(-1, 1, 1)) # same weight across item/bundle in each batch-instance

    def draw_conditional_profiles(self, conditioned_player: int,
                                  conditioned_observation: torch.Tensor,
                                  inner_batch_size: int, device: Device  = None
                                  ) -> Tuple[torch.Tensor, torch.Tensor]:
        device = device or self.default_device
        outer_batch_size, observation_size = conditioned_observation.shape

        conditioned_observation = conditioned_observation.to(device)
        # let i be the index of the player conditioned on.
        i = conditioned_player

        # repeat each entry of conditioned_observation inner_batch_size times.
        v_i = conditioned_observation \
            .view(outer_batch_size, 1, observation_size) \
            .repeat(1,inner_batch_size, 1)


        # each observation v_i has a probability of self.gamma of being the
        # common component v_i=s, and (1-self.gamma) of being the player's
        # individual component v_i=z_i. In the former case, all players have
        # the same obs v_j=s=v_i. Otherwise, each has valuation v_j=z_j
        # equal to their individual component z_j.

        # Individual components z_j are conditionally independent of z_i.
        # Start by sampling these (and overwriting ith entry with actual obs.)
        # (ith's entry is technically incorrect, but the cases
        # where v_i != z_i are disregarded by the weights drawn below.)
        z = torch.empty([outer_batch_size,inner_batch_size,
                         self.n_players,
                         self.valuation_size], device = device) \
            .uniform_(self.u_lo, self.u_hi)
        z[...,i,:] = v_i

        # NOTE: with our current test (e.g. testing correlation matrix
        # of conditional valuation profile for large outer_batch and
        # inner_batch of 1), we cannot use the same drawn weights in each outer
        # batch, otherwise, we'll always end up perfectly correlated, or not
        # at all, rather than the correct amount.

        w = torch.empty([outer_batch_size, inner_batch_size, 1, 1], device=device) \
            .bernoulli_(self.gamma)

        # sample valuations directly:
        # either individual component of each player z_j,
        # or common component s=v_i, which we stored in the ith entry of the z tensor
        v = (1-w)*z + w * z[...,[i],:]

        # private values setting: observations = valuations
        return v, v


class ConstantWeightCorrelatedSymmetricUniformPVSampler(CorrelatedSymmetricUniformPVSampler):

    def __init__(self, n_players: int, valuation_size: int, correlation: float,
                 u_lo: float or torch.Tensor = 0.0, u_hi: float or torch.Tensor = 1.0,
                 default_batch_size = 1, default_device: Device = None):
        super().__init__(u_lo, u_hi, n_players, valuation_size, correlation,
                         'constant', default_batch_size, default_device)
        self._weight = 0.5 if correlation == 0.5 \
            else (correlation - sqrt(correlation*(1-correlation))) / \
                (2*correlation - 1)
        self._weight = torch.tensor(self._weight, device = self.default_device)


    def _get_weights(self, batch_sizes: List[int], device) -> torch.Tensor:
        """Draws Bernoulli distributed weights along the batch size.
        """
        # batch size is ignored, we always return a scalar.
        return self._weight.to(device)

    def draw_conditional_profiles(self,
                                  conditioned_player: int,
                                  conditioned_observation: torch.Tensor,
                                  inner_batch_size: int,
                                  device: Device = None) -> Tuple[torch.Tensor, torch.Tensor]:

        device = device or self.default_device
        inner_batch_size = inner_batch_size or self.default_batch_size
        outer_batch_size = conditioned_observation.shape[0]

        conditioned_observation = conditioned_observation.to(device)

        # let i be the index of the player conditioned on.
        i = conditioned_player

        # repeat each entry of conditioned_observation inner_batch_size times.
        v_i = (conditioned_observation
               .to(device)
               # add a dim for inner_batch after outer batch_dim
               .unsqueeze(1)
               .repeat(1, inner_batch_size, 1)
               )

        # individual components z_j are conditionally independent of z_i.
        # start by sampling these (and overwriting ith entry with actual obs.)

        # create repeated entries for conditioned_player
        z = torch.empty([outer_batch_size,
                         inner_batch_size,
                         self.n_players,
                         self.valuation_size], device = device) \
            .uniform_(self.u_lo, self.u_hi)
        z[...,i,:] = self._draw_z_given_v(v_i)

        # we have
        # v_j = w*s + (1-w) z_j
        #     = w*( 1/w*(v_i - (1-w)*z_i) ) + (1-w)z_j
        #     = v_i + (1-w)*(z_j - z_i)

        v =(1 - self._weight)*(z - z[...,[i],:]) + \
            v_i.view(outer_batch_size, inner_batch_size, 1, self.valuation_size)

        # private values setting: observations = valuations
        return v, v

    def _draw_z_given_v(self, v: torch.Tensor):
        """
        Sample private component of local bidder conditioned on its
        observation.
        (Remember: v_i = (1-w) z_i +  s) where z_i is common component,
        s is shared component.

        Returns tensor of shape (v.shape[0], batch_size).

        TODO: For now see online supplement of Nature submission for how these
        are computed. --> add detailed documentation here.

        Args:
            v (tensor) a tensor of shape (outer_batch_size, inner_batch_size, valuation_size)

        Returns:
            z (tensor) of shape (outer_batch_size, inner_batch_size,
                                 valuation_size) on the same device as v
        """
        # TODO: we might want to ensure that we use the same (quasi-)random
        # nubmers in the inner_batch for each outer_batch? [If we use
        # quasi-randomness, this would certainly be required, for pseudo-random
        # numbers it shouldn't be an issue.]
        # NOTE: the above todo would break the correlation test for
        # inner_batch_sizes of 1!
        device = v.device

        # degenerate case: semantically, z doesn't matter,
        # but we still need a separate implementation of the interface to
        # avoid division by 0 (as gama=1 implies w=1).
        if self.gamma == 1.0:
            return torch.empty_like(v) \
                .uniform_(self.u_lo, self.u_hi)

        # the conditional V_1 is uniformly distributed on [lower, upper] below:
        w = self._weight.to(device)
        lower = torch.max(self.u_lo*torch.ones_like(v), (v - w*self.u_hi)/(1 - w))
        upper = torch.min(self.u_hi*torch.ones_like(v), (v - w*self.u_lo)/(1 - w))

        return (upper - lower) * torch.empty_like(v).uniform_(0,1) + lower


class MineralRightsValuationObservationSampler(ValuationObservationSampler):
    """The 'Mineral Rights' model is a common value model:
    There is a uniformly distributed common value of the item(s),
    each agent's  observation is then uniformly drawn from U[0,2v].
    See Kishna (2009), Example 6.1
    """

    def __init__(self, n_players: int, valuation_size: int = 1,
                 common_value_lo: float = 0.0, common_value_hi: float = 1.0,
                 default_batch_size: int = 1, default_device = None):
        """
        Args:
            n_players
            valuation_size
            common_value_lo: lower bound for uniform common value
            common_value_hi: upper bound for uniform common value
            default_batch_size
            default_device
        """
        observation_size = valuation_size

        assert common_value_lo >= 0, "valuations must be nonnegative"
        assert common_value_hi >= common_value_lo, "upper bound must larger than lower bound"

        self._common_value_lo = common_value_lo
        self._common_value_hi = common_value_hi

        support_bounds = torch.tensor(
            [common_value_lo, common_value_hi]
            ).repeat([n_players, valuation_size, 1])

        super().__init__(
            n_players,
            valuation_size,
            observation_size,
            support_bounds,
            default_batch_size=default_batch_size,
            default_device=default_device)

    def draw_profiles(self, batch_sizes: List[int] = None, device = None) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_sizes = self._parse_batch_sizes_arg(batch_sizes)
        device = device or self.default_device

        common_value = (self._common_value_hi - self._common_value_lo) * \
            torch.empty([*batch_sizes, 1, self.valuation_size],
                        device=device).uniform_() + \
            self._common_value_lo

        valuations = common_value.repeat([*([1]*len(batch_sizes)), self.n_players, 1])

        individual_factors = torch.empty_like(valuations).uniform_()
        observations = 2*individual_factors*common_value

        return valuations, observations

    def draw_conditional_profiles(self,
                                  conditioned_player: int,
                                  conditioned_observation: torch.Tensor,
                                  inner_batch_size: int,
                                  device: Device = None) -> Tuple[torch.Tensor, torch.Tensor]:

        device = device or self.default_device
        inner_batch = inner_batch_size or self.default_batch_size
        outer_batch = conditioned_observation.shape[0]

        # shorthands
        i = conditioned_player

        v = self._draw_v_given_o(conditioned_observation, inner_batch)

        valuations = v.to(device) \
                      .view(outer_batch, inner_batch, 1, self.valuation_size) \
                      .repeat([1, 1, self.n_players, 1])

        # draw individual factors, then overwrite for i
        x = torch.empty(
            [outer_batch, inner_batch, self.n_players, self.valuation_size],
            device=device).uniform_()
        x[..., i, :] = conditioned_observation \
            .view(outer_batch, 1, -1) \
            .repeat(1, inner_batch, 1) / (2*v)
        # previous operation introduces NaNs when cond_observation == 0
        # and u_lo ==0 ==> v == 0, thus x/v = 0/0.
        # in this case, we set x to 0.
        # this should only happen for player i, if it happens somewhere else,
        # we don't overwrite to avoid missing errors downstream.
        x[...,i,:][x[...,i,:].isnan()] = 0.0

        observations = 2*valuations*x

        return valuations, observations

    def _draw_v_given_o(self, o_i, inner_batch_size):
        """For a batch of player i's observation o_i, draws inner_batch_size
        many possible common valuation v.

        To do so, we use the inverse CDF method.

        Args:
            o_i: observation of a single player (outer_batch_size x valuation_size)

        Returns:
            common_value: tensor of dim (outer_batch, inner_batch, valuation_size) on same device as o_i
        """

        o_i = o_i.unsqueeze(1).repeat(1, inner_batch_size, 1)

        # Let o = 2*v*x where v is U[lo,hi], x is U[0,1]. Then
        #  1. f(o|v) ∝ 1/v on [0, 2v]
        #  2. with new_lo = max(o/2, lo), and c=log(hi)-log(new_lo), we get
        #  f(o) = ∫_v=new_lo^hi f(o|v) ∝ c on [new_lo, hi]
        #  3. pdf: f(v|o) = f(o|v)f(v)/f(o) ∝  1/(cv) on [new_lo, hi]
        #     cdf: F(v|o) = (log(v) - log(new_lo)) / c
        #     icdf: inv(F)(u) = hi**u * lo**(1-u)
        # we can then sample V|o via the icdf method:

        # adjusted bounds for V|o
        lo = torch.max(o_i/2, torch.zeros_like(o_i) + self._common_value_lo)
        hi = torch.zeros_like(o_i) + self._common_value_hi

        u = torch.empty_like(o_i).uniform_()
        v =(u* hi.log() + (1-u) * lo.log()).exp()
        # alternative form of the same: v= b**u * a**(1-u). Which is faster/more stable?

        return v



class AffiliatedValuationObservationSampler(ValuationObservationSampler):
    """The 'Affiliated Values Model' model. (Krishna 2009, Example 6.2).
       This is a private values model.

       Two bidders have signals

        .. math::
        o_i = z_i + s

        and valuations
        .. math::
        v_i = s + (z_1+z_2)/2 = mean_i(o_i)

        where z_i and s are i.i.d. standard uniform.
    """

    def __init__(self, n_players: int, valuation_size: int = 1,
                 u_lo: float = 0.0, u_hi: float = 1.0,
                 default_batch_size: int = 1, default_device = None):
        """
        Args:
            n_players
            valuation_size
            u_lo: lower bound for uniform distribution of z_i and s
            u_hi: upper bound for uniform distribtuion of z_i and s
            default_batch_size
            default_device
        """
        observation_size = valuation_size

        assert u_lo >= 0, "valuations must be nonnegative"
        assert u_hi > u_lo, "upper bound must larger than lower bound"

        self._u_lo = u_lo
        self._u_hi = u_hi

        support_bounds = torch.tensor(
            [u_lo, 2*u_hi]
            ).repeat([n_players, valuation_size, 1])


        super().__init__(
            n_players,
            valuation_size,
            observation_size,
            support_bounds,
            default_batch_size=default_batch_size,
            default_device=default_device)


    def draw_profiles(self, batch_sizes: List[int] = None, device = None) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_sizes = self._parse_batch_sizes_arg(batch_sizes)
        device = device or self.default_device

        z_and_s = torch.empty([*batch_sizes, self.n_players+1, self.valuation_size],
                              device=device).uniform_(self._u_lo, self._u_hi)

        weights_v = torch.column_stack([torch.ones([self.n_players]*2, device = device) / self.n_players,
                                       torch.ones([self.n_players, 1], device=device)])

        weights_o = torch.column_stack([torch.eye(self.n_players, device=device),
                                       torch.ones([self.n_players, 1], device = device)])

        # dim u represents the n+1 uniform vectors
        valuations =   torch.einsum('buv,nu->bnv', z_and_s, weights_v)
        observations = torch.einsum('buv,nu->bnv', z_and_s, weights_o)

        return valuations, observations

    def draw_conditional_profiles(self,
                                  conditioned_player: int,
                                  conditioned_observation: torch.Tensor,
                                  inner_batch_size: int,
                                  device: Device = None) -> Tuple[torch.Tensor, torch.Tensor]:
        device = device or self.default_device
        inner_batch = inner_batch_size or self.default_batch_size
        outer_batch = conditioned_observation.shape[0]

        i = conditioned_player
        o_i = conditioned_observation.repeat_interleave(inner_batch, dim=0)

        # S|o_i is uniform on [max(lo, o-hi),  min(hi, o-lo)]
        # z_i = o_i - s
        # z_j is conditionally independent of o_i
        # we can then sample o_j and v directly

        lo = torch.zeros_like(o_i) + self._u_lo
        hi = torch.zeros_like(o_i) + self._u_hi

        s_lo = torch.max(lo, o_i - hi)
        s_hi = torch.min(hi, o_i - lo)

        s = (s_hi - s_lo) * torch.empty_like(o_i).uniform_() + s_lo

        #sample for all players then overwrite for i
        z = torch.empty(
            [outer_batch*inner_batch, self.n_players, self.valuation_size],
            device = device).uniform_(self._u_lo, self._u_hi)
        z[:,i,:] = o_i - s

        observations = z + s.view(-1, 1, self.valuation_size)
        valuations = torch.sum(z, dim = 1) / self.n_players + s

        return valuations, observations


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

    ## TODO Stefan: We may want to override grid sampling in this class in
    ## order to sample only from 'triangle' rather than rectangle?

    def __init__(self, n_players: int, n_items: int = 1,
                 max_demand: int = None,
                 u_lo: float = 0.0, u_hi: float = 1.0,
                 default_batch_size: int = 1, default_device = None):
        """
        Args:
            n_players
            n_items: the number of items
            max_demand: the maximal number of items a bidder is interested in winning.
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

        return profile

class CompositeValuationObservationSampler(ValuationObservationSampler):
    """A class representing composite prior distributions that are
    made up of several groups of bidders, each of which can be represented by
    an atomic ValuationObservationSampler, and which are independent between-group
    (but not necessarily within-group).

    Limitation: The current implementation requires that all players nevertheless
    have the same valuation_size.
    """

    def __init__(self, n_players: int, valuation_size: int, observation_size: int,
                 subgroup_samplers: List[ValuationObservationSampler],
                 default_batch_size = 1, default_device = None):

        self.n_groups = len(subgroup_samplers)
        self.group_sizes = [sampler.n_players for sampler in subgroup_samplers]
        assert sum(self.group_sizes) == n_players, "number of players in subgroup don't match total n_players."
        for sampler in subgroup_samplers:
            assert sampler.valuation_size == valuation_size, "incorrect valuation size in subgroup sampler."
            assert sampler.observation_size == observation_size, "incorrect observation size in subgroup sampler"

        self.group_samplers = subgroup_samplers
        self.group_indices: List[torch.IntTensor] = [
            torch.tensor(range(sum(self.group_sizes[:i]),
                               sum(self.group_sizes[:i+1])))
            for i in range(self.n_groups)
        ]

        ## concatenate bounds in player dimension
        support_bounds = torch.vstack([s.support_bounds for s in self.group_samplers])

        super().__init__(n_players, valuation_size, observation_size, support_bounds, 
                         default_batch_size, default_device)



    def draw_profiles(self, batch_sizes: int or List[int] = None, device=None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Draws and returns a batch of valuation and observation profiles.

        Kwargs:
            batch_size (optional): int, the batch_size to draw. If none provided,
            `self.default_batch_size` will be used.
            device (optional): torch.cuda.Device, the device to draw profiles on

        Returns:
            valuations: torch.Tensor (batch_size x n_players x valuation_size): a valuation profile
            observations: torch.Tensor (batch_size x n_players x observation_size): an observation profile
        """
        device = device or self.default_device
        batch_sizes = self._parse_batch_sizes_arg(batch_sizes)

        v = torch.empty([*batch_sizes, self.n_players, self.valuation_size], device=device)
        o = torch.empty([*batch_sizes, self.n_players, self.observation_size], device=device)

        ## Draw independently for each group.

        for g in range(self.n_groups):
            # player indices in the group
            players = self.group_indices[g]
            v[:, players, :], o[:, players, :] = self.group_samplers[g].draw_profiles(batch_sizes, device)

        return v,o


    def draw_conditional_profiles(self,
                                  conditioned_player: int,
                                  conditioned_observation: torch.Tensor,
                                  inner_batch_size: int,
                                  device: Device = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Draws and returns batches conditional valuation and corresponding observation profile.
        For each entry of `conditioned_observation`, `batch_size` samples will be drawn!

        Note that here, we are returning full profiles instead (including
        `conditioned_player`'s observation and others' valuations.)

        Args:
            conditioned_player: int
                Index of the player whose observation we are conditioning on.
            conditioned_observation: torch.Tensor (`outer_batch_size` (implicit), `observation_size`)
                A (batch of) observations of player `conditioned_player`.

        Kwargs:
            batch_size (optional): int, the "inner"batch_size to draw - i.e.
            how many conditional samples to draw for each provided `conditional_observation`.
            If none provided, will use `self.default_batch_size` of the class.

        Returns:
            valuations: torch.Tensor (batch_size x n_players x valuation_size):
                a conditional valuation profile
            observations: torch.Tensor ((batch_size * `outer_batch_size`) x n_players x observation_size):
                a corresponding conditional observation profile.
                observations[:,conditioned_observation,:] will be equal to
                `conditioned_observation` repeated `batch_size` times
        """

        device = device or self.default_device
        inner_batch = inner_batch_size or self.default_batch_size
        outer_batch = conditioned_observation.shape[0]
        full_batch = inner_batch * outer_batch

        i = conditioned_player
        o_i = conditioned_observation.repeat_interleave(inner_batch, dim=0)

        cv = torch.empty([inner_batch_size, self.n_players, self.valuation_size], device=device)
        co = torch.empty([inner_batch_size, self.n_players, self.observation_size], device=device)

        ## Draw independently for each group.

        for g in range(self.n_groups):
            # player indices in the group
            players = self.group_indices[g]

            if i in players:
                # this is the group of the conditioned player, we need to sample
                # from the group's conditional distribtuion

                # i's relative position in the subgroup:
                sub_i =  i - sum(self.group_sizes[:g])

                cv[:, players, :], co[:, players, :] = \
                    self.group_samplers[g].draw_conditional_profiles(
                        conditioned_player= sub_i,
                        conditioned_observation= conditioned_observation,
                        inner_batch_size = inner_batch,
                        device = device
                    )
            else:
                # the conditioned player is not in this group, the groups draw
                # is independent of the observation
                cv[:, players, :], co[:, players, :] = \
                    self.group_samplers[g].draw_profiles(full_batch, device)

        return cv, co

class LocalGlobalCompositePVSampler(CompositeValuationObservationSampler):
    """Settings with two groups of players: The local players have
    symmetric (possibly correlated) uniform valuations on [0,1]. The
    global bidders have symmetric (possibly correlated) uniform valuations on
    [0,2].
    """

    def __init__(self, n_locals: int, n_globals: int, valuation_size: int,
                 correlation_locals = 0.0, correlation_method_locals = None,
                 correlation_globals = 0.0, correlation_method_globals = None,
                 default_batch_size = 1 , default_device = None):

        assert 0 <=correlation_locals  <= 1, "invalid locals correlation"
        assert 0 <=correlation_globals <= 1, "invalid globals correlation"

        # setup local sampler
        if correlation_locals > 0.0:
            if correlation_method_locals == 'Bernoulli':
                local_sampler_class = BernoulliWeightsCorrelatedSymmetricUniformPVSampler
            elif correlation_method_locals == 'constant':
                local_sampler_class = ConstantWeightCorrelatedSymmetricUniformPVSampler
            else:
                raise ValueError('Only "Bernoulli" and "constant" correlation methods are implemented for LocalGlobal samplers')

            sampler_locals = local_sampler_class(
                n_players=n_locals, valuation_size = valuation_size,
                correlation = correlation_locals, u_lo = 0.0, u_hi = 1.0,
                default_batch_size=default_batch_size, default_device=default_device)
        else:
            # no correlation between locals
            if correlation_locals is not None:
                warnings.warn("Warning: You specified a correlation method, but correlation is 0.0.")
            sampler_locals = UniformSymmetricIPVSampler(
                0.0, 1.0, n_locals, valuation_size, default_batch_size, default_device)

        # setup global sampler
        if correlation_globals > 0.0:
            if correlation_method_globals == 'Bernoulli':
                global_sampler_class = BernoulliWeightsCorrelatedSymmetricUniformPVSampler
            elif correlation_method_globals == 'constant':
                global_sampler_class = ConstantWeightCorrelatedSymmetricUniformPVSampler
            else:
                raise ValueError('Only "Bernoulli" and "constant" correlation methods are implemented for LocalGlobal samplers')

            sampler_globals = global_sampler_class(
                n_players=n_globals, valuation_size = valuation_size,
                correlation = correlation_globals, u_lo = 0.0, u_hi = 2.0,
                default_batch_size=default_batch_size, default_device=default_device)
        else:
            # no correlation between globals
            if correlation_globals is not None:
                warnings.warn("Warning: You specified a correlation method, but correlation is 0.0.")
            sampler_globals = UniformSymmetricIPVSampler(
                0.0, 2.0, n_globals, valuation_size, default_batch_size, default_device)

        n_players = n_locals + n_globals
        observation_size = valuation_size # this is a PV setting, valuations = observations
        subgroup_samplers = [sampler_locals, sampler_globals]

        super().__init__(n_players, valuation_size, observation_size, subgroup_samplers, default_batch_size, default_device)

class LLGSampler(LocalGlobalCompositePVSampler):
    """A sampler for the LLG settings in Ausubel & Baranov.

    Args:
        correlation (float), correlation coefficient between local bidders,
            takes values in [0.0, 1.0]
        correlation_method (str or None, default: None): The type of correlation
            model. For correlation > 0.0, must be one of 'Bernoulli' or 'constant'

    """
    def __init__(self, correlation = 0.0, correlation_method = None,
                 default_batch_size = 1, default_device= None):
        super().__init__(n_locals =2, n_globals = 1, valuation_size = 1,
                         correlation_locals=correlation, correlation_method_locals=correlation_method,
                         correlation_globals=0.0, correlation_method_globals=None,
                         default_batch_size=default_batch_size, default_device=default_device)

class LLLLGGSampler(LocalGlobalCompositePVSampler):
    """A sampler for the LLLLGG settings in Bosshard et al (2020).

    Note: while the auction is for 6 players and 8 items, our auction implementation uses symmetries and
        encodes each player's valuations with a valuation_size of 2!

    Args:
        correlation_locals (float), correlation coefficient between local bidders,
            takes values in [0.0, 1.0]
        correlation_method_locals (str or None, default: None): The type of correlation
            model. For correlation > 0.0, must be one of 'Bernoulli' or 'constant'

    """
    def __init__(self, correlation = 0.0, correlation_method = None,
                 default_batch_size = 1, default_device= None):
        super().__init__(n_locals =4, n_globals = 2, valuation_size = 2,
                         correlation_locals=correlation, correlation_method_locals=correlation_method,
                         correlation_globals=0.0, correlation_method_globals=None,
                         default_batch_size=default_batch_size, default_device=default_device)
