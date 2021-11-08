"""This module implements samplers for correlated private values settings."""

from abc import ABC, abstractmethod
from math import sqrt, ceil
import warnings
from typing import List, Tuple
import torch
from torch.cuda import _device_t as Device
from .base import PVSampler, CompositeValuationObservationSampler
from .samplers_ipv import UniformSymmetricIPVSampler

ERR_MSG_INVALID_LOCAL_GLOBAL_CORRELATION_METHOD = \
    'Only "Bernoulli" and "constant" correlation methods are implemented for LocalGlobal samplers'
WRN_MSG_CORRELATED_BUT_CORR_IS_ZERO = \
    "Warning: You specified a correlation method, but correlation is 0.0."

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
            w: Tensor of shape (*batch_sizes, 1, 1)
        """
        return (torch.empty(batch_sizes, device=device)
                .bernoulli_(self.gamma) # different weight per batch
                .view(*batch_sizes, 1, 1)) # same weight across item/bundle in each batch-instance

    def draw_conditional_profiles(self, conditioned_player: int,
                                  conditioned_observation: torch.Tensor,
                                  inner_batch_size: int, device: Device  = None
                                  ) -> Tuple[torch.Tensor, torch.Tensor]:
        device = device or self.default_device
        *outer_batch_sizes, observation_size = conditioned_observation.shape

        conditioned_observation = conditioned_observation.to(device)
        # let i be the index of the player conditioned on.
        i = conditioned_player

        # repeat each entry of conditioned_observation inner_batch_size times.
        v_i = conditioned_observation \
            .view(*outer_batch_sizes, 1, observation_size) \
            .repeat(*([1] * len(outer_batch_sizes)), inner_batch_size, 1)

        # each observation v_i has a probability of self.gamma of being the
        # common component v_i=s, and (1-self.gamma) of being the player's
        # individual component v_i=z_i. In the former case, all players have
        # the same obs v_j=s=v_i. Otherwise, each has valuation v_j=z_j
        # equal to their individual component z_j.

        # Individual components z_j are conditionally independent of z_i.
        # Start by sampling these (and overwriting ith entry with actual obs.)
        # (ith's entry is technically incorrect, but the cases
        # where v_i != z_i are disregarded by the weights drawn below.)
        z = torch.empty([*outer_batch_sizes,inner_batch_size,
                         self.n_players,
                         self.valuation_size], device = device) \
            .uniform_(self.u_lo, self.u_hi)
        z[...,i,:] = v_i

        # NOTE: with our current test (e.g. testing correlation matrix
        # of conditional valuation profile for large outer_batch and
        # inner_batch of 1), we cannot use the same drawn weights in each outer
        # batch, otherwise, we'll always end up perfectly correlated, or not
        # at all, rather than the correct amount.

        w = torch.empty([*outer_batch_sizes, inner_batch_size, 1, 1], device=device) \
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
        """Returns the constant weight as a scalar tensor.
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
        *outer_batch_sizes, observation_size = conditioned_observation.shape #pylint: disable=unused-variable

        conditioned_observation = conditioned_observation.to(device)

        # let i be the index of the player conditioned on.
        i = conditioned_player

        # repeat each entry of conditioned_observation inner_batch_size times.
        v_i = (conditioned_observation
               .to(device)
               # add a dim for inner_batch after outer_batch dim(s)
               # (dim before the last dim which is val_size)
               .unsqueeze(-2)
               .repeat(*([1] * len(outer_batch_sizes)), inner_batch_size, 1)
               )

        # individual components z_j are conditionally independent of z_i.
        # start by sampling these (and overwriting ith entry with actual obs.)

        # create repeated entries for conditioned_player
        z = torch.empty([*outer_batch_sizes,
                         inner_batch_size,
                         self.n_players,
                         self.valuation_size], device = device) \
            .uniform_(self.u_lo, self.u_hi)
        z[..., i, :] = self._draw_z_given_v(v_i)

        # we have
        # v_j = w*s + (1-w) z_j
        #     = w*( 1/w*(v_i - (1-w)*z_i) ) + (1-w)z_j
        #     = v_i + (1-w)*(z_j - z_i)

        v =(1 - self._weight)*(z - z[...,[i],:]) + \
            v_i.view(*outer_batch_sizes, inner_batch_size, 1, self.valuation_size)

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
        # NOTE: the above todo would break the correlation test for inner_batch_sizes of 1!
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

        u_lo = 0.0
        u_hi_locals = 1.0
        u_hi_globals = 2.0


        sampler_locals = self._get_group_sampler(
            n_locals, correlation_locals, correlation_method_locals,
            u_lo, u_hi_locals,
            valuation_size, default_batch_size, default_device)

        sampler_globals = self._get_group_sampler(
            n_globals, correlation_globals, correlation_method_globals,
            u_lo, u_hi_globals,
            valuation_size, default_batch_size, default_device)

        n_players = n_locals + n_globals
        observation_size = valuation_size # this is a PV setting, valuations = observations
        subgroup_samplers = [sampler_locals, sampler_globals]

        super().__init__(n_players, valuation_size, observation_size, subgroup_samplers, default_batch_size, default_device)

    def _get_group_sampler(self, n_group_players, correlation, correlation_method,
                           u_lo, u_hi, 
                           valuation_size,  default_batch_size, default_device) -> PVSampler:
        """Returns a sampler of possibly correlated Uniform PV players for a
            symmetric group of players (e.g. the locals or globals)"""
        # setup local sampler
        if correlation > 0.0:
            if correlation_method == 'Bernoulli':
                sampler_class = BernoulliWeightsCorrelatedSymmetricUniformPVSampler
            elif correlation_method == 'constant':
                sampler_class = ConstantWeightCorrelatedSymmetricUniformPVSampler
            else:
                raise ValueError(ERR_MSG_INVALID_LOCAL_GLOBAL_CORRELATION_METHOD)

            sampler = sampler_class(
                n_players=n_group_players, valuation_size = valuation_size,
                correlation = correlation, u_lo = 0.0, u_hi = 1.0,
                default_batch_size=default_batch_size, default_device=default_device)
        else:
            # no correlation between locals
            if correlation_method is not None:
                warnings.warn(WRN_MSG_CORRELATED_BUT_CORR_IS_ZERO)
            sampler = UniformSymmetricIPVSampler(
                u_lo, u_hi, n_group_players, valuation_size, default_batch_size, default_device)
        return sampler

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

class LLGFullSampler(LLGSampler):
    """A sampler for the LLG full setting."""
    def _generate_grid(self, player_position: int, minimum_number_of_points: int,
                       reduced: bool, dtype=torch.float, device=None,
                       support_bounds: torch.Tensor=None) -> torch.Tensor:
        device = device or self.default_device

        if support_bounds is None:
            support_bounds = self.support_bounds
        bounds = support_bounds[player_position]

        # dimensionality
        dims = 1 if reduced else 3

        # use equal density in each dimension of the valuation, such that
        # the total number of points is at least as high as the specified one
        n_points_per_dim = ceil(minimum_number_of_points**(1/dims))

        # create equidistant lines along the support in each dimension
        lines = [torch.linspace(bounds[0][0], bounds[0][1], n_points_per_dim,
                                device=device, dtype=dtype)
                 for _ in range(dims)]
        grid = torch.stack(torch.meshgrid(lines), dim=-1).view(-1, dims)

        return grid

    def generate_valuation_grid(self, player_position: int, minimum_number_of_points: int,
                                dtype=torch.float, device=None, support_bounds=None) -> torch.Tensor:
        """Here, the grid needs to be three dimensional, as bidders can bid on
        all three items, even though they're only interested in one.
        """
        return self._generate_grid(player_position, minimum_number_of_points, False,
                                   dtype, device, support_bounds)

    def generate_reduced_grid(self, player_position: int, minimum_number_of_points: int,
                              dtype=torch.float, device=None) -> torch.Tensor:
        """Valuations are actually three dimensional, but as two dims are allways
        zero, it is sufficient to sample one dimensional data.
        """
        return self._generate_grid(player_position, minimum_number_of_points, True,
                                   dtype, device)

class LLLLGGSampler(LocalGlobalCompositePVSampler):
    """A sampler for the LLLLGG settings in Bosshard et al (2020).

    Note: while the auction is for 6 players and 8 items, our auction implementation uses symmetries and
        encodes each player's valuations with a valuation_size of 2.

    Args:
        correlation_locals (float), correlation coefficient between local bidders,
            takes values in [0.0, 1.0]
        correlation_method_locals (str or None, default: None): The type of correlation
            model. For correlation > 0.0, must be one of 'Bernoulli' or 'constant'

    """
    def __init__(self, correlation_locals = 0.0, correlation_method_locals = None,
                 default_batch_size = 1, default_device= None):
        super().__init__(n_locals =4, n_globals = 2, valuation_size = 2,
                         correlation_locals=correlation_locals, correlation_method_locals=correlation_method_locals,
                         correlation_globals=0.0, correlation_method_globals=None,
                         default_batch_size=default_batch_size, default_device=default_device)
