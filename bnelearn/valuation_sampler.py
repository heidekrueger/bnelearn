"""This class implements drawing of valuation and observation profiles."""

from abc import ABC, abstractmethod
from math import sqrt
from typing import List, Tuple

import torch
from torch.cuda import _device_t as Device, device_of
from torch.distributions import Distribution
from torch.overrides import is_tensor_like


class ValuationObservationSampler(ABC):
    """Provides functionality to draw valuation and observation profiles."""

    def __init__(self, n_players, valuation_size, observation_size,
                 default_batch_size = 1, default_device = None):
        self.n_players: int = n_players # The number of players in the valuation profile
        self.valuation_size: int = valuation_size # The dimensionality / length of a single valuation vector
        self.observation_size: int = observation_size # The dimensionality / length of a single observation vector
        self.default_batch_size: int = default_batch_size # a default batch size
        self.default_device: Device = (default_device or 'cuda') if torch.cuda.is_available() else 'cpu'

    @abstractmethod
    def draw_profiles(self, batch_size: int = None, device=None) -> Tuple[torch.Tensor, torch.Tensor]:
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
                                  batch_size: int, device: Device = None) -> Tuple[torch.Tensor, torch.Tensor]:
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


class PVSampler(ValuationObservationSampler, ABC):
    """A sampler for Private Value settings, i.e. when observations and
     valuations are identical
    """
    def __init__(self, n_players: int, valuation_size: int,
                 default_batch_size: int = 1, default_device: Device = None):
        super().__init__(n_players, valuation_size, valuation_size,
                         default_batch_size, default_device)

    @abstractmethod
    def _sample(self, batch_size: int, device: Device) -> torch.Tensor:
        """Returns a batch of profiles (which are both valuations and observations)"""

    def draw_profiles(self, batch_size: int = None,
                      device: Device = None) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = batch_size or self.default_batch_size
        device = device or self.default_device
        # In the PV setting, valuations and observations are identical.
        profile = self._sample(batch_size, device)
        return profile, profile

#TODO: change name to express that this is symmetric also along items, not just players?
class SymmetricIPVSampler(PVSampler):
    """A Valuation Oracle that draws valuations independently and symmetrically
    for all players and each entry of their valuation vector according to a specified
    distribution.

    This base class works with all torch.distributions but requires sampling on
    cpu then moving to the device. When using cuda, use the faster,
    distribution-specific subclasses instead where provided.
    """
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
        super().__init__(n_players, valuation_size, default_batch_size, #pylint: disable=arguments-out-of-order
                         default_device)

    def _sample(self, batch_size: int, device: Device) -> torch.Tensor:
        """Draws a batch of observation/valuation profiles (equivalent in PV)"""
        batch_size = batch_size or self.default_batch_size
        device = device or self.default_device

        return self.distribution.sample([batch_size]).to(device)



    def draw_conditional_profiles(self,
                                  conditioned_player: int,
                                  conditioned_observation: torch.Tensor,
                                  batch_size: int,
                                  device: Device = None) -> Tuple[torch.Tensor, torch.Tensor]:
        # Due to independence, we can simply draw a full profile and replace the
        # conditioned_observations by their specified value.
        # Due to PV, valuations are equal to the observations

        device = device or self.default_device
        inner_batch_size = batch_size or self.default_batch_size
        outer_batch_size = conditioned_observation.shape[0]

        profile = self._sample(outer_batch_size * inner_batch_size, device)

        profile[:,conditioned_player, :] = \
            conditioned_observation.repeat(inner_batch_size, 1)

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

    def _sample(self, batch_size, device) -> torch.Tensor:
        # create an empty tensor on the output device, then sample in-place
        return torch.empty(
            [batch_size, self.n_players, self.valuation_size],
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

    def _sample(self, batch_size, device) -> torch.Tensor:
        # create empty tensor, sample in-place, clip
        return torch.empty([batch_size, self.n_players, self.valuation_size],
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
        self.u_lo = u_lo
        self.u_hi = u_hi
        self.gamma = correlation
        if weight_method in ['Bernoulli', 'constant']:
            self.method = weight_method
        else:
            raise ValueError('Unknown method, must be one of "Bernoulli", "constant"')

        super().__init__(n_players, valuation_size,
                         default_batch_size, default_device)

    @abstractmethod
    def _get_weights(self, batch_size: int, device: Device) -> torch.Tensor:
        """Returns a batch of weights according to the model
           or a constant weight for use in the entire batch.
        """

    def _sample(self, batch_size: int, device: Device) -> torch.Tensor:
        """Draws a batch of observation/valuation profiles (equivalent in PV)"""
        batch_size = batch_size or self.default_batch_size
        device = device or self.default_device

        individual_components = torch.empty(
            [batch_size, self.n_players, self.valuation_size],
            device = device) \
            .uniform_(self.u_lo, self.u_hi)

        common_component = torch.empty(
            [batch_size, 1, self.valuation_size],
            device = device) \
            .uniform_(self.u_lo, self.u_hi)

        w = self._get_weights(batch_size, device)

        return (1-w) * individual_components + w * common_component

class BernoulliWeightsCorrelatedSymmetricUniformPVSampler(CorrelatedSymmetricUniformPVSampler):

    def __init__(self, n_players: int, valuation_size: int, correlation: float,
                 u_lo: float or torch.Tensor = 0.0, u_hi: float or torch.Tensor = 1.0,
                 default_batch_size = 1, default_device: Device = None):
        super().__init__(u_lo, u_hi, n_players, valuation_size, correlation,
                         'Bernoulli', default_batch_size, default_device)

    def _get_weights(self, batch_size: int, device: Device) -> torch.Tensor:
        """Draws Bernoulli distributed weights along the batch size.

        Returns:
            w: Tensor of shape (batch_size, 1, 1)
        """

        return (torch.empty([batch_size], device=device)
                .bernoulli_(self.gamma) # different weight per batch
                .view(-1, 1, 1)) # same weight across item/bundle in each batch-instance

    def draw_conditional_profiles(self, conditioned_player: int,
                                  conditioned_observation: torch.Tensor,
                                  batch_size: int, device: Device  = None
                                  ) -> Tuple[torch.Tensor, torch.Tensor]:
        device = device or self.default_device
        outer_batch_size = conditioned_observation.shape[0]
        inner_batch_size = batch_size

        conditioned_observation = conditioned_observation.to(device)
        # let i be the index of the player conditioned on.
        i = conditioned_player

        # repeat each entry of conditioned_observation inner_batch_size times.
        v_i = conditioned_observation \
            .repeat_interleave(inner_batch_size, dim=0)


        # each observation v_i has a probability of self.gamma of being the
        # common component v_i=s, and (1-self.gamma) of being the player's
        # individual component v_i=z_i. In the former case, all players have
        # the same obs v_j=s=v_i. Otherwise, each has valuation v_j=z_j
        # equal to their individual component z_j.

        # Individual components z_j are conditionally independent of z_i.
        # Start by sampling these (and overwriting ith entry with actual obs.)
        # (ith's entry is technically incorrect, but the cases
        # where v_i != z_i are disregarded by the weights drawn below.)
        z = torch.empty([outer_batch_size*inner_batch_size,
                         self.n_players,
                         self.valuation_size], device = device) \
            .uniform_(self.u_lo, self.u_hi)
        z[:,i,:] = v_i



        # NOTE: with our current test (e.g. testing correlation matrix
        # of conditional valuation profile for large outer_batch and
        # inner_batch of 1), we cannot use the same drawn weights in each outer
        # batch, otherwise, we'll always end up perfectly correlated, or not
        # at all, rather than the correct amount.

        w = torch.empty([outer_batch_size * inner_batch_size, 1, 1], device=device) \
            .bernoulli_(self.gamma) \

        # sample valuations directly:
        # either individual component of each player z_j,
        # or common component s=v_i, which we stored in the ith entry of the z tensor
        v = (1-w)*z + w * z[:,[i],:]

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


    def _get_weights(self, batch_size: int, device) -> torch.Tensor:
        """Draws Bernoulli distributed weights along the batch size.
        """
        # batch size is ignored, we always return a scalar.
        return self._weight.to(device)

    def draw_conditional_profiles(self,
                                  conditioned_player: int,
                                  conditioned_observation: torch.Tensor,
                                  batch_size: int,
                                  device: Device = None) -> Tuple[torch.Tensor, torch.Tensor]:

        device = device or self.default_device
        inner_batch_size = batch_size or self.default_batch_size
        outer_batch_size = conditioned_observation.shape[0]

        conditioned_observation = conditioned_observation.to(device)

        # let i be the index of the player conditioned on.
        i = conditioned_player

        # repeat each entry of conditioned_observation inner_batch_size times.
        v_i = conditioned_observation \
            .to(device) \
            .repeat_interleave(inner_batch_size, dim=0)

        # individual components z_j are conditionally independent of z_i.
        # start by sampling these (and overwriting ith entry with actual obs.)

        # create repeated entries for conditioned_player
        z = torch.empty([outer_batch_size*inner_batch_size,
                         self.n_players,
                         self.valuation_size], device = device) \
            .uniform_(self.u_lo, self.u_hi)
        z[:,i,:] = self._draw_z_given_v(v_i)

        # we have
        # v_j = w*s + (1-w) z_j
        #     = w*( 1/w*(v_i - (1-w)*z_i) ) + (1-w)z_j
        #     = v_i + (1-w)*(z_j - z_i)

        v =(1 - self._weight)*(z - z[:,[i],:]) + \
            v_i.view(outer_batch_size*inner_batch_size, 1, self.valuation_size)

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
            v (tensor) a tensor of shape (outer_batch_size*inner_batch_size, valuation_size)

        Returns:
            z (tensor) of shape (outer_batch_size * inner_batch_size,
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
            #return torch.empty((inner_batch_size, 1), device=device) \
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

        super().__init__(
            n_players,
            valuation_size,
            observation_size,
            default_batch_size=default_batch_size,
            default_device=default_device)

        assert common_value_lo >= 0, "valuations must be nonnegative"
        assert common_value_hi >= common_value_lo, "upper bound must larger than lower bound"

        self._common_value_lo = common_value_lo
        self._common_value_hi = common_value_hi


    def draw_profiles(self, batch_size: int = None, device = None) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = batch_size or self.default_batch_size
        device = device or self.default_device

        common_value = (self._common_value_hi - self._common_value_lo) * \
            torch.empty([batch_size, 1, self.valuation_size],
                        device=device).uniform_() + \
            self._common_value_lo

        valuations = common_value.repeat([1, self.n_players, 1])

        individual_factors = torch.empty_like(valuations).uniform_()
        observations = 2*individual_factors*common_value

        return valuations, observations

    def draw_conditional_profiles(self,
                                  conditioned_player: int,
                                  conditioned_observation: torch.Tensor,
                                  batch_size: int,
                                  device: Device = None) -> Tuple[torch.Tensor, torch.Tensor]:

        device = device or self.default_device
        inner_batch = batch_size or self.default_batch_size
        outer_batch = conditioned_observation.shape[0]

        # shorthands
        i = conditioned_player

        v = self._draw_v_given_o(conditioned_observation, inner_batch)

        valuations = v.to(device) \
                      .view(-1, 1, self.valuation_size) \
                      .repeat([1, self.n_players, 1])

        # draw individual factors, then overwrite for i
        x = torch.empty(
            [outer_batch*inner_batch, self.n_players, self.valuation_size],
            device=device).uniform_()
        x[:, i, :] = conditioned_observation \
            .repeat_interleave(inner_batch, dim=0) / (2*v)
        # previous operation introduces NaNs when cond_observation == 0
        # and u_lo ==0 ==> v == 0, thus x/v = 0/0.
        # in this case, we set x to 0.
        # this should only happen for player i, if it happens somewhere else,
        # we don't overwrite to avoid missing errors downstream.
        x[:,i,:][x[:,i,:].isnan()] = 0.0

        observations = 2*valuations*x

        return valuations, observations

    def _draw_v_given_o(self, o_i, inner_batch_size):
        """For a batch of player i's observation o_i, draws inner_batch_size
        many possible common valuation v.

        To do so, we use the inverse CDF method.

        Args:
            o_i: observation of a single player (outer_batch_size x valuation_size)

        Returns:
            common_value: tensor of dim (outer_batch x inner_batch, valuation_size) on same device as o_i
        """

        o_i = o_i.repeat_interleave(inner_batch_size, dim=0)

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

        super().__init__(
            n_players,
            valuation_size,
            observation_size,
            default_batch_size=default_batch_size,
            default_device=default_device)

        assert u_lo >= 0, "valuations must be nonnegative"
        assert u_hi > u_lo, "upper bound must larger than lower bound"

        self._u_lo = u_lo
        self._u_hi = u_hi


    def draw_profiles(self, batch_size: int = None, device = None) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = batch_size or self.default_batch_size
        device = device or self.default_device

        z_and_s = torch.empty([batch_size, self.n_players+1, self.valuation_size],
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
                                  batch_size: int,
                                  device: Device = None) -> Tuple[torch.Tensor, torch.Tensor]:
        device = device or self.default_device
        inner_batch = batch_size or self.default_batch_size
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
