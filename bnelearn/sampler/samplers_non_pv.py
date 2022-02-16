"""This module implements samplers that do not adhere to the private values setting."""

from typing import List, Tuple
import torch
from torch.cuda import _device_t as Device
from .base import ValuationObservationSampler, PVSampler

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
        *outer_batch_sizes, observation_size = conditioned_observation.shape

        assert observation_size == self.valuation_size

        # shorthands
        i = conditioned_player

        v = self._draw_v_given_o(conditioned_observation, inner_batch)

        valuations = v.to(device) \
                      .view(*outer_batch_sizes, inner_batch, 1, observation_size) \
                      .repeat(*([1]*len(outer_batch_sizes)), 1, self.n_players, 1)

        # draw individual factors, then overwrite for i
        x = torch.empty(
            [*outer_batch_sizes, inner_batch, self.n_players, observation_size],
            device=device).uniform_()
        x[..., i, :] = conditioned_observation \
            .unsqueeze(-2) \
            .repeat(*([1]*len(outer_batch_sizes)), inner_batch, 1) / (2*v)
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
            o_i: observation of a single player (*outer_batch_sizes x valuation_size)

        Returns:
            common_value: tensor of dim (*outer_batch_sizes, inner_batch, valuation_size)
                on same device as o_i
        """

        *outer_batch_sizes, _ = o_i.shape

        o_i = o_i.unsqueeze(-2).repeat(*([1]*len(outer_batch_sizes)), inner_batch_size, 1)

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
        v = (u* hi.log() + (1-u) * lo.log()).exp()
        # alternative form of the same: v= b**u * a**(1-u). Which is faster/more stable?

        return v

    def generate_valuation_grid(self, player_position: int, minimum_number_of_points: int,
                                dtype=torch.float, device = None,
                                support_bounds: torch.Tensor = None) -> torch.Tensor:
        """This setting needs larger bounds for the grid."""
        return 2 * super().generate_valuation_grid(player_position=player_position,
                                                   minimum_number_of_points=minimum_number_of_points,
                                                   dtype=dtype, device=device, support_bounds=support_bounds)

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
        *outer_batch_sizes, observation_size = conditioned_observation.shape

        assert observation_size == self.valuation_size

        i = conditioned_player
        o_i = conditioned_observation \
            .view(*outer_batch_sizes, 1, observation_size) \
            .repeat(*([1]*len(outer_batch_sizes)), inner_batch_size, 1)

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
            [*outer_batch_sizes, inner_batch, self.n_players, self.valuation_size],
            device = device).uniform_(self._u_lo, self._u_hi)
        z[..., i, :] = (o_i - s).view_as(z[..., i, :])

        observations = z + s.view(*outer_batch_sizes, inner_batch, 1, self.valuation_size)
        valuations = torch.sum(z, dim=-2) / self.n_players + s

        # same valuations for all agents
        valuations = valuations \
            .view(*outer_batch_sizes, inner_batch, 1, self.valuation_size) \
            .repeat(*([1]*len(outer_batch_sizes)), 1, self.n_players, 1)

        return valuations, observations

class CommonValueSampler(PVSampler):
    
    def __init__(self, 
                 n_players, 
                 valuation_size, 
                 support_bounds, 
                 default_batch_size=1, 
                 default_device=None):  

        self.common_value = support_bounds[0][0][0]

        super().__init__(n_players, valuation_size, support_bounds, default_batch_size, default_device)

    def draw_profiles(self, batch_sizes: List[int] = None, device = None) -> Tuple[torch.Tensor, torch.Tensor]:
        
        valuations = torch.ones(*batch_sizes, self.n_players, self.valuation_size, device=device) * self.common_value
        observations = torch.ones(*batch_sizes, self.n_players, self.valuation_size, device=device) * self.common_value

        return valuations, observations

    def _sample(self, batch_sizes, device) -> torch.Tensor:

        batch_sizes = self._parse_batch_sizes_arg(batch_sizes)

        # create an empty tensor on the output device, then sample in-place
        return torch.ones([*batch_sizes, self.n_players, self.valuation_size], device=device) * self.common_value

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