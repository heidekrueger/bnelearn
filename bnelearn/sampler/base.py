"""This class provides primitives to implement samplers that support drawing of
   valuation and observation profiles for a set of players."""

from abc import ABC, abstractmethod
from math import ceil
from typing import List, Tuple, Union
import torch
from torch.cuda import _device_t as Device


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

    def _parse_batch_sizes_arg(self, batch_sizes_argument: Union[int , List[int] , None]) -> List[int]:
        """Parses an integer batch_size_argument into a list. If none given,
           defaults to the list containing the default_batch_size of the instance.
        """
        batch_sizes = batch_sizes_argument or self.default_batch_size
        if isinstance(batch_sizes, int):
            batch_sizes = [batch_sizes]
        return batch_sizes

    @abstractmethod
    def draw_profiles(self, batch_sizes: Union[int, List[int]] = None,
                      device=None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Draws and returns a batch of valuation and observation profiles.

        Kwargs:
            batch_sizes (optional): List[int], the batch_sizes to draw. If none provided,
            `[self.default_batch_size]` will be used.
            device (optional): torch.cuda.Device, the device to draw profiles on

        Returns:
            valuations: torch.Tensor (*batch_sizes x n_players x valuation_size): a valuation profile
            observations: torch.Tensor (*batch_sizes x n_players x observation_size): an observation profile
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
            conditioned_observation: torch.Tensor (*`outer_batch_sizes` (implicit), `observation_size`)
                A (batch of) observations of player `conditioned_player`.

        Kwargs:
            batch_size (optional): int, the "inner"batch_size to draw - i.e.
            how many conditional samples to draw for each provided `conditional_observation`.
            If none provided, will use `self.default_batch_size` of the class.

        Returns:
            valuations: torch.Tensor (outer_batch_size, inner_batch_size, n_players, valuation_size):
                a conditional valuation profile
            observations: torch.Tensor (*`outer_batch_size`s, inner_batch_size, n_players, observation_size):
                a corresponding conditional observation profile.
                observations[:,conditioned_observation,:] will be equal to
                `conditioned_observation` repeated `batch_size` times
        """
        pass

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

        device = device or self.default_device

        if support_bounds is None:
            support_bounds = self.support_bounds
        bounds = support_bounds[player_position]

        # dimensionality
        dims = self.valuation_size

        # use equal density in each dimension of the valuation, such that
        # the total number of points is at least as high as the specified one
        n_points_per_dim = ceil(minimum_number_of_points**(1/dims))

        # create equidistant lines along the support in each dimension
        lines = [torch.linspace(bounds[d][0], bounds[d][1], n_points_per_dim,
                                device=device, dtype=dtype)
                 for d in range(dims)]
        grid = torch.stack(torch.meshgrid(lines), dim=-1).view(-1, dims)

        return grid

    def generate_reduced_grid(self, player_position: int, minimum_number_of_points: int,
                              dtype=torch.float, device = None) -> torch.Tensor:
        """For some samplers, the action dimension is smaller and the grid can
        be reduced to that lower dimension.
        """
        return self.generate_valuation_grid(player_position, minimum_number_of_points,
                                            dtype, device)

    def generate_action_grid(self, player_position: int, minimum_number_of_points: int,
                             dtype=torch.float, device = None) -> torch.Tensor:
        """As the grid is also used for finding best responses, some samplers
        need extensive grids that sample a broader area. (E.g. when a bidder
        with high valuations massively shads her bids.)
        """
        support_bounds = self.support_bounds

        # Grid bids should always start at zero if not specified otherwise
        support_bounds[:, :, 0] = 0

        return self.generate_valuation_grid(
            player_position=player_position, minimum_number_of_points=minimum_number_of_points,
            dtype=dtype, device=device, support_bounds=support_bounds)

    def generate_grid_cell_bounds(self, player_position: int, minimum_number_of_points: int,
                                  dtype=torch.float, device = None,
                                  support_bounds: torch.Tensor = None) -> torch.Tensor:
        """For the epsilon estimate, we need to evaluate the strategies at the
        boundaries of the grid cells.
        """
        device = device or self.default_device

        if support_bounds is None:
            support_bounds = self.support_bounds
        bounds = support_bounds[player_position]

        # dimensionality
        dims = self.valuation_size

        # use equal density in each dimension of the valuation, such that
        # the total number of points is at least as high as the specified one
        n_points_per_dim = ceil(minimum_number_of_points**(1/dims))

        # create equidistant lines along the support in each dimension
        lines = [torch.linspace(bounds[d][0], bounds[d][1], n_points_per_dim,
                                device=device, dtype=dtype)
                 for d in range(dims)]
        corners = torch.stack(torch.meshgrid(lines), dim=-1)

        # TODO: speedup -> for more than 2 dims, we can reduce the number of
        # corners to 2 for these rectangular grids

        # TODO: only tested for 1 and 2 dims
        grid_cell_lower_corners = grid_cell_upper_corners = corners
        for d in range(dims):
            grid_cell_lower_corners = grid_cell_lower_corners.narrow(d, 0, n_points_per_dim-1)
            grid_cell_upper_corners = grid_cell_upper_corners.narrow(d, 1, n_points_per_dim-1)

        return grid_cell_lower_corners.flatten(0, -2), grid_cell_upper_corners.flatten(0, -2)

class PVSampler(ValuationObservationSampler, ABC):
    """A sampler for Private Value settings, i.e. when observations and
     valuations are identical.
    """

    def __init__(self, n_players: int, valuation_size: int, support_bounds,
                 default_batch_size: int = 1, default_device: Device = None):
        super().__init__(n_players, valuation_size, valuation_size, support_bounds,
                         default_batch_size, default_device)

    @abstractmethod
    def _sample(self, batch_sizes: Union[int, List[int]], device: Device) -> torch.Tensor:
        """Returns a batch of profiles (which are both valuations and observations)"""

    def draw_profiles(self, batch_sizes: int or List[int] = None,
                      device: Device = None) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_sizes = self._parse_batch_sizes_arg(batch_sizes)
        device = device or self.default_device
        # In the PV setting, valuations and observations are identical.
        profile = self._sample(batch_sizes, device)
        return profile, profile

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
            batch_sizes (optional): List[int], the batch_size to draw. If none provided,
            `self.default_batch_size` will be used.
            device (optional): torch.cuda.Device, the device to draw profiles on

        Returns:
            valuations: torch.Tensor (*batch_sizes x n_players x valuation_size): a valuation profile
            observations: torch.Tensor (*batch_sizes x n_players x observation_size): an observation profile
        """
        device = device or self.default_device
        batch_sizes: List[int] = self._parse_batch_sizes_arg(batch_sizes)

        v = torch.empty([*batch_sizes, self.n_players, self.valuation_size], device=device)
        o = torch.empty([*batch_sizes, self.n_players, self.observation_size], device=device)

        ## Draw independently for each group.

        for g in range(self.n_groups):
            # player indices in the group
            players = self.group_indices[g]
            v[..., players, :], o[..., players, :] = self.group_samplers[g].draw_profiles(batch_sizes, device)

        return v, o

    def draw_conditional_profiles(self,
                                  conditioned_player: int,
                                  conditioned_observation: torch.Tensor,
                                  inner_batch_size: int,
                                  device: Device = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Draws and returns batches conditional valuation and corresponding observation profile.
        For each entry of `conditioned_observation`, `inner_batch_size` samples will be drawn!

        Note that here, we are returning full profiles instead (including
        `conditioned_player`'s observation and others' valuations.)

        Args:
            conditioned_player: int
                Index of the player whose observation we are conditioning on.
            conditioned_observation: torch.Tensor (`*outer_batch_sizes` (implicit), `observation_size`)
                A batch of/batches of observations of player `conditioned_player`.

        Kwargs:
            batch_size (optional): int, the "inner"batch_size to draw - i.e.
            how many conditional samples to draw for each provided `conditional_observation`.
            If none provided, will use `self.default_batch_size` of the class.

        Returns:
            valuations: torch.Tensor (batch_size x n_players x valuation_size):
                a conditional valuation profile
            observations: torch.Tensor (`*outer_batch_sizes`, inner_batch_size, n_players, observation_size):
                a corresponding conditional observation profile.
                observations[...,conditioned_observation,:] will be equal to
                `conditioned_observation` repeated `batch_size` times
        """

        device = device or self.default_device
        inner_batch = inner_batch_size or self.default_batch_size
        *outer_batches, observation_size = conditioned_observation.shape #pylint: disable=unused-variable

        i = conditioned_player

        cv = torch.empty([*outer_batches, inner_batch, self.n_players, self.valuation_size], device=device)
        co = torch.empty([*outer_batches, inner_batch, self.n_players, self.observation_size], device=device)

        ## Draw independently for each group.

        for g in range(self.n_groups):
            # player indices in the group
            players = self.group_indices[g]

            if i in players:
                # this is the group of the conditioned player, we need to sample
                # from the group's conditional distribution

                # i's relative position in the subgroup:
                sub_i =  i - sum(self.group_sizes[:g])

                cv[..., players, :], co[..., players, :] = \
                    self.group_samplers[g].draw_conditional_profiles(
                        conditioned_player=sub_i,
                        conditioned_observation=conditioned_observation,
                        inner_batch_size=inner_batch,
                        device=device
                    )
            else:
                # the conditioned player is not in this group, the group's draw
                # is independent of the observation
                cv[..., players, :], co[..., players, :] = \
                    self.group_samplers[g].draw_profiles([*outer_batches, inner_batch], device)

        return cv, co

    def generate_valuation_grid(self, player_position: int, minimum_number_of_points: int,
                                dtype=torch.float, device = None,
                                support_bounds: torch.Tensor = None) -> torch.Tensor:
        """Possibly need to call specific grid sampling"""
        for g in range(self.n_groups):  # iterate over groups
            players = self.group_indices[g]  # player_positions within group
            for i, pos in enumerate(players):  # need to keep track of list index: i != pos
                if player_position == pos:
                    return self.group_samplers[g].generate_valuation_grid(
                        i, minimum_number_of_points, dtype, device, support_bounds
                    )
