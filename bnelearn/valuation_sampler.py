"""This class implements drawing of valuation and observation profiles."""

from abc import ABC, abstractmethod
from typing import Tuple, List, Dict
import torch
from torch import distributions
from torch.distributions import Distribution
from torch.distributions.independent import Independent
from torch.cuda import _device_t as Device


class ValuationObservationSampler(ABC):
    """Provides functionality to draw valuation and observation profiles."""
    n_players: int = None # The number of players in the valuation profile
    valuation_size: int = None # The dimensionality / length of a single valuation vector
    observation_size: int = None # The dimensionality / length of a single observation vector
    default_batch_size: int = None # a default batch size 

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
                                  batch_size: int, device = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Draws and returns batches conditional valuation and corresponding observation profile.
        For each entry of `conditioned_observation`, `batch_size` samples will be drawn!

        Note that here, we are returning full profiles instead (including 
        `conditioned_player`'s observation and others' valuations.)

        Args:
            conditioned_player: int 
                Index of the player whose observation we are conditioning on.
            conditioned_observation: torch.Tensor (`outer_batch_size` (implicit) x `observation_size`)
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


# class ValuationSampler(ABC):
#     """Provides functionality to draw valuation profiles."""
#     n_players: int = None # The number of players in the valuation profile
#     valuation_size: int = None # The dimensionality / length of a single valuation vector
#     default_batch_size: int = None # a default batch size 

#     @abstractmethod
#     def sample(self, batch_size: int = None, device = None) -> torch.Tensor:
#         """Draws and returns a batch of valuation profiles.

#         Kwargs: 
#             batch_size (optional): int, the batch_size to draw. If none provided,
#             `self.default_batch_size` will be used.
        
#         Returns:
#             valuations: torch.Tensor (batch_size x n_players x valuation_size): a valuation profile
#         """


class SymmetricIPVSampler(ValuationObservationSampler): #TODO: change name to express that this is symmetric also along items, not just players?
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
        self.n_players = n_players
        self.valuation_size = valuation_size
        self.default_batch_size = default_batch_size
        self.default_device = (default_device or 'cuda') if torch.cuda.is_available() else 'cpu'
        self.base_distribution = distribution
        self.distribution = self.base_distribution.expand([n_players, valuation_size])


    def _sample(self, batch_size: int, device: Device) -> torch.Tensor:
        """Draws a batch of observation/valuation profiles (equivalent in IPV)"""
        batch_size = batch_size or self.default_batch_size
        device = device or self.default_device

        return self.distribution.sample([batch_size]).to(device)

    def draw_profiles(self, batch_size: int = None,
                      device: Device = None) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = batch_size or self.default_batch_size
        device = device or self.default_device
        # In the symmetric IPV setting, valuations and observations are identical.
        profile = self._sample(batch_size, device)
        return profile, profile

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

