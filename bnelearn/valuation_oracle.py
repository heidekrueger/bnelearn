"""This class implements drawing of valuation and observation profiles."""

from abc import ABC, abstractmethod
from typing import Tuple, List, Dict
import torch
from torch import distributions
from torch.distributions import Distribution


class ValuationObservationOracle(ABC):
    """Provides functionality to draw valuation and observation profiles."""
    n_players: int = None # The number of players in the valuation profile
    valuation_size: int = None # The dimensionality / length of a single valuation vector
    observation_size: int = None # The dimensionality / length of a single observation vector
    default_batch_size: int = None # a default batch size 

    @abstractmethod
    def draw_profiles(self, batch_size: int = None) -> Tuple(torch.Tensor, torch.Tensor):
        """Draws and returns a batch of valuation and observation profiles.

        Kwargs: 
            batch_size (optional): int, the batch_size to draw. If none provided,
            `self.default_batch_size` will be used.
        
        Returns:
            valuations: torch.Tensor (batch_size x n_players x valuation_size): a valuation profile
            observations: torch.Tensor (batch_size x n_players x observation_size): an observation profile
        """

    @abstractmethod
    def draw_conditional_profiles(self,
                                  conditioned_player: int,
                                  conditioned_observation: torch.Tensor,
                                  batch_size: int) -> Tuple(torch.Tensor, torch.Tensor):
        """Draws and returns batches conditional valuation and corresponding observation profile.
        For each entry of `conditioned_observation`, `batch_size` samples will be drawn!

        These conditioanl profiles are especially relevant when estimating the 
        utility loss / exploitability of a given strategy for `conditioned_player`,
        in that case, we need access to conditional observations of others, as well as
        conditional valuations of `conditioned_player`.

        Note that here, we are returning full profiles instead (inlcuding 
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


class ValuationObservationOracle(ABC):
    """Provides functionality to draw valuation profiles."""
    n_players: int = None # The number of players in the valuation profile
    valuation_size: int = None # The dimensionality / length of a single valuation vector
    default_batch_size: int = None # a default batch size 

    @abstractmethod
    def draw_profiles(self, batch_size: int = None) -> Tuple(torch.Tensor, torch.Tensor):
        """Draws and returns a batch of valuation profiles.

        Kwargs: 
            batch_size (optional): int, the batch_size to draw. If none provided,
            `self.default_batch_size` will be used.
        
        Returns:
            valuations: torch.Tensor (batch_size x n_players x valuation_size): a valuation profile
        """


class SymmetricIPVOracle(ValuationObservationOracle): #TODO: change name to express that this is symmetric also along items, not just players?
    """A Valuation Oracle that draws valuations independently and symmetrically
    for all players and each entry of their valuation vector according to a specified
    distribution.
    """
    def __init__(self, distribution: Distribution, 
                 n_players: int, valuation_size: int = 1,
                 default_batch_size: int = 1
                ):
        """
        Args:
            distribution: a single-dimensional torch.distributions.Distribution.
            n_players: the number of players
            valuation_size: the length of each valuation vector
            default_batch_size: the default batch size for sampling from this instance.
        """
        self.n_players = n_players
        self.valuation_size = valuation_size
        self.default_batch_size = default_batch_size
        self.distribution = distribution

        


