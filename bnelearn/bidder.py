from abc import ABC, abstractmethod
import torch
from torch.distributions import Distribution

class Player(ABC):
    """
        A player in a game, determined by her 
        - strategy
        - utility function over outcomes
    """
    @abstractmethod
    def get_action(self):
        """Chooses an action according to the player's strategy."""
        pass

    @abstractmethod
    def get_utility(self, outcome):
        """Calculates player's utility based on outcome of a game."""
        pass
    


class Bidder(Player):
    """
        A player in an auction game. Has a distribution over valuations/types that is common knowledge.
    """
    def __init__(self, value_distribution: Distribution, strategy):
        self.value_distribution = value_distribution
        self.strategy = strategy

        self.valuation = torch.zeros(1)

    def draw_valuations_(self, batch_size):
        self.valuations = self.value_distribution.sample_n(batch_size)
        return self.valuation

