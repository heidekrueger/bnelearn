from abc import ABC, abstractmethod
import torch
from torch.distributions import Distribution

class Player(ABC):
    """
        A player in a game, determined by her 
        - strategy
        - utility function over outcomes
    """

    def __init__(self, strategy, batch_size=1, n_players=2, cuda=True):
        self.cuda = cuda and torch.cuda.is_available()
        self.device = 'cuda' if self.cuda else 'cpu'
        self.strategy = strategy
        self.batch_size = batch_size
        self.n_players = n_players
    
    def get_action(self):
        """Chooses an action according to the player's strategy."""
        return self.strategy.play()

    @abstractmethod
    def get_utility(self, outcome):
        """Calculates player's utility based on outcome of a game."""
        pass

class Bidder(Player):
    """
        A player in an auction game. Has a distribution over valuations/types that is common knowledge.
    """
    def __init__(self,
                 value_distribution: Distribution,
                 strategy,
                 batch_size=1,
                 n_items = 1, n_players=2,
                 cuda=True
                 ):
        super().__init__(strategy, batch_size, n_players, cuda)

        self.value_distribution = value_distribution
        self.n_items = n_items
        self.valuations = torch.zeros(batch_size, n_items, device = self.device)
        

    ### Alternative Constructors #############
    @classmethod
    def uniform(cls, lower, upper, strategy, **kwargs):
        dist = torch.distributions.uniform.Uniform(low = lower, high=upper)
        return cls(dist, strategy, **kwargs)
    
    @classmethod
    def normal(cls, mean, stddev, strategy, **kwargs):
        dist = torch.distributions.normal.Normal(loc = mean, scale = stddev)
        return cls(dist, strategy, **kwargs)

    ### Members ####################

    def draw_valuations_(self):
        # If in place sampling is available for our distribution, use it!
        # This will save time for memory allocation and/or copying between devices
        # As sampling from general torch.distribution is only available on CPU.

        # uniform
        if isinstance(self.value_distribution, torch.distributions.uniform.Uniform):
            self.valuations.uniform_(self.value_distribution.low, self.value_distribution.high)
        elif isinstance(self.value_distribution, torch.distributions.normal.Normal):
            self.valuations.normal_(mean = self.value_distribution.loc, std = self.value_distribution.scale)
        # TODO: add additional internal in-place samplers
        else:
            self.valuations = self.value_distribution.rsample(self.valuations.size()).to(self.device)
        return self.valuations

    def get_utility(self, allocations, payments):

        assert allocations.dim() == 2 # batch_size x items
        assert payments.dim() == 1 # batch_size

        self.utility = (self.valuations * allocations).sum(dim=1) - payments
        return self.utility
    
    def get_action(self):
        inputs = self.valuations.view(self.batch_size, -1)
        return self.strategy.play(inputs)
