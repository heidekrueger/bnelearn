from abc import ABC, abstractmethod

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
    


class Bidder(Player, ABC):

    @abstractmethod
    def get_valuations(self):
        pass
    
    @abstractmethod
    def get_bids(self):
        self

