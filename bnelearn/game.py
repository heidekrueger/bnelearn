from abc import ABC, abstractmethod

class Game(ABC):
    """
        Abstract class for games 
    """ 

    def __init__(self, players, mechanism):
        self.players = players
        self.mechanism = mechanism
    
    @abstractmethod
    def play(self):
        # get actions from players and define outcome
        pass
