from abc import ABC, abstractmethod

class Game(ABC):
    def __init__(self, players, mechanism):
        self.players = players
        self.mechanism = mechanism

    @abstractmethod
    def play(self, actions):
        # get actions from players and define outcome
        pass


class BayesianGame(Game, ABC):
    pass


class Auction(BayesianGame):
    pass
