from abc import ABC, abstractmethod
import numpy as np

class Mechanism(ABC):
    """
    Abstract class.
    A Mechanism collects bids from all players, then allocates available
    items as well as payments for each of the players.
    """

    @abstractmethod
    def play(self, bids):
        pass


class FirstPriceSealedBidAuction(Mechanism):

    @staticmethod
    def play(bids: np.array):
        allocations = {}
        payments = {}
        
        
        highest_bidders = bids[bids == bids.max()]
        

        winner = np.random.choice(highest_bidders)
        winning_bid = bids[winner]

        allocations.update({winner: 1})
        payments.update({winner: winning_bid})

        return(allocations, payments)
