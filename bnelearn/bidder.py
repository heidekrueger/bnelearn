from abc import ABC, abstractmethod

class Bidder(ABC):

    @abstractmethod
    def set_valuations(self):
        pass
    
    @abstractmethod
    def get_bids(self):
        self

