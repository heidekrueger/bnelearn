# -*- coding: utf-8 -*-

"""
This module implements double auctions.
"""

from abc import ABC, abstractmethod
import torch
from .mechanism import Mechanism


class DoubleAuctionMechanism(Mechanism, ABC):
    """
    Double Auction Mechanism - Interpreted as a Bayesian game.
    A Mechanism collects bids from all players, then allocates available
    items as well as payments for each of the players.
    """

    def __init__(self, n_buyers, n_sellers, **kwargs): 
        super().__init__(**kwargs)

        # 0:n_buyers: indices for buyers
        # n_buyers+1 : n_players : indices for sellers

        self.n_buyers = n_buyers
        self.n_sellers = n_sellers
        self.min_player_dim = min(self.n_buyers, self.n_sellers)


    def play(self, action_profile):
        return self.run(bids=action_profile)

    @abstractmethod
    def run(self, bids):
        """Alias for play for double auction mechanisms"""
        raise NotImplementedError()
