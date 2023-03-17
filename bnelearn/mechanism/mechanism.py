# -*- coding: utf-8 -*-

"""
This module implements games such as matrix games and auctions.
"""

from abc import ABC, abstractmethod
from typing import Tuple
import warnings

# pylint: disable=E1102
import torch


class Game(ABC):
    """
    Base class for any kind of games
    """

    def __init__(self, cuda: bool = True):
        self.cuda = cuda and torch.cuda.is_available()
        self.device = 'cuda' if self.cuda else 'cpu'

    @abstractmethod
    def play(self, action_profile):
        """Play the game!"""
        # get actions from players and define outcome
        raise NotImplementedError()


class Mechanism(Game, ABC):
    """
    Auction Mechanism - Interpreted as a Bayesian game.
    A Mechanism collects bids from all players, then allocates available
    items as well as payments for each of the players.
    """
    def __init__(self, cuda: bool = True, smoothing_temperature: float = None):
        super().__init__(cuda)
        if smoothing_temperature == 0:
            warnings.warn('Smoothing temperature must be larger than zero.')
            self.smoothing_temperature = 1e-16
        else:
            self.smoothing_temperature = smoothing_temperature

    def play(self, action_profile, smooth_market: bool=False) -> Tuple[torch.Tensor, torch.Tensor]:
        if smooth_market:
            return self.run(bids=action_profile, smooth_market=True)
        else:  # some mechanisms do not support smooth markets yet
            return self.run(bids=action_profile)

    @abstractmethod
    def run(self, bids) -> Tuple[torch.Tensor, torch.Tensor]:
        """Alias for play for auction mechanisms"""
        raise NotImplementedError()
