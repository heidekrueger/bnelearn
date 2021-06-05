"""This module tests implemented learner in a 'static' environment."""

import torch

from bnelearn.strategy import TruthfulStrategy
from bnelearn.mechanism import FirstPriceSealedBidAuction
from bnelearn.bidder import Bidder
from bnelearn.environment import AuctionEnvironment
from bnelearn.valuation_sampler import UniformSymmetricIPVSampler

# Shared objects
cuda = torch.cuda.is_available()
device = 'cuda' if cuda else 'cpu'

n_players = 3

batch_size = 2**5
observation_size = valuation_size = action_size = 1
u_lo = 0
u_hi = 10



def strat_to_bidder(strategy, batch_size, player_position=0):
    """creates a bidder from a strategy"""
    return Bidder(strategy, player_position, batch_size)


def test_auction_environment():
    sampler = UniformSymmetricIPVSampler(
        u_lo, u_hi, n_players, valuation_size, batch_size, device
    )

    bidders = [
        strat_to_bidder(TruthfulStrategy(), batch_size, i)
        for i in range(n_players)]

    env = AuctionEnvironment(
        FirstPriceSealedBidAuction(cuda), bidders, sampler, batch_size, n_players, strat_to_bidder
    )

    reward_0 = env.get_reward(bidders[0])


    assert 1==1