"""Testing correctness of regret estimator for a number of settings.
        Estimates the potential benefit of deviating from the current energy, as:
            regret(v_i) = Max_(b_i)[ E_(b_(-i))[u(v_i,b_i,b_(-i))] ]
            regret_max = Max_(v_i)[ regret(v_i) ]
            regret_expected = E_(v_i)[ regret(v_i) ]
        Input:
            agent: 1
            bid_profile: (batch_size x n_player x n_items)
            bid_i: (bid_size x n_items)
        Output:
            regret_max
            regret_expected
    bid_i always used as val_i and only using truthful bidding
"""

import pytest
import torch
from bnelearn.mechanism import LLLLGGAuction, FirstPriceSealedBidAuction
from bnelearn.strategy import TruthfulStrategy
from bnelearn.environment import AuctionEnvironment
from bnelearn.bidder import Bidder


eps = 0.0001
bids_i = torch.tensor([[0/6 + eps, 1/6 + eps,
                       2/6 + eps,3/6 + eps,
                       4/6 + eps, 5/6 + eps,
                       6/6 + eps]] , dtype = torch.float)

bids_i_comb = torch.tensor([
        [0/3 + eps, 1/3 + eps, 2/3 + eps,
         3/3 + eps]], dtype = torch.float)

# 1 Batch, 2 bidders, 1 item
bid_profile_1_2_1 = torch.tensor([
    [
        [0.9999],
        [0]]
], dtype = torch.float)
# n_bidders x 2 (avg, max)
regret_1_2_1 = torch.tensor([
    [bid_profile_1_2_1[0,0,0]-bids_i[0,0], bid_profile_1_2_1[0,0,0]-bids_i[0,0]],
    [0,0]
], dtype = torch.float)

# 2 Batch, 3 bidders, 1 item
bid_profile_2_3_1 = torch.tensor([
    [
        [0.1],
        [0.3],
        [0.5]],
    [
        [0.6],
        [0.3],
        [0.5]]
], dtype = torch.float)
regret_2_3_1 = torch.tensor([
    [0.04995,0.0999],
    [0,0],
    [0.0833,0.0833]
], dtype = torch.float)

# LLLLGG: 1 Batch, 6 bidders,2 items (bid on each, 8 in total)
bid_profile_1_6_2 = torch.tensor([
    [
        [0.011, 0.512],
        [0.021, 0.22],
        [0.031, 0.32],
        [0.041, 0.42],

        [0.89, 0.052],
        [0.061, 0.062]],
],
 dtype = torch.float)
regret_1_6_2 = torch.tensor([
    [0.512 - 1/3+eps,0.512 - 1/3+eps],
    [0.22 - 0/3+eps,0.22 - 0/3+eps],
    [0.32 - 0/3+eps,0.32 - 0/3+eps],
    [0.42 - 1/3+eps,0.42 - 1/3+eps],
    [0,0],
    [0,0]
], dtype = torch.float)
#TODO: Add one test with other pricing rule (-> and positive utility in agent)



# each test input takes form rule: string, bids:torch.tensor,
#                            expected_allocation: torch.tensor, expected_payments: torch.tensor
# Each tuple specified here will then be tested for all implemented solvers.
ids, testdata = zip(*[
    ['fpsb - single-batch, 2 bidders, 1 item', ('fpsb', FirstPriceSealedBidAuction(), bid_profile_1_2_1, bids_i, regret_1_2_1)],
    ['fpsb - 2-batch, 3 bidders, 1 item', ('fpsb', FirstPriceSealedBidAuction(), bid_profile_2_3_1, bids_i, regret_2_3_1)],
    ['fpsb - single-batch, 6 bidders, 2 item', ('fpsb', LLLLGGAuction(bid_profile_1_6_2.shape[0]), bid_profile_1_6_2, bids_i_comb, regret_1_6_2)]])


# def strat_to_bidder(strategy, batch_size=batch_size, player_position=None, cache_actions=False):
#         return Bidder.uniform(u_lo, u_hi, strategy, batch_size = batch_size,
#                               player_position=player_position, cache_actions=cache_actions, risk=risk)

@pytest.mark.parametrize("rule, mechanism, bids_profile, bids_i, expected_regret", testdata, ids=ids)
def test_regret_estimator(rule, mechanism, bids_profile, bids_i, expected_regret):
    """Run correctness test for a given LLLLGG rule"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    batch_size, n_bidders, n_items = bids_profile.shape

    agents = [None] * n_bidders
    for i in range(n_bidders):
        #TODO: Add player position
        agents[i] = Bidder.uniform(0,1,TruthfulStrategy(), player_position = i, batch_size = batch_size)
        agents[i].valuations = bids_profile[:,i,:].to(device)
    env = AuctionEnvironment(mechanism, agents, batch_size, n_bidders)
    #regret = torch.tensor(torch.zeros(n_bidders,2), dtype = torch.float, device = device)

    for i in range(n_bidders):
        player_position = i

        regret = env.get_regret(bids_profile.to(device), player_position, agents[i].valuations,
                                bids_profile.to(device)[:,i,:], bids_i.squeeze().to(device))
        assert torch.allclose(regret.mean(),expected_regret[i,0], atol = 0.001), "Wrong avg regret calculation"
        assert torch.allclose(regret.max(),expected_regret[i,1], atol = 0.001), "Wrong max regret calculation"

