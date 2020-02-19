import pytest
import torch
from bnelearn.mechanism import LLLLGGAuction, FirstPriceSealedBidAuction
from bnelearn.strategy import TruthfulStrategy
from bnelearn.environment import AuctionEnvironment
from bnelearn.bidder import Bidder

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
bids_i = torch.tensor([
        [0 + 0.0001],
        [1/6 + 0.0001],
        [2/6 + 0.0001],
        [3/6 + 0.0001],
        [4/6 + 0.0001],
        [5/6 + 0.0001],
        [1 + 0.0001]
], dtype = torch.float)

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

# 4 Batch, 3 bidders, 1 item
bid_profile_4_3_1 = torch.tensor([
    [
        [0.1],
        [0.3],
        [0.5]],
    [
        [0.6],
        [0.3],
        [0.5]]
], dtype = torch.float)
#TODO: Finish this! 99 only dummies!
regret_4_3_1 = torch.tensor([
    [0.05,0.1],
    [99,99],
    [99,99]
], dtype = torch.float)

# LLLLGG: 1 Batch, 6 bidders,2 items (bid on each, 8 in total)
bid_profile_4_3_2 = torch.tensor([
    [
        [0.9, 0.9],
        [0.2, 0.8],
        [0.3, 0.7],
        [0.4, 0.9],

        [0.95, 0.75],
        [0.35, 0.75]],
], dtype = torch.float)
#TODO: regret,...





# each test input takes form rule: string, bids:torch.tensor,
#                            expected_allocation: torch.tensor, expected_payments: torch.tensor
# Each tuple specified here will then be tested for all implemented solvers.
ids, testdata = zip(*[
    ['fpsb - single-batch, 2 bidders, 1 item', ('fpsb', bid_profile_1_2_1, bids_i, regret_1_2_1)],])


# def strat_to_bidder(strategy, batch_size=batch_size, player_position=None, cache_actions=False):
#         return Bidder.uniform(u_lo, u_hi, strategy, batch_size = batch_size,
#                               player_position=player_position, cache_actions=cache_actions, risk=risk)

@pytest.mark.parametrize("rule, bids_profile, bids_i, expected_regret", testdata, ids=ids)
def test_regret_estimator(rule, bids_profile, bids_i, expected_regret):
    """Run correctness test for a given LLLLGG rule"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    

    batch_size, n_bidders, n_items = bids_profile.shape
    
    mechanism = FirstPriceSealedBidAuction()

    agents = [None] * n_bidders
    for i in range(n_bidders):
        #TODO: Add player position
        agents[i] = Bidder.uniform(0,1,TruthfulStrategy(), player_position = i)
        agents[i].valuations = bids_profile[:,i,:].to(device)
    env = AuctionEnvironment(mechanism, agents, batch_size, n_bidders)
    #regret = torch.tensor(torch.zeros(n_bidders,2), dtype = torch.float, device = device)

    for i in range(n_bidders):
        player_position = i

        regret = env.get_regret(env.agents[player_position], bids_profile.to(device), bids_i.to(device))
        assert regret[0] == expected_regret[i,0], "Wrong avg regret calculation"
        assert regret[1] == expected_regret[i,1], "Wrong avg regret calculation"

