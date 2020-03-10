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
from bnelearn.strategy import TruthfulStrategy, ClosureStrategy
import bnelearn.util.metrics as metrics
from bnelearn.bidder import Bidder

eps = 0.0001 #TODO Stefan: why + eps?
# bid candidates to be evaluated against
bids_i = torch.linspace(0, 1, steps=7).unsqueeze(0) + eps
bids_i_comb = torch.linspace(0,1, steps=4).unsqueeze(0) + eps

# 1 Batch, 2 bidders, 1 item
## First player wins the item for 0.999

bid_profile_1_2_1 = torch.tensor(
    [[[0.9999], [0]]],
    dtype = torch.float)
# n_bidders x 2 (avg, max)
expected_regret_1_2_1 = torch.tensor(
    [   #mean                                   max
        [bid_profile_1_2_1[0,0,0] - bids_i[0,0], bid_profile_1_2_1[0,0,0] - bids_i[0,0] ],
        [0                                     , 0                                      ]
    ], dtype = torch.float)

# 2 Batch, 3 bidders, 1 item
bid_profile_2_3_1 = torch.tensor(
    [[
        [0.1], [0.3], [0.5]
     ],[
        [0.6], [0.3], [0.5]
    ]], dtype = torch.float)
expected_regret_2_3_1_sixths = torch.tensor(
    [   # mean    #max
        [0.04995, 0.0999],
        [0      , 0     ],
        [0.0833 , 0.1666]#[0.0833 , 0.0833]
    ], dtype = torch.float)

b_i_tenths = torch.linspace(0, 1, steps=11).unsqueeze(0) + eps
expected_regret_2_3_1_tenths = torch.tensor(
    [   # mean    #max
        [0.04995, 0.0999],
        [0      , 0     ],
        [0.09995 , 0.1999]#[0.0833 , 0.0833]
    ], dtype = torch.float)

# LLLLGG: 1 Batch, 6 bidders,2 items (bid on each, 8 in total)
bid_profile_1_6_2 = torch.tensor([[
        [0.011, 0.512],
        [0.021, 0.22],
        [0.031, 0.32],
        [0.041, 0.42],
        [0.89, 0.052],
        [0.061, 0.062]
    ]], dtype = torch.float)

expected_regret_1_6_2 = torch.tensor([
        [0.512 - 1/3 + eps, 0.512 - 1/3 + eps],
        [0.22        + eps, 0.22        + eps],
        [0.32        + eps, 0.32        + eps],
        [0.42  - 1/3 + eps, 0.42  - 1/3 + eps],
        [0,                 0                ],
        [0,                 0                ]
    ], dtype = torch.float)
#TODO: Add one test with other pricing rule (-> and positive utility in agent)




# each test input takes form rule: string, bids:torch.tensor,
#                            expected_allocation: torch.tensor, expected_payments: torch.tensor
# Each tuple specified here will then be tested for all implemented solvers.
ids, testdata = zip(*[
    ['fpsb - 1 batch, 2 bidders, 1 item',
        ('fpsb', FirstPriceSealedBidAuction(), bid_profile_1_2_1, bids_i, expected_regret_1_2_1)],
    ['fpsb - 2 batches, 3 bidders, 1 item, steps of sixths',
        ('fpsb', FirstPriceSealedBidAuction(), bid_profile_2_3_1, bids_i, expected_regret_2_3_1_sixths)],
    ['fpsb - 2 batches, 3 bidders, 1 item, steps of tenths',
        ('fpsb', FirstPriceSealedBidAuction(), bid_profile_2_3_1, b_i_tenths, expected_regret_2_3_1_tenths)],
    ['fpsb - 1 batch, 6 bidders, 2 item',
        ('fpsb', LLLLGGAuction(bid_profile_1_6_2.shape[0]), bid_profile_1_6_2, bids_i_comb, expected_regret_1_6_2)]
    ])


# def strat_to_bidder(strategy, batch_size=batch_size, player_position=None, cache_actions=False):
#         return Bidder.uniform(u_lo, u_hi, strategy, batch_size = batch_size,
#                               player_position=player_position, cache_actions=cache_actions, risk=risk)

@pytest.mark.parametrize("rule, mechanism, bid_profile, bids_i, expected_regret", testdata, ids=ids)
def test_ex_post_regret_estimator_truthful(rule, mechanism, bid_profile, bids_i, expected_regret):
    """Run correctness test for a given LLLLGG rule"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    batch_size, n_bidders, n_items = bid_profile.shape

    agents = [None] * n_bidders
    for i in range(n_bidders):
        #TODO: Add player position
        agents[i] = Bidder.uniform(0,1,TruthfulStrategy(), player_position = i, batch_size = batch_size)
        agents[i].valuations = bid_profile[:,i,:].to(device)

    for i in range(n_bidders):
        regret = metrics.ex_post_regret(mechanism, bid_profile.to(device), agents[i], bids_i.squeeze().to(device))
        assert torch.allclose(regret.mean(), expected_regret[i,0], atol = 0.001), "Unexpected avg regret"
        assert torch.allclose(regret.max(),  expected_regret[i,1], atol = 0.001), "Unexpected max regret"

def test_ex_interim_regret_estimator_fpsb_bne():
    """Test the regret in BNE of fpsb. - ex interim regret should be close to zero"""
    # TODO: currently broken because using ex_post regret.
    n_players = 3
    grid_size = 2**5
    batch_size = 2**12
    n_items = 1
    risk = 1
    if risk != 1:
        raise NotImplementedError("ex-interim regret can't handle this yet!")

    u_lo = 0.0
    u_hi = 1.0

    mechanism = FirstPriceSealedBidAuction()

    def optimal_bid(valuation):
            return u_lo + (valuation - u_lo) * (n_players - 1) / (n_players - 1.0 + risk)

    strat = ClosureStrategy(optimal_bid)

    agents = [
        Bidder.uniform(u_lo, u_hi, strat, player_position=i, batch_size=batch_size)
        for i in range(n_players)
    ]

    grid = torch.linspace(0,1, steps = grid_size).unsqueeze(-1)

    bid_profile = torch.empty(batch_size, n_players, n_items, device = agents[0].valuations.device)
    for i,a in enumerate(agents):
        bid_profile[:,i,:] = a.get_action()
    # assert first player has (near) zero regret
    regret = metrics.ex_interim_regret(mechanism, bid_profile, player_position = 0,
                                       agent_valuation = agents[0].valuations,
                                       agent_bid_actual = agents[0].get_action(),
                                       grid = grid
                                       )
    mean_regret = regret.mean()
    max_regret = regret.max()

    # TODO: mean regret should be close to 0
    assert 1 == 0