"""Testing correctness of util_loss estimator for a number of settings.
        Estimates the potential benefit of deviating from the current energy, as:
            util_loss(v_i) = Max_(b_i)[ E_(b_(-i))[u(v_i,b_i,b_(-i))] ]
            util_loss_max = Max_(v_i)[ util_loss(v_i) ]
            util_loss_expected = E_(v_i)[ util_loss(v_i) ]
        Input:
            agent: 1
            bid_profile: (batch_size x n_player x n_items)
            bid_i: (bid_size x n_items)
        Output:
            util_loss_max
            util_loss_expected
    bid_i always used as val_i and only using truthful bidding
"""

import pytest
import torch
from bnelearn.mechanism import LLLLGGAuction, FirstPriceSealedBidAuction
from bnelearn.mechanism.auctions_multiunit import FPSBSplitAwardAuction
from bnelearn.strategy import TruthfulStrategy, ClosureStrategy
import bnelearn.util.metrics as metrics
from bnelearn.bidder import Bidder, ReverseBidder
from bnelearn.experiment.multi_unit_experiment import _optimal_bid_splitaward2x2_1
from bnelearn.environment import AuctionEnvironment


eps = 0.0001
# bid candidates to be evaluated against
bids_i = torch.linspace(0, 1, steps=7).unsqueeze(0) + eps
bids_i_comb = torch.linspace(0,1, steps=4).unsqueeze(0) + eps

# 1 Batch, 2 bidders, 1 item
## First player wins the item for 0.999

valuations_1_2_1 = torch.tensor(
    [[[0.9999], [0]]],
    dtype = torch.float)
# n_bidders x 2 (avg, max)
expected_util_loss_1_2_1 = torch.tensor(
    [   #mean                                   max
        [valuations_1_2_1[0,0,0] - bids_i[0,0], valuations_1_2_1[0,0,0] - bids_i[0,0] ],
        [0                                     , 0                                      ]
    ], dtype = torch.float)

# 2 Batch, 3 bidders, 1 item
# TODO, later: add a player that has highest valuation SOMETIMES but with different behavior of opponents across batches!
valuations_2_3_1 = torch.tensor(
    [[
        [0.1], [0.3], [0.5]
     ],[
        [0.6], [0.3], [0.5]
    ]], dtype = torch.float)
expected_ex_post_util_loss_2_3_1_sixths = torch.tensor(
    [   # mean    #max
        [0.04995, 0.0999],
        [0      , 0     ],
        [0.0833 , 0.1666]
    ], dtype = torch.float)

expected_ex_interim_util_loss_2_3_1_sixths = torch.tensor(
    [   # mean    #max
        [0.04995, 0.0999],
        [0      , 0     ],
        [0.0833 , 0.0833]
    ], dtype = torch.float)

b_i_tenths = torch.linspace(0, 1, steps=11).unsqueeze(0) + eps
expected_ex_post_util_loss_2_3_1_tenths = torch.tensor(
    [   # mean    #max
        [0.04995, 0.0999],
        [0      , 0     ],
        [0.09995 , 0.1999]
    ], dtype = torch.float)

# LLLLGG: 1 Batch, 6 bidders,2 items (bid on each, 8 in total)
valuations_1_6_2 = torch.tensor([[
        [0.011, 0.512],#[,*]
        [0.021, 0.22],#[,*]
        [0.031, 0.32],#[,*]
        [0.041, 0.42],#[,*]
        [0.89, 0.052],
        [0.061, 0.062]
    ]], dtype = torch.float)

# same for interim and post
expected_util_loss_1_6_2 = torch.tensor([
        [0.512 - 1/3 + eps, 0.512 - 1/3 + eps],
        [0.22        + eps, 0.22        + eps],
        [0.32        + eps, 0.32        + eps],
        [0.42  - 1/3 + eps, 0.42  - 1/3 + eps],
        [0,                 0                ],
        [0,                 0                ]
    ], dtype = torch.float)
#TODO, later: Add one test with other pricing rule (-> and positive utility in agent)
#TODO, Paul: @Nils add tests for your settings




# each test input takes form rule: string, bids:torch.tensor,
#                            expected_allocation: torch.tensor, expected_payments: torch.tensor
# Each tuple specified here will then be tested for all implemented solvers.
ids_ex_post, testdata_ex_post = zip(*[
    ['fpsb - 1 batch, 2 bidders, 1 item',
        ('first_price', FirstPriceSealedBidAuction(), valuations_1_2_1, bids_i, expected_util_loss_1_2_1)],
    ['fpsb - 2 batches, 3 bidders, 1 item, steps of sixths',
        ('first_price', FirstPriceSealedBidAuction(), valuations_2_3_1, bids_i, expected_ex_post_util_loss_2_3_1_sixths)],
    ['fpsb - 2 batches, 3 bidders, 1 item, steps of tenths',
        ('first_price', FirstPriceSealedBidAuction(), valuations_2_3_1, b_i_tenths, expected_ex_post_util_loss_2_3_1_tenths)],
    ['fpsb - 1 batch, 6 bidders, 2 item',
        ('first_price', LLLLGGAuction(), valuations_1_6_2, bids_i_comb, expected_util_loss_1_6_2)]
    ])

ids_ex_interim, testdata_ex_interim = zip(*[
    ['fpsb - 1 batch, 2 bidders, 1 item',
        ('first_price', FirstPriceSealedBidAuction(), valuations_1_2_1, bids_i, expected_util_loss_1_2_1)],
    ['fpsb - 2 batches, 3 bidders, 1 item, steps of sixths',
        ('fpfirst_pricesb', FirstPriceSealedBidAuction(), valuations_2_3_1, bids_i, expected_ex_interim_util_loss_2_3_1_sixths)],
    ['fpsb - 1 batch, 6 bidders, 2 item',
        ('first_price', LLLLGGAuction(), valuations_1_6_2, bids_i_comb, expected_util_loss_1_6_2)]
    ])

@pytest.mark.parametrize("rule, mechanism, bid_profile, bids_i, expected_util_loss", testdata_ex_post, ids=ids_ex_post)
def test_ex_post_util_loss_estimator_truthful(rule, mechanism, bid_profile, bids_i, expected_util_loss):
    """Run correctness test for a given LLLLGG rule"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    batch_size, n_bidders, n_items = bid_profile.shape

    agents = [None] * n_bidders
    for i in range(n_bidders):
        agents[i] = Bidder.uniform(0,1,TruthfulStrategy(), player_position = i, batch_size = batch_size)
        agents[i].valuations = bid_profile[:,i,:].to(device)

    for i in range(n_bidders):
        util_loss = metrics.ex_post_util_loss(mechanism, bid_profile.to(device), agents[i], bids_i.squeeze().to(device))
        assert torch.allclose(util_loss.mean(), expected_util_loss[i,0], atol = 0.001), "Unexpected avg util_loss"
        assert torch.allclose(util_loss.max(),  expected_util_loss[i,1], atol = 0.001), "Unexpected max util_loss"

@pytest.mark.parametrize("rule, mechanism, bid_profile, bids_i, expected_util_loss",
                         testdata_ex_interim, ids=ids_ex_interim)
def test_ex_interim_util_loss_estimator_truthful(rule, mechanism, bid_profile, bids_i, expected_util_loss):
    """Run correctness test for a given LLLLGG rule"""
    # TODO Nils @Stefan:
    #   (1) bids_i obsolete
    #   (2) do we want utility_actual to be against this hard coded grid or the expectation against the
    #       actual opponenents? (Only the later is supported by `ex_interim_util_loss`.)
    #   (3) ex_interim_util_loss: mean=max for batch_size of 1

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    batch_size, n_bidders, n_items = bid_profile.shape

    agents = [None] * n_bidders
    for i in range(n_bidders):
        agents[i] = Bidder.uniform(0, 1, strategy=TruthfulStrategy(), n_items=n_items,
                                   player_position=i, batch_size=batch_size,
                                   cache_actions=True)
        agents[i].valuations = bid_profile[:, i, :].to(device)
        agents[i]._valuations_changed = True

    env = AuctionEnvironment(
        mechanism = mechanism,
        agents = agents,
        batch_size = batch_size,
        n_players = n_bidders
    )

    opponent_batch_size = 2**10
    grid_size = 2**10

    for i in range(n_bidders):
        # TODO current problem: cannot redraw opponents valuations as their batch
        #      size is too small: 1.
        pass
        # util_loss = metrics.ex_interim_util_loss(env, i, batch_size, grid_size, opponent_batch_size)
        # assert torch.allclose(util_loss.mean(), expected_util_loss[i, 0], atol = 1), \
        #     "Unexpected avg util_loss {}".format(util_loss.mean() - expected_util_loss[i, 0])
        # assert torch.allclose(util_loss.max(), expected_util_loss[i, 1], atol = 1), \
        #     "Unexpected max util_loss {}".format(util_loss.max() - expected_util_loss[i, 1])

def test_ex_interim_util_loss_estimator_fpsb_bne():
    """Test the util_loss in BNE of fpsb. - ex interim util_loss should be close to zero"""
    n_players = 3
    grid_size = 2**10
    batch_size = 2**10
    n_items = 1
    risk = 1

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

    env = AuctionEnvironment(
        mechanism = mechanism,
        agents = agents,
        batch_size = batch_size,
        n_players = n_players
    )

    # assert first player has (near) zero util_loss
    util_loss = metrics.ex_interim_util_loss(env, 0, batch_size, grid_size)

    mean_util_loss = util_loss.mean()
    max_util_loss = util_loss.max()

    assert mean_util_loss < 0.02, "Util_loss {} in BNE should be (close to) zero!".format(util_loss.mean())
    assert max_util_loss < 0.05, "Util_loss {} in BNE should be (close to) zero!".format(util_loss.max())

def test_ex_interim_util_loss_estimator_splitaward_bne():
    """Test the util_loss in BNE of fpsb split-award auction. - ex interim util_loss should be close to zero"""
    n_players = 2
    grid_size = 2**5
    batch_size = 2**9
    n_items = 2

    class SpltAwardConfig:
        """Data class for split-award setting"""
        u_lo = [1, 1]
        u_hi = [1.4, 1.4]
        efficiency_parameter = .3
    config = SpltAwardConfig()

    mechanism = FPSBSplitAwardAuction()
    strat = ClosureStrategy(_optimal_bid_splitaward2x2_1(config))

    agents = [
        ReverseBidder.uniform(config.u_lo[0], config.u_hi[0], strat,
                              n_items=n_items, efficiency_parameter=config.efficiency_parameter,
                              player_position=i, batch_size=batch_size)
        for i in range(n_players)
    ]

    env = AuctionEnvironment(
        mechanism = mechanism,
        agents = agents,
        batch_size = batch_size,
        n_players = n_players
    )

    # assert first player has (near) zero util_loss
    util_loss = metrics.ex_interim_util_loss(env, 0, batch_size, grid_size)

    mean_util_loss = util_loss.mean()
    max_util_loss = util_loss.max()

    assert mean_util_loss < 0.02, "util_loss in BNE should be (close to) zero " \
        + "but is {}!".format(mean_util_loss)
    assert max_util_loss < 0.07, "util_loss in BNE should be (close to) zero " \
        + "but is {}!".format(max_util_loss)
