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
from torch.cuda import device
from bnelearn.mechanism import LLLLGGAuction, FirstPriceSealedBidAuction
from bnelearn.mechanism.auctions_multiunit import FPSBSplitAwardAuction
from bnelearn.strategy import TruthfulStrategy, ClosureStrategy
import bnelearn.util.metrics as metrics
from bnelearn.bidder import Bidder, ReverseBidder
#from bnelearn.experiment.multi_unit_experiment import _optimal_bid_splitaward2x2_1
from bnelearn.environment import AuctionEnvironment
import bnelearn.valuation_sampler as samplers

u_lo = 0
u_hi = 1


## Stefan TODO: @Nils, this was already commented out, please fix
# @pytest.mark.parametrize("rule, mechanism, bid_profile, bids_i, expected_util_loss",
#                          testdata_ex_interim, ids=ids_ex_interim)
# def test_ex_interim_util_loss_estimator_truthful(rule, mechanism, bid_profile, bids_i, expected_util_loss):
#     """Run correctness test for a given LLLLGG rule"""

#     device = 'cuda' if torch.cuda.is_available() else 'cpu'

#     batch_size, n_bidders, n_items = bid_profile.shape

#     agents = [None] * n_bidders
#     for i in range(n_bidders):
#         agents[i] = Bidder.uniform(0, 1, strategy=TruthfulStrategy(), n_items=n_items,
#                                    player_position=i, batch_size=batch_size,
#                                    enable_action_caching=True)
#         agents[i].valuations = bid_profile[:, i, :].to(device)
#         agents[i]._valuations_changed = True

#     env = AuctionEnvironment(
#         mechanism = mechanism,
#         agents = agents,
#         batch_size = batch_size,
#         n_players = n_bidders
#     )

#     opponent_batch_size = 2**10
#     grid_size = 2**10

#     for i in range(n_bidders):
#         # TODO current problem: cannot redraw opponents valuations as their batch
#         #      size is too small: 1.
#         pass
#         # util_loss, best_respone = metrics.ex_interim_util_loss(env, i, batch_size, grid_size, opponent_batch_size)
#         # assert torch.allclose(util_loss.mean(), expected_util_loss[i, 0], atol = 1), \
#         #     "Unexpected avg util_loss {}".format(util_loss.mean() - expected_util_loss[i, 0])
#         # assert torch.allclose(util_loss.max(), expected_util_loss[i, 1], atol = 1), \
#         #     "Unexpected max util_loss {}".format(util_loss.max() - expected_util_loss[i, 1])

def test_ex_interim_util_loss_estimator_fpsb_bne():
    """Test the util_loss in BNE of fpsb. - ex interim util_loss should be close to zero"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    n_players = 3
    grid_size = 2**10
    batch_size = 2**10
    opponent_batch_size = 2**10
    valuation_size = 1
    risk = 1

    # the agent that we will test the estimator for
    player_position = 0


    mechanism = FirstPriceSealedBidAuction()

    def optimal_bid(observation):
        return u_lo + (observation - u_lo) * (n_players - 1) / (n_players - 1.0 + risk)

    strat = ClosureStrategy(optimal_bid)

    sampler = samplers.UniformSymmetricIPVSampler(
        u_lo,u_hi, n_players, valuation_size, batch_size, device)

    agents = [
        Bidder(strat, player_position=i, batch_size=batch_size)
        for i in range(n_players)
    ]

    env = AuctionEnvironment(
        mechanism = mechanism,
        agents = agents,
        valuation_observation_sampler=sampler,
        batch_size = batch_size,
        n_players = n_players
    )

    valuations, observations = sampler.draw_profiles(batch_size, device)
    player_observations = observations[:,player_position, :]
    # assert first player has (near) zero util_loss
    util_loss, best_response = metrics.ex_interim_util_loss(env, player_position,
                                             player_observations, grid_size,
                                             opponent_batch_size)

    mean_util_loss = util_loss.mean()
    max_util_loss = util_loss.max()

    assert mean_util_loss < 0.02, "Util_loss {} in BNE should be (close to) zero!".format(util_loss.mean())
    assert max_util_loss < 0.05, "Util_loss {} in BNE should be (close to) zero!".format(util_loss.max())

## TODO Stefan: @Nils: this test needs multi-unit sampling
def test_ex_interim_util_loss_estimator_splitaward_bne():
    """Test the util_loss in BNE of fpsb split-award auction. - ex interim util_loss should be close to zero"""
    n_players = 2
    grid_size = 2**5
    batch_size = 2**9
    n_items = 2

    pytest.skip("Multi-Unit Not yet implemented!")

    # class SpltAwardConfig:
    #     """Data class for split-award setting"""
    #     u_lo = [1, 1]
    #     u_hi = [1.4, 1.4]
    #     efficiency_parameter = .3
    # config = SpltAwardConfig()

    # mechanism = FPSBSplitAwardAuction()
    # strat = ClosureStrategy(_optimal_bid_splitaward2x2_1(config))

    # agents = [
    #     ReverseBidder.uniform(config.u_lo[0], config.u_hi[0], strat,
    #                           n_items=n_items, efficiency_parameter=config.efficiency_parameter,
    #                           player_position=i, batch_size=batch_size)
    #     for i in range(n_players)
    # ]

    # env = AuctionEnvironment(
    #     mechanism = mechanism,
    #     agents = agents,
    #     batch_size = batch_size,
    #     n_players = n_players
    # )

    # # assert first player has (near) zero util_loss
    # util_loss, best_response = metrics.ex_interim_util_loss(env, 0, batch_size, grid_size)

    # mean_util_loss = util_loss.mean()
    # max_util_loss = util_loss.max()

    # assert mean_util_loss < 0.02, "util_loss in BNE should be (close to) zero " \
    #     + "but is {}!".format(mean_util_loss)
    # assert max_util_loss < 0.07, "util_loss in BNE should be (close to) zero " \
    #     + "but is {}!".format(max_util_loss)
