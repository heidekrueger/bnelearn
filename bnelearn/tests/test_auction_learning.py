""" Test auction learning in symmetric and asymmetric implementations,
    using a 2p-FPSB setup.

    This script tests
    - whether the loop runs without runtime exceptions for a small number of iterations
    - whether the model learnt the appropriate bid for the top-range of valuations
      (this value is expected to be learned _very_ fast as it's most significant
       and as such should always be found (up to a certain range) even in a short amount of time)
    - Further, the script tests whether the utility after 200 iterations is in the expected range,
       if it isn't it won't fail but issue a warning (because this might just be due to
        stochasticity as it would take a significantly longer test time / more iterations to make sure.)
"""
import warnings
import torch
import torch.nn as nn

from bnelearn.bidder import Bidder
from bnelearn.environment import AuctionEnvironment
from bnelearn.mechanism import FirstPriceSealedBidAuction
from bnelearn.learner import ESPGLearner
from bnelearn.strategy import NeuralNetStrategy
from bnelearn.sampler import UniformSymmetricIPVSampler

cuda = torch.cuda.is_available()
device = 'cuda' if cuda else 'cpu'
specific_gpu = None
if cuda and specific_gpu:
    torch.cuda.set_device(specific_gpu)

n_players = 2
n_items = 1
u_lo = 0
u_hi = 10

batch_size = 2**14
input_length = 1
hidden_nodes = [5,5]
hidden_activations = [nn.SELU(), nn.SELU()]
epoch = 100

learner_hyperparams = {
    'sigma': 0.1,
    'population_size': 64,
    'scale_sigma_by_model_size': False
}

optimizer_type = torch.optim.SGD
optimizer_hyperparams = {
    'lr': 1e-2,
    'momentum': 0.7
}

mechanism = FirstPriceSealedBidAuction(cuda = True)
sampler = UniformSymmetricIPVSampler(u_lo, u_hi, n_players, n_items, batch_size, device)

def strat_to_bidder(strategy, batch_size, player_position=None): #pylint: disable=redefined-outer-name,missing-docstring
    return Bidder(strategy,
        batch_size = batch_size,
        player_position=player_position,
        enable_action_caching=False
        )

def test_learning_in_fpsb_environment():
    """Tests the same setting as above (2p FPSB symmetric uniform), but with a
       fixed-environment implementation. (2 named agents with a shared model.)
    """
    model = NeuralNetStrategy(input_length,
                              hidden_nodes= hidden_nodes,
                              hidden_activations= hidden_activations,
                              ensure_positive_output=torch.tensor([float(u_hi)])
                             ).to(device)

    bidder1 = strat_to_bidder(model, batch_size,0)
    bidder2 = strat_to_bidder(model, batch_size,1)

    env = AuctionEnvironment(mechanism,
                             agents = [bidder1, bidder2],
                             valuation_observation_sampler = sampler,
                             batch_size = batch_size,
                             n_players =n_players,
                             strategy_to_player_closure = strat_to_bidder)
    learner = ESPGLearner(
        model = model,
        environment = env,
        hyperparams = learner_hyperparams,
        optimizer_type = optimizer_type,
        optimizer_hyperparams = optimizer_hyperparams,
        strat_to_player_kwargs={'player_position':bidder1.player_position})

    for _ in range(epoch+1):
        learner.update_strategy()

    utility = env.get_reward(env.agents[0])

    ## no fail until here means the loop ran properly (i.e. no runtime errors)

    ## for upper bound of valuation range, value should be close to optimal.
    bid_at_10 = model(torch.tensor([10.], dtype=torch.float, device = device))
    assert 4 < bid_at_10 < 7, \
        "Model failed to learn optimal bid at upper bound. Found {}, expected range [4,7]".format(bid_at_10)

    # after 200 iterations, utility should be reliably above 0.5
    ## warn if not
    if not 1 < utility < 3:
        warnings.warn('Utility {:.2f} is not in expected range [1,3]!'.format(utility))
