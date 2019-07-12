""" Test auction learning in symmetric and asymmetric implementations,
    using a 2p-FPSB setup.

    This script tests
    - whether the loop runs without runtime exceptions for a small number of iterations
    - whether the model learnt the appropriate bid for the top-range of valuations
      (this value is expected to be learned _very_ fast as it's most significant
       and as such should always be found (up to a certain range) even in a short amount of time
      )
    - Further, the script tests whether the utility after 200 iterations is in the expected range,
       if it isn't it won't fail but issue a warning (because this might just be due to
        stochasticity as it would take a significantly longer test time / more iterations to make sure.)
"""
import warnings
import torch

from bnelearn.bidder import Bidder
from bnelearn.environment import AuctionEnvironment
from bnelearn.mechanism import FirstPriceSealedBidAuction
from bnelearn.optimizer import ES
from bnelearn.strategy import NeuralNetStrategy

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
size_hidden_layer = 10
epoch = 200
learning_rate = 1e-1
lr_decay = False
baseline = True
momentum = .7
sigma = .02
n_perturbations = 64

mechanism = FirstPriceSealedBidAuction(cuda = True)

def strat_to_bidder(strategy, batch_size, player_position=None): #pylint: disable=redefined-outer-name,missing-docstring
    return Bidder.uniform(
        u_lo, u_hi, strategy,
        batch_size = batch_size,
        player_position=player_position
        )

def test_learning_in_static_environment():
    """Tests the same setting as above (2p FPSB symmetric uniform), but with a
       fixed-environment implementation. (2 named agents with a shared model.)
    """
    model = NeuralNetStrategy(input_length,
                              size_hidden_layer = size_hidden_layer,
                              requires_grad=False,
                              ensure_positive_output=torch.tensor([float(u_hi)])
                             ).to(device)

    bidder1 = strat_to_bidder(model, batch_size,0)
    bidder2 = strat_to_bidder(model, batch_size,1)

    env = AuctionEnvironment(mechanism,
                             agents = [bidder1, bidder2], #static
                             batch_size = batch_size,
                             n_players =n_players,
                             strategy_to_bidder_closure = strat_to_bidder
                             )

    # we'll simply bidder1's model, as it's shard between players.
    optimizer = ES(model=model, environment = env,
                   lr = learning_rate, momentum=momentum,
                   sigma=sigma, n_perturbations=n_perturbations, baseline=baseline,
                   strat_to_bidder_kwargs={'player_position':bidder1.player_position})

    for _ in range(epoch+1):
        utility = -optimizer.step()

    ## no fail until here means the loop ran properly (i.e. no runtime errors)

    ## for upper bound of valuation range, value should be close to optimal.
    bid_at_10 = model(torch.tensor([10.], dtype=torch.float, device = device))
    assert 4 < bid_at_10 < 7, \
        "Model failed to learn optimal bid at upper bound. Found {}, expected range [4,7]".format(bid_at_10)

    # after 200 iterations, utility should be reliably above 0.5
    ## warn if not
    if not 1 < utility < 3:
        warnings.warn('Utility {:.2f} is not in expected range [1,3]!'.format(utility))
