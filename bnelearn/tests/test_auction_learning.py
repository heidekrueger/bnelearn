""" Test learning in assymetric bidder implementation of auctions"""

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

def test_nplayer_playing():
    #pylint: disable=too-many-locals
    n_players = 3
    # valuation distribution
    u_lo =0
    u_hi =10

    def strat_to_bidder(strategy, batch_size, player_position=None):
        return Bidder.uniform(u_lo, u_hi, strategy,
                              player_position = player_position,
                              batch_size = batch_size, n_players=n_players,)

    # settings
    batch_size = 2**5
    input_length = 1
    
    size_hidden_layer = 10

    epoch = 100
    learning_rate = 1e-1
    baseline = True
    momentum = 0.5

    sigma = .02 #ES noise parameter
    n_perturbations = 32

    model = NeuralNetStrategy(input_length,
                          size_hidden_layer = size_hidden_layer,
                          requires_grad=False
                         ).to(device)

    bidder1 = strat_to_bidder(model, batch_size,0)
    bidder2 = strat_to_bidder(model, batch_size,1)
    bidder3 = strat_to_bidder(model, batch_size,2)

    mechanism = FirstPriceSealedBidAuction(cuda = True)
    env = AuctionEnvironment(mechanism,
                agents = [bidder1, bidder2, bidder3], #dynamically built
                max_env_size = 3, #
                batch_size = batch_size,
                n_players =n_players,
                strategy_to_bidder_closure = strat_to_bidder
                )
    optimizer = ES(model=model, environment = env,
                lr = learning_rate, momentum=momentum,
                sigma=sigma, n_perturbations=n_perturbations,
                baseline=baseline, env_type = 'static',
                strat_to_bidder_kwargs={'player_position':bidder1.player_position})

    for e in range(epoch+1):
        utility = -optimizer.step()

        if e % 100 == 0:
            print(utility)

    # passing for now means no error so far.

    # TODO: make sure it's learning correctly, add more tests
