import warnings
import pytest
import torch
from bnelearn.strategy import NeuralNetStrategy
from bnelearn.mechanism import StaticMechanism
from bnelearn.bidder import Bidder
from bnelearn.optimizer import ES, SimpleReinforce

"""Setup shared objects"""

cuda = torch.cuda.is_available()
device = 'cuda' if cuda else 'cpu'
BATCH_SIZE = 2**10
SIZE_HIDDEN_LAYER = 20
input_length = 1

epoch = 1000
LEARNING_RATE = 2e-2
lr_decay = True
lr_decay_every = 2000
lr_decay_factor = 0.8

sigma = .1 #ES noise parameter
n_perturbations = 32

u_lo = 0
u_hi = 10

def strat_to_bidder(strategy, batch_size):
    return Bidder.uniform(u_lo,u_hi, strategy, batch_size = batch_size, n_players=1)

mechanism = StaticMechanism(cuda=cuda)

# TODO: write tests
def test_static_mechanism():
    """Test whether the mechanism for testing the optimizers returns expected results"""
    model = NeuralNetStrategy(input_length, size_hidden_layer = SIZE_HIDDEN_LAYER, requires_grad=False).to(device)
    bidder = strat_to_bidder(model, BATCH_SIZE)

    bidder.valuations.zero_().add_(10)
    bids = bidder.valuations.clone().detach().view(-1,1,1)
    bids.zero_().add_(10)

    allocations, payments = mechanism.run(bids)
    # subset for single player
    allocations = allocations[:,0,:].view(-1, 1)
    payments = payments[:,0].view(-1)
    utilities = bidder.get_utility(allocations=allocations, payments=payments)
    assert torch.isclose(utilities.mean(), torch.tensor(5., device=device), atol=1e-2), "StaticMechanism returned unexpected rewards."

    bids.add_(-5)
    allocations, payments = mechanism.run(bids)
    # subset for single player
    allocations = allocations[:,0,:].view(-1, 1)
    payments = payments[:,0].view(-1)
    utilities = bidder.get_utility(allocations=allocations, payments=payments)
    assert torch.isclose(utilities.mean(), torch.tensor(3.75., device=device), atol=1e-2), "StaticMechanism returned unexpected rewards."

def test_ES_optimizer():
    pass