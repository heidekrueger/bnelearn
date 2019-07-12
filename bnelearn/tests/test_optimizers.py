"""This module tests implemented optimizers in a 'static' environment."""
import warnings
import pytest
import torch
from bnelearn.strategy import NeuralNetStrategy
from bnelearn.mechanism import StaticMechanism
from bnelearn.bidder import Bidder
from bnelearn.optimizer import ES #, SimpleReinforce
from bnelearn.environment import AuctionEnvironment


# Shared objects
cuda = torch.cuda.is_available()
device = 'cuda' if cuda else 'cpu'

SIZE_HIDDEN_LAYER = 20
input_length = 1

u_lo = 0
u_hi = 10

def strat_to_bidder(strategy, batch_size, player_position=0):
    """creates a bidder from a strategy"""
    return Bidder.uniform(u_lo,u_hi, strategy, batch_size = batch_size, player_position=player_position)

mechanism = StaticMechanism(cuda=cuda)

def test_static_mechanism():
    """Test whether the mechanism for testing the optimizers returns expected results"""
    BATCH_SIZE = 2**18

    if BATCH_SIZE < 2**18:
        warnings.warn("Test not run due to low batch size!")
        pytest.skip("Batch size too low to perform this test!")

    model = NeuralNetStrategy(
        input_length,
        size_hidden_layer = SIZE_HIDDEN_LAYER,
        requires_grad=False,
        ensure_positive_output=torch.tensor([float(u_hi)])
        ).to(device)


    bidder = strat_to_bidder(model, BATCH_SIZE)

    bidder.valuations.zero_().add_(10)
    bids = bidder.valuations.clone().detach().view(-1,1,1)
    bids.zero_().add_(10)

    allocations, payments = mechanism.run(bids)
    # subset for single player
    allocations = allocations[:,0,:]
    payments = payments[:,0]
    utilities = bidder.get_utility(allocations=allocations, payments=payments)

    assert torch.isclose(utilities.mean(), torch.tensor(5., device=device), atol=1e-2), \
        "StaticMechanism returned unexpected rewards."

    bids.add_(-5)
    allocations, payments = mechanism.run(bids)
    # subset for single player
    allocations = allocations[:,0,:]
    payments = payments[:,0]
    utilities = bidder.get_utility(allocations=allocations, payments=payments)
    assert torch.isclose(utilities.mean(), torch.tensor(3.75, device=device), atol=3e-2), \
        "StaticMechanism returned unexpected rewards."

def test_ES_optimizer():
    """Tests ES optimizer in static environment.
       This does not test complete convergence but 'running in the right direction'.
    """

    BATCH_SIZE = 2**18
    epoch = 200
    LEARNING_RATE = 1e-1
    lr_decay = True
    lr_decay_every = 150
    lr_decay_factor = 0.3

    sigma = .1 #ES noise parameter
    n_perturbations = 32

    model = NeuralNetStrategy(
        input_length,
        size_hidden_layer = SIZE_HIDDEN_LAYER,
        requires_grad=False,
        ensure_positive_output=torch.tensor([float(u_hi)])
        ).to(device)
    
    bidder = strat_to_bidder(model, BATCH_SIZE, 0)

    #bidder = strat_to_bidder(model, BATCH_SIZE)
    env = AuctionEnvironment(
        mechanism,
        agents = [bidder],
        strategy_to_bidder_closure=strat_to_bidder,
        batch_size = BATCH_SIZE,
        n_players=1
        )

    optimizer = ES(
        model=model,
        environment = env,
        env_type = 'fixed',
        lr = LEARNING_RATE,
        sigma=sigma,
        n_perturbations=n_perturbations
        )

    torch.cuda.empty_cache()

    for e in range(epoch+1):

        # lr decay?
        if lr_decay and e % lr_decay_every == 0 and e > 0:
            LEARNING_RATE = LEARNING_RATE * lr_decay_factor
            for param_group in optimizer.param_groups:
                param_group['lr'] = LEARNING_RATE

        # always: do optimizer step
        utility = -optimizer.step()
    torch.cuda.empty_cache()

    assert utility > 1.4, "optimizer did not learn sufficiently (1.4), got {:.2f}".format(utility)

def test_ES_momentum():
    """Tests ES optimizer in static environment.
       This does not test complete convergence but 'running in the right direction'.
    """

    # skip this test for now because it's stupid and takes too long :-P
    #pytest.skip("skipping because too long...")

    BATCH_SIZE = 2**12
    epoch = 200
    LEARNING_RATE = 1e-1
    MOMENTUM = 0.5
    lr_decay = True
    lr_decay_every = 150
    lr_decay_factor = 0.3

    sigma = .1 #ES noise parameter
    n_perturbations = 32


    model = NeuralNetStrategy(
        input_length,
        size_hidden_layer = SIZE_HIDDEN_LAYER,
        requires_grad=False,
        ensure_positive_output=torch.tensor([float(u_hi)])
        ).to(device)
    
    bidder = strat_to_bidder(model, BATCH_SIZE, 0)

    #bidder = strat_to_bidder(model, BATCH_SIZE)
    env = AuctionEnvironment(
        mechanism,
        agents = [bidder],
        strategy_to_bidder_closure=strat_to_bidder,
        batch_size = BATCH_SIZE,
        n_players=1
        )

    optimizer = ES(
        model=model,
        environment = env,
        env_type = 'fixed',
        lr = LEARNING_RATE,
        momentum = MOMENTUM,
        sigma=sigma,
        n_perturbations=n_perturbations
        )

    torch.cuda.empty_cache()

    for e in range(epoch+1):

        # lr decay?
        if lr_decay and e % lr_decay_every == 0 and e > 0:
            LEARNING_RATE = LEARNING_RATE * lr_decay_factor
            for param_group in optimizer.param_groups:
                param_group['lr'] = LEARNING_RATE

        # always: do optimizer step
        utility = -optimizer.step()

    torch.cuda.empty_cache()

    assert utility > 1.4, "optimizer did not learn the optimum. Utility is {:.2f}".format(utility)
