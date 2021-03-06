"""This module tests implemented learner in a 'static' environment."""
import warnings
import pytest
import torch
import torch.nn as nn
from bnelearn.strategy import NeuralNetStrategy
from bnelearn.mechanism import StaticMechanism, StaticFunctionMechanism
from bnelearn.bidder import Bidder
from bnelearn.environment import AuctionEnvironment
from bnelearn.learner import ESPGLearner, PGLearner, PSOLearner, AESPGLearner
from bnelearn.sampler import UniformSymmetricIPVSampler

# Shared objects
cuda = torch.cuda.is_available()
device = 'cuda' if cuda else 'cpu'

hidden_nodes = [5,5]
hidden_activations = [nn.SELU(), nn.SELU()]
input_length = 1
u_lo = 0.0
u_hi = 10.0

def strat_to_bidder(strategy, batch_size, player_position=0):
    """creates a bidder from a strategy"""
    return Bidder(strategy, player_position, batch_size)
mechanism_auction = StaticMechanism(cuda=cuda)
mechanism_function = StaticFunctionMechanism(cuda=cuda)

def test_static_mechanism():
    """Test whether the mechanism for testing the optimizers returns expected results"""
    BATCH_SIZE = 2**18

    if BATCH_SIZE < 2**18:
        warnings.warn("Test not run due to low batch size!")
        pytest.skip("Batch size too low to perform this test!")

    model = NeuralNetStrategy(
        input_length,
        hidden_nodes=hidden_nodes,
        hidden_activations=hidden_activations,
        ensure_positive_output=torch.tensor([float(u_hi)])
        ).to(device)

    bidder = strat_to_bidder(model, BATCH_SIZE)

    valuations = torch.zeros(BATCH_SIZE, 1, device=device).add_(10)
    bids = valuations.clone().detach().view(-1,1,1)
    bids.zero_().add_(10)

    allocations, payments = mechanism_auction.run(bids)
    # subset for single player
    utilities = bidder.get_utility(
        allocations=allocations[:,0,:], payments=payments[:,0], valuations=valuations)

    assert torch.isclose(utilities.mean(), torch.tensor(5., device=device), atol=1e-2), \
        "StaticMechanism returned unexpected rewards."

    bids.add_(-5)
    allocations, payments = mechanism_auction.run(bids)
    # subset for single player
    utilities = bidder.get_utility(
        allocations=allocations[:,0,:], payments=payments[:,0], valuations=valuations)
    assert torch.isclose(utilities.mean(), torch.tensor(3.75, device=device), atol=3e-2), \
        "StaticMechanism returned unexpected rewards."

def test_ES_learner_SGD():
    """Tests ES PG learner with SGD optimizer in static environment.
       This does not test complete convergence but 'running in the right direction'.
    """
    BATCH_SIZE = 2**16
    epoch = 100

    optimizer_type = torch.optim.SGD
    optimizer_hyperparams = {'lr': 1e-1, 'momentum': 0.3}
    learner_hyperparams = {'sigma': .1, 'population_size': 32, 'scale_sigma_by_model_size': False}

    model = NeuralNetStrategy(
        input_length,
        hidden_nodes=hidden_nodes,
        hidden_activations=hidden_activations,
        ensure_positive_output=torch.tensor([float(u_hi)])
        ).to(device)

    bidder = strat_to_bidder(model, BATCH_SIZE, 0)


    env = AuctionEnvironment(
        mechanism_auction, agents=[bidder],
        valuation_observation_sampler=UniformSymmetricIPVSampler(u_lo, u_hi, 1, 1, BATCH_SIZE, device),
        strategy_to_player_closure=strat_to_bidder,
        batch_size = BATCH_SIZE, n_players=1)
    
    env.draw_valuations()

    learner = ESPGLearner(
        model=model,
        environment=env,
        hyperparams=learner_hyperparams,
        optimizer_type=optimizer_type,
        optimizer_hyperparams=optimizer_hyperparams
    )

    for e in range(epoch+1):
        utility = learner.update_strategy_and_evaluate_utility()

    assert utility > 1.34, "optimizer did not learn sufficiently (1.34), got {:.2f}".format(utility)

def test_PG_learner_SGD():
    """Tests the standard policy gradient learner in static env.
    This does not test complete convergence but 'running in the right direction'.
    """
    BATCH_SIZE = 2**12
    epoch = 1000

    optimizer_type = torch.optim.SGD
    optimizer_hyperparams = {'lr': 1e-3, 'momentum': 0.2}
    learner_hyperparams = {}

    model = NeuralNetStrategy(
        input_length,
        hidden_nodes=hidden_nodes,
        hidden_activations=hidden_activations,
        ensure_positive_output=torch.tensor([float(u_hi)])
        ).to(device)

    bidder = strat_to_bidder(model, BATCH_SIZE, 0)
    env = AuctionEnvironment(
        mechanism_function, agents=[bidder],
        valuation_observation_sampler=UniformSymmetricIPVSampler(u_lo, u_hi, 1, 1, BATCH_SIZE, device),
        strategy_to_player_closure=strat_to_bidder,
        batch_size = BATCH_SIZE, n_players=1)

    learner = PGLearner(
        model=model,
        environment=env,
        hyperparams=learner_hyperparams,
        optimizer_type=optimizer_type,
        optimizer_hyperparams=optimizer_hyperparams
    )

    for _ in range(epoch+1):
        utility = learner.update_strategy_and_evaluate_utility()
    print(utility)
    assert utility > 2.3, "optimizer did not learn sufficiently (2.2), got {:.2f}".format(utility)

def test_PSO_learner_SGD():
    """Tests ES PG learner with SGD optimizer in static environment.
       This does not test complete convergence but 'running in the right direction'.
    """
    BATCH_SIZE = 2**16
    epoch = 100

    optimizer_type = torch.optim.SGD
    optimizer_hyperparams = {'lr': 1e-1, 'momentum': 0.3}

    model = NeuralNetStrategy(
        input_length,
        hidden_nodes =hidden_nodes,
        hidden_activations=hidden_activations,
        ensure_positive_output=torch.tensor([float(u_hi)])
        ).to(device)

    bidder = strat_to_bidder(model, BATCH_SIZE, 0)

    env = AuctionEnvironment(
        mechanism_auction, agents = [bidder],
        valuation_observation_sampler= UniformSymmetricIPVSampler(u_lo, u_hi, 1, 1, BATCH_SIZE, device),
        strategy_to_player_closure=strat_to_bidder,
        batch_size = BATCH_SIZE, n_players=1)
    
    env.draw_valuations()

    hyperparams = {
        'swarm_size': 32,
        'topology': 'von_neumann',
        'reevaluation_frequency': 10,
        'inertia_weight': .5,
        'cognition': .8,
        'social': .8,
    }

    learner = PSOLearner(
        model=model,
        hyperparams=hyperparams,
        environment=env,
        optimizer_type=optimizer_type,
        optimizer_hyperparams=optimizer_hyperparams
    )

    for e in range(epoch+1):
        utility = learner.update_strategy_and_evaluate_utility()

    assert utility > 1.34, "optimizer did not learn sufficiently (1.34), got {:.2f}".format(utility)

@pytest.mark.xfail(reason="AESP is still experimental.")
def test_AESPG_learner_SGD():
    """Tests the standard policy gradient learner in static env.
    This does not test complete convergence but 'running in the right direction'.
    """
    BATCH_SIZE = 2**10
    epoch = 100

    optimizer_type = torch.optim.SGD
    optimizer_hyperparams = {'lr': 1e-3, 'momentum': 0.5}
    learner_hyperparams = {'sigma': .5, 'population_size': 32}

    model = NeuralNetStrategy(
        input_length,
        hidden_nodes =hidden_nodes,
        hidden_activations=hidden_activations,
        ensure_positive_output=torch.tensor([float(u_hi)])
        ).to(device)

    bidder = strat_to_bidder(model, BATCH_SIZE, 0)
    env = AuctionEnvironment(
        mechanism_auction, agents = [bidder],
        strategy_to_player_closure=strat_to_bidder,
        batch_size = BATCH_SIZE, n_players=1)

    learner = AESPGLearner(
        model = model,
        environment=env,
        hyperparams=learner_hyperparams,
        optimizer_type=optimizer_type,
        optimizer_hyperparams=optimizer_hyperparams
    )

    for _ in range(epoch+1):
        utility = learner.update_strategy_and_evaluate_utility()

    assert utility > 1.4, "optimizer did not learn sufficiently (1.4), got {:.2f}".format(utility)
