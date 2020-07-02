
import math
import pytest

import torch

from bnelearn.bidder import MatrixGamePlayer
from bnelearn.environment import MatrixGameEnvironment
from bnelearn.mechanism import (BattleOfTheSexes, MatchingPennies,
                                PrisonersDilemma)
from bnelearn.strategy import (FictitiousPlayMixedStrategy,
                               FictitiousPlaySmoothStrategy,
                               FictitiousPlayStrategy, MatrixGameStrategy)

# Cuda
torch.cuda.is_available()
cuda = torch.cuda.is_available()
device = 'cuda' if cuda else 'cpu'

## Environment settings
# Dummies here
batch_size = 1
input_length = 1
# Params
epochs = 500

# Wrapper transforming a strategy to bidder, used by the optimizer
# this is a dummy, valuation doesn't matter
def strat_to_player(strategy, batch_size, player_position=None):
    return MatrixGamePlayer(strategy, batch_size = batch_size, player_position=player_position)


def init_setup(game, strategy, initial_beliefs):
    strats = [None] * game.n_players
    players = [None] * game.n_players

    # init strategies
    if strategy is FictitiousPlayStrategy or strategy is FictitiousPlaySmoothStrategy:
        for i in range(game.n_players):
            strats[i] = strategy(game = game, initial_beliefs = initial_beliefs)
    else:
        strat0 = strategy(game = game, initial_beliefs = initial_beliefs)
        for i in range(game.n_players):
            strats[i] = strat0
    # init players
    for i in range(game.n_players):
        players[i] = strat_to_player(strats[i], batch_size = batch_size, player_position = i)

    # init environment
    env = MatrixGameEnvironment(game = game,
                                agents = players,
                                n_players = game.n_players,
                                batch_size = batch_size,
                                strategy_to_player_closure = strat_to_player)

    return strats, players, env


def train(epochs, players, strats, tau_update = 1, tau = 0.99, tau_minimum = 0.0001):
    for e in range(epochs):
        actions = [None] * len(players)
        for i,playr in enumerate(players):
            actions[i] = playr.get_action()

        # if e%(epochs/10) == 0:
        #     print(actions)

        for _,strategy in enumerate(strats):
            strategy.update_observations(actions)
            strategy.update_beliefs()
            if ((isinstance(strategy, FictitiousPlaySmoothStrategy) or
                isinstance(strategy, FictitiousPlayMixedStrategy)) and
                e > 0 and e%tau_update == 0 and strategy.tau >= tau_minimum):
                strategy.update_tau(tau)
    return strats, players

############################################# Fictitious Play #################################################
#TODO: Do I even need env. anymore?
def test_FictitiousPlayStrategy_PD():
    strats, players, env = init_setup(PrisonersDilemma(), FictitiousPlayStrategy, None)
    strats, players = train(epochs, players, strats)
    # Testing convergence
    for i,playr in enumerate(players):
        assert math.isclose(playr.get_action(),1, abs_tol=0.1)

def test_FictitiousPlayStrategy_MP():
    strats, players, env = init_setup(MatchingPennies(), FictitiousPlayStrategy, None)
    strats, players = train(epochs, players, strats)
    # Testing convergence
    for i,strat in enumerate(strats):
        for s in strat.probs[i]:
            assert math.isclose(s,0.5, abs_tol = 0.1)

def test_FictitiousPlayStrategy_BoS():
    # Init para for testing
    '''
    initial_beliefs = [None] * 2
    #initial_beliefs[0] = torch.Tensor([[0.8308,0.5793],[0.4064,0.4113]]).to(device)    <- converges
    #initial_beliefs[1] = torch.Tensor([[0.2753,0.4043],[0.1596,0.6916]]).to(device)    <- converges

    #initial_beliefs[0] = torch.Tensor([[0.5892,0.4108],[0.4970,0.5030]]).to(device)    #<- converges
    #initial_beliefs[1] = torch.Tensor([[0.4051,0.5949],[0.1875,0.8125]]).to(device)    #<- converges

    #initial_beliefs[0] = torch.Tensor([[0.59,0.41],[0.49,0.51]]).to(device)    #<- converges
    #initial_beliefs[1] = torch.Tensor([[0.41,0.59],[0.19,0.81]]).to(device)    #<- converges

    #initial_beliefs[0] = torch.Tensor([[0.59,0.41],[0.49,0.51]]).to(device)    #<- converges
    #initial_beliefs[1] = torch.Tensor([[0.59,0.41],[0.49,0.51]]).to(device)    #<- converges

    #initial_beliefs[0] = torch.Tensor([[0.59,0.41],[0.41,0.59]]).to(device)    #<- converges
    #initial_beliefs[1] = torch.Tensor([[0.59,0.41],[0.41,0.59]]).to(device)    #<- converges

    #initial_beliefs[0] = torch.Tensor([[59.5,40.5],[40.5,59.5]]).to(device)    #<- converges
    #initial_beliefs[1] = torch.Tensor([[59.5,40.5],[40.5,59.5]]).to(device)    #<- converges

    #initial_beliefs[0] = torch.Tensor([[59,41],[49,51]]).to(device)    #<- doesn't converge
    #initial_beliefs[1] = torch.Tensor([[41,59],[19,81]]).to(device)    #<- doens't converge

    #initial_beliefs[0] = torch.Tensor([[0.6,0.4],[0.5,0.5]]).to(device)    #<- doesn't converge
    #initial_beliefs[1] = torch.Tensor([[0.4,0.6],[0.1875,0.8125]]).to(device)    #<- doesn't converge

    #initial_beliefs[0] = torch.Tensor([[0.6,0.4],[0.5,0.5]]).to(device)    #<- doesn't converge
    #initial_beliefs[1] = torch.Tensor([[0.4,0.6],[0.2,0.8]]).to(device)    #<- doesn't converge

    initial_beliefs = torch.Tensor([[60,41],[41,60]]).to(device)   #<- doesn't converge
    #initial_beliefs[1] = torch.Tensor([[60,41],[41,60]]).to(device)   #<- doesn't converge

    # -> It converges if the init is very close to MNE play for at least one player but not exactly!
    # -> My hypotheses: it has to be close to cycle. If it is exact,
    #    it is indifferent and takes a random direction, diverging away.
    # -> $$ The question now is whether we should/have track historical actions with integer!?
    # -> No. In Fudenberg (1999) - Learning and Equilirbium, p. 389
    #    they init FP with (1,sqrt(2)), so obviously use float as well.
    '''
    strats, players, env = init_setup(BattleOfTheSexes(), FictitiousPlayStrategy, None)
    strats, players = train(epochs, players, strats)
    # Testing convergence
    assert (math.isclose(players[0].get_action(), 0, abs_tol=0.1) or
            math.isclose(players[0].get_action(), 1, abs_tol=0.1)), \
            "Player 0's action: {} is neither 0 nor 1".format(players[0].get_action())
######################################################################################################################
############################################# Smooth Fictitious Play #################################################
def test_FictitiousPlaySmoothStrategy_PD():
    strats, players, env = init_setup(PrisonersDilemma(), FictitiousPlaySmoothStrategy, None)
    strats, players = train(epochs, players, strats)
    # Testing convergence
    for i,playr in enumerate(players):
        assert math.isclose(playr.get_action(),1, abs_tol=0.1)

def test_FictitiousPlaySmoothStrategy_MP():
    strats, players, env = init_setup(MatchingPennies(), FictitiousPlaySmoothStrategy, None)
    strats, players = train(epochs, players, strats)
    # Testing convergence
    for i,strat in enumerate(strats):
        for s in strat.probs[i]:
            assert math.isclose(s,0.5, abs_tol = 0.1)

def test_FictitiousPlaySmoothStrategy_BoS():
    # Converge to PN
    strats, players, env = init_setup(BattleOfTheSexes(), FictitiousPlaySmoothStrategy, None)
    strats, players = train(epochs, players, strats)
    # Testing convergence
    assert math.isclose(players[0].get_action(),players[1].get_action(), abs_tol=0.1), \
           "Player 0's action: {} is different than player 1's action: {}".format(
               players[0].get_action(), players[1].get_action())
    assert (math.isclose(players[0].get_action(), 0, abs_tol=0.1) or
            math.isclose(players[0].get_action(), 1, abs_tol=0.1)), \
            "Player 0's action: {} is neither 0 nor 1".format(players[0].get_action())

    # Can't hold converge to MNE!
    # Params
    tau_update =  10
    tau = 0.99
    tau_minimum = 0.5
    initial_beliefs = torch.Tensor([[60,40],[40,60]]).to(device)


    strats, players, env = init_setup(BattleOfTheSexes(), FictitiousPlaySmoothStrategy, initial_beliefs)
    strats, players = train(5000, players, strats, tau_update = tau_update, tau = tau, tau_minimum = tau_minimum)
    # Testing convergence

    #TODO: fix this
    pytest.skip("something is wrong with this test -- it 'passes' when the difference is LARGE")

    assert abs(strats[0].probs[0][0] - 0.6) > 0.1, \
        "Strategy 0's probs: {} is not more than 0.1 different than equilibrium 0.6".format(strats[0].probs[0])

    assert abs(strats[1].probs[1][0] - 0.4) > 0.1, \
        "Strategy 1's probs: {} is not more than 0.1 different than equilibrium 0.4".format(strats[1].probs[1])
#####################################################################################################################
############################################# Mixed Fictitious Play #################################################
def test_FictitiousPlayMixedStrategy_PD():
    strats, players, env = init_setup(PrisonersDilemma(), FictitiousPlayMixedStrategy, None)
    strats, players = train(epochs, players, strats)
    # Testing convergence
    for i,playr in enumerate(players):
        assert (math.isclose(playr.get_action()[1],1, abs_tol=0.1))

def test_FictitiousPlayMixedStrategy_MP():
    strats, players, env = init_setup(MatchingPennies(), FictitiousPlayMixedStrategy, None)
    strats, players = train(epochs, players, strats)
    # Testing convergence
    for i,strat in enumerate(strats):
        for s in strat.probs[i]:
            assert math.isclose(s,0.5, abs_tol = 0.1)

def test_FictitiousPlayMixedStrategy_BoS():
    # Converge to PN
    strats, players, env = init_setup(BattleOfTheSexes(), FictitiousPlayMixedStrategy, None)
    strats, players = train(epochs, players, strats)
    # Testing convergence
    assert math.isclose(players[0].get_action()[0],players[1].get_action()[0], abs_tol=0.1), \
            "Player 0's action: {} is different than player 1's action: {}".format(
            players[0].get_action()[0], players[1].get_action()[0])
    assert math.isclose(players[0].get_action()[1],players[1].get_action()[1], abs_tol=0.1), \
            "Player 0's action: {} is different than player 1's action: {}".format(
            players[0].get_action()[1], players[1].get_action()[1])
    assert (math.isclose(players[0].get_action()[0], 0, abs_tol=0.1) or
            math.isclose(players[0].get_action()[0], 1, abs_tol=0.1)), \
            "Player 0's action: {} is neither 0 nor 1".format(players[0].get_action()[0])

    # Can hold converge to MNE!
    # Params
    tau_update =  10
    tau = 0.99
    tau_minimum = 0.5
    initial_beliefs = torch.Tensor([[60,40],[40,60]]).to(device)

    strats, players, env = init_setup(BattleOfTheSexes(), FictitiousPlayMixedStrategy, initial_beliefs)
    strats, players = train(5000, players, strats, tau_update = tau_update, tau = tau, tau_minimum = tau_minimum)
    # Testing convergence
    assert math.isclose(strats[0].probs[0][0],0.6, abs_tol=0.1), \
        "Strategy 0's probs: {} is different than equilibrium 0.6".format(strats[0].probs[0])

    assert math.isclose(strats[1].probs[1][0],0.4, abs_tol=0.1), \
        "Strategy 1's probs: {} is different than equilibrium 0.4".format(strats[1].probs[1])
#####################################################################################################################
