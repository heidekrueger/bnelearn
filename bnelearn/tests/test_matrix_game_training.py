import os
import sys
import torch
from bnelearn.strategy import MatrixGameStrategy
from bnelearn.bidder import Bidder, Player, MatrixGamePlayer
from bnelearn.mechanism import MatrixGame, PrisonersDilemma, BattleOfTheSexes, MatchingPennies
from bnelearn.optimizer import ES
from bnelearn.environment import Environment, AuctionEnvironment, MatrixGameEnvironment

import numpy as np


"""Shared objects between tests"""
cuda = torch.cuda.is_available()
device = 'cuda' if cuda else 'cpu'

def test_train_prisoners_dilemma():
    """
    trains an instance of prisoners dilemma - using an anonymous AuctionEnvironment.
    Expected behavior: training works and converges to equilibrium.
    """

    n_players = 2
    batch_size = 64
    input_length = 1

    epoch = 25
    learning_rate =1
    lr_decay = False
    lr_decay_every = 1000
    lr_decay_factor = 0.8

    sigma = 5 #ES noise parameter
    n_perturbations = 8

    def strat_to_player(strategy, batch_size, player_position=None):
        return MatrixGamePlayer(
            strategy,
            batch_size = batch_size,
            n_players=n_players,
            player_position=player_position
            )

    model = MatrixGameStrategy(n_actions = 2).to(device)
    game = PrisonersDilemma()
    env = AuctionEnvironment(game, 
                    agents=[],
                    max_env_size =1,
                    n_players=n_players,
                    batch_size=batch_size,
                    strategy_to_bidder_closure=strat_to_player)
    
    optimizer = ES(
        model=model, 
        environment = env, 
        lr = learning_rate, 
        sigma=sigma, 
        n_perturbations=n_perturbations
        )

    torch.cuda.empty_cache()
    for e in range(epoch+1):    
        
        # lr decay?
        if lr_decay and e % lr_decay_every == 0 and e > 0:
            learning_rate = learning_rate * lr_decay_factor
            for param_group in optimizer.param_groups:
                param_group['lr'] = learning_rate
            
        # always: do optimizer step
        utility = -optimizer.step()
        #print(list(model.named_parameters()))
        print((e, utility))

    torch.cuda.empty_cache()

    player = strat_to_player(model, 100)

    assert player.get_action().float().mean().item() > .99, \
                "Player should have learnt to play action 1 ('defect')"
    
def test_matrix_game_environment_training():
    """
    Tests training in an assymetric Matrix Game using the MatrixGameEnvironment
    with unique players.
    """

    # Experiment setup
    n_players = 2

    ## Environment settings
    #training batch size
    batch_size = 5
    input_length = 1

    # optimization params
    epoch = 25
    learning_rate = 1
    lr_decay = False
    lr_decay_every = 1000
    lr_decay_factor = 0.8

    sigma = 5 #ES noise parameter
    n_perturbations = 8

    # Wrapper transforming a strategy to bidder, used by the optimizer
    # this is a dummy, valuation doesn't matter
    def strat_to_player(strategy, batch_size, player_position=None):
        return MatrixGamePlayer(strategy, batch_size = batch_size, n_players=2, player_position=player_position)

    # following game has NE at action profile (0,1)
    # i.e. rowPlayer: Top, colPlayer: Right,
    # resulting in outcome of (3,1)
    game = TwoByTwoBimatrixGame(
        outcomes =torch.tensor([[[2, 0],[3, 1]], [[ 4, 0],[2,2]]]),
        cuda = cuda
    )

    model1 = MatrixGameStrategy(n_actions=2).to(device)
    model2 = MatrixGameStrategy(n_actions=2).to(device)

    env = MatrixGameEnvironment(game, agents=[model1, model2],
                 n_players=n_players,
                 batch_size=batch_size,
                 strategy_to_player_closure=strat_to_player
                 )
    
    optimizer1 = ES(
        model=model1, environment = env,  env_type = 'fixed',
        lr = learning_rate,
        sigma=sigma, n_perturbations=n_perturbations, player_position=0
        )
    optimizer2 = ES(
        model=model2, environment = env, env_type = 'fixed',
        lr = learning_rate, 
        sigma=sigma, n_perturbations=n_perturbations, player_position=1
        )
    
    ## Training ---
    torch.cuda.empty_cache()

    for e in range(epoch+1):        
        # lr decay?
        if lr_decay and e % lr_decay_every == 0 and e > 0:
            for optimizer in [optimizer1, optimizer2]:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = learning_rate

        # always: do optimizer step
        utility1 = -optimizer1.step()        
        utility2 = -optimizer2.step()
        print((e, utility1, utility2))
            
    torch.cuda.empty_cache()

