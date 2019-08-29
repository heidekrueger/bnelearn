"""Test whether agent training in simple matrix games works."""

import torch

from bnelearn.bidder import MatrixGamePlayer
from bnelearn.environment import MatrixGameEnvironment
from bnelearn.mechanism import MatrixGame, PrisonersDilemma
from bnelearn.optimizer import ES
from bnelearn.strategy import MatrixGameStrategy

#Shared objects between tests ##################
cuda = torch.cuda.is_available()
device = 'cuda' if cuda else 'cpu'

def test_prisoners_dilemma_training_shared_model():
    """
    Tests training in MatrixGameEnvironment with a shared model between players.
    """

    # Experiment setup
    n_players = 2

    ## Environment settings
    batch_size = 100

    # optimization params
    epoch = 25
    learning_rate = 1.
    lr_decay = False
    lr_decay_every = 1000
    lr_decay_factor = 0.8

    sigma = 1. #ES noise parameter
    n_perturbations = 8

    # Wrapper transforming a strategy to bidder, used by the optimizer
    # this is a dummy, valuation doesn't matter
    def strat_to_player(strategy, batch_size, player_position=None):
        return MatrixGamePlayer(strategy, batch_size = batch_size, player_position=player_position)

    # following game has NE at action profile (0,1)
    # i.e. rowPlayer: Top, colPlayer: Right,
    # resulting in outcome of (3,1)
    game = PrisonersDilemma(cuda=cuda)
    model = MatrixGameStrategy(n_actions=2).to(device)
    player1 = strat_to_player(model, batch_size, 0)
    player2 = strat_to_player(model, batch_size, 1)
    env = MatrixGameEnvironment(game, agents=[player1, player2],
                                n_players=n_players,
                                batch_size=batch_size,
                                strategy_to_player_closure=strat_to_player
                                )

    optimizer = ES(
        model=model, environment = env,
        lr = learning_rate, sigma=sigma, n_perturbations=n_perturbations,
        strat_to_player_kwargs= {'player_position': 0}
        )

    ## Training ---
    torch.cuda.empty_cache()

    for e in range(epoch+1):
        if lr_decay and e % lr_decay_every == 0 and e > 0:
            for param_group in optimizer.param_groups:
                param_group['lr'] = learning_rate * lr_decay_factor

        # always: do optimizer step
        utility = -optimizer.step()
        #print((e, utility))
        # this only tested whether the loop runs without runtime errors!

    assert player1.get_action().float().mean().item() > .95, \
            "Player1 should have learnt to play action 1 ('defect')"

    assert player2.get_action().float().mean().item() > .95, \
            "Player2 should have learnt to play action 1 ('defect')"


def test_prisoners_dilemma_training_separate_models():
    """
    Tests training in MatrixGameEnvironment
    with unique models for both players.
    """

    n_players = 2
    batch_size = 100
    # optimization params
    epoch = 25
    learning_rate = 1.
    lr_decay = False
    lr_decay_every = 1000
    lr_decay_factor = 0.8

    sigma = 1. #ES noise parameter
    n_perturbations = 8

    # Wrapper transforming a strategy to bidder, used by the optimizer
    # this is a dummy, valuation doesn't matter
    def strat_to_player(strategy, batch_size, player_position=None):
        return MatrixGamePlayer(strategy, batch_size = batch_size, player_position=player_position)

    # following game has NE at action profile (0,1)
    # i.e. rowPlayer: Top, colPlayer: Right,
    # resulting in outcome of (3,1)
    game = PrisonersDilemma(cuda=cuda)

    model1 = MatrixGameStrategy(n_actions=2).to(device)
    model2 = MatrixGameStrategy(n_actions=2).to(device)
    player1 = strat_to_player(model1, batch_size, 0)
    player2 = strat_to_player(model2, batch_size, 1)
    env = MatrixGameEnvironment(game, agents=[player1, player2],
                                n_players=n_players,
                                batch_size=batch_size,
                                strategy_to_player_closure=strat_to_player
                                )

    optimizer1 = ES(
        model=model1, environment = env,
        lr = learning_rate, sigma=sigma, n_perturbations=n_perturbations,
        strat_to_player_kwargs= {'player_position': 0}
        )
    optimizer2 = ES(
        model=model2, environment = env,
        lr = learning_rate, sigma=sigma, n_perturbations=n_perturbations,
        strat_to_player_kwargs= {'player_position': 1}
        )

    ## Training ---
    torch.cuda.empty_cache()

    for e in range(epoch+1):
        if lr_decay and e % lr_decay_every == 0 and e > 0:
            for optimizer in [optimizer1, optimizer2]:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = learning_rate * lr_decay_factor

        # always: do optimizer step
        utility1 = -optimizer1.step()
        utility2 = -optimizer2.step()
        #print((e, utility1, utility2))
        # this only tested whether the loop runs without runtime errors!
    prob_defect_p1 = player1.get_action().float().mean().item()
    prob_defect_p2 = player2.get_action().float().mean().item()
    assert  prob_defect_p1 > .95, \
            "Player1 should play 'defect' with high prob (>95%). Got {}".format(prob_defect_p1)

    assert  prob_defect_p2 > .95, \
            "Player2 should play 'defect' with high prob (>95%). Got {}".format(prob_defect_p2)
