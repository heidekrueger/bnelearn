import torch

from bnelearn.environment import MatrixGameEnvironment
from bnelearn.mechanism import (BattleOfTheSexes, BattleOfTheSexes_Mod,
                                JordanGame, MatchingPennies, PaulTestGame,
                                PrisonersDilemma, RockPaperScissors)

# Shared objects
cuda = torch.cuda.is_available()
device = 'cuda' if cuda else 'cpu'
batch_size = 64


def run_fudenberg_fictitious_play(n_players, game, initial_beliefs=None, iterations=1000):
    game = game
    env = MatrixGameEnvironment(game,
                                agents=[],
                                max_env_size=1,
                                n_players=n_players,
                                batch_size=batch_size)

    return env.solve_with_fudenberg_fictitious_play(
        dev = device,
        initial_beliefs=initial_beliefs,
        iterations = iterations
    )

def run_smooth_fictitious_play_known_fct(n_players, game, initial_beliefs=None, iterations=1000):
    game = game
    env = MatrixGameEnvironment(game,
                                agents=[],
                                max_env_size=1,
                                n_players=n_players,
                                batch_size=batch_size
                                )

    return env.solve_with_smooth_fictitious_play_known_fct(
        dev = device,
        initial_beliefs=initial_beliefs,
        iterations = iterations,
    )

def run_fudenberg_smooth_fictitious_play(n_players, game, initial_beliefs=None, iterations=1000):
    game = game
    env = MatrixGameEnvironment(game,
                                agents=[],
                                max_env_size=1,
                                n_players=n_players,
                                batch_size=batch_size
                                )

    return env.solve_with_fudenberg_fictitious_play(
        dev = device,
        initial_beliefs=initial_beliefs,
        iterations = iterations,
        smooth = True
    )

def run_gerding_smooth_fictitious_play(n_players, game, initial_beliefs=None, iterations=1000):
    game = game
    env = MatrixGameEnvironment(game,
                                agents=[],
                                max_env_size=1,
                                n_players=n_players,
                                batch_size=batch_size
                                )

    return env.solve_with_gerding_smooth_fictitious_play(
        dev = device,
        initial_beliefs=initial_beliefs,
        iterations = iterations
    )

def run_gerding_k_smooth_fictitious_play(n_players, game, initial_beliefs=None, iterations=1000):
    game = game
    env = MatrixGameEnvironment(game,
                                agents=[],
                                max_env_size=1,
                                n_players=n_players,
                                batch_size=batch_size)

    return env.solve_with_gerding_k_smooth_fictitious_play(
        dev = device,
        initial_beliefs=initial_beliefs,
        iterations = iterations
    )

# Test smooth Fictitious Play

def test_gerding_smooth_fictitious_play_PrisonersDilemma_2x2():
    strategy = run_gerding_smooth_fictitious_play(2, PrisonersDilemma(cuda = cuda),
                                          iterations = 1000)[2][:]

    for i, strat in enumerate(strategy):
        assert torch.allclose(
            strat,
            torch.tensor([0,1],dtype = torch.float,device = device),
            atol = 0.001
            ), "Invalid strategy for player {}. Expected [0,1], got {}".format(i, strat)

def test_gerding_smooth_fictitious_play_with_MatchingPennies_2x2():
    strategy = run_gerding_smooth_fictitious_play(2, MatchingPennies(cuda = cuda),
                                          iterations = 1000)[2][:]

    for i, strat in enumerate(strategy):
        assert torch.allclose(
            strat,
            torch.tensor([0.5,0.5],device = device),
            atol = 0.03
            ), "Invalid strategy for player {}, expected [.5,.5], got {}".format(i,strat)

def test_gerding_smooth_fictitious_play_BattleOfTheSexes_2x2():
    # -> All initial beliefs which are mirrored lead to mixed Nash
    strategy = run_gerding_smooth_fictitious_play(
        2, BattleOfTheSexes(cuda = cuda),
        initial_beliefs = [torch.tensor([0.5,0.5], device = device),
                           torch.tensor([0.5,0.5], device = device)],
        iterations = 3000)[2][:]

    assert torch.allclose(strategy[0], torch.tensor([0.6,0.4], device = device),atol = 0.01)
    assert torch.allclose(strategy[1], torch.tensor([0.4,0.6], device = device),atol = 0.01)

    # Test pure Nash
    strategy = run_gerding_smooth_fictitious_play(
        2, BattleOfTheSexes(cuda = cuda),
        initial_beliefs = [torch.tensor([0.8,0.2], device = device),
                           torch.tensor([0.8,0.2], device = device)],
        iterations = 3000)[2][:]

    for i, strat in enumerate(strategy):
        assert torch.allclose(strat,
                              torch.tensor([1.,0],device = device),
                              atol = 0.01), \
            "Invalid strategy for player {}. Expected [1,0], got {}".format(i, strat)

def test_gerding_smooth_fictitious_play_PaulTestGame_2x2x2():
    _, _, strategy = run_gerding_smooth_fictitious_play(
        3, PaulTestGame(cuda = cuda),
        initial_beliefs = [torch.tensor([0.,1], device = device),
                           torch.tensor([0.,1], device = device),
                           torch.tensor([0.,1], device = device)],
        iterations = 100)

    assert torch.allclose(strategy[0], torch.tensor([1.,0], device = device),atol = 0.01)
    assert torch.allclose(strategy[1], torch.tensor([0.,1], device = device),atol = 0.01)
    assert torch.allclose(strategy[2], torch.tensor([0.,1], device = device),atol = 0.01)

   # Test k Smooth Fictitious Play

def test_gerding_k_smooth_fictitious_play_with_PrisonersDilemma_2x2():
    strategy = run_gerding_k_smooth_fictitious_play(
        2, PrisonersDilemma(cuda = cuda),
        iterations = 1000)[2][:]

    for i, strat in enumerate(strategy):
        assert torch.allclose(
            strat,
            torch.tensor([0.,1],device = device),
            atol = 0.001
            ), "Invalid strategy for player {}, expected [0,1] found {}".format(i, strat)

def test_gerding_k_smooth_fictitious_play_with_MatchingPennies_2x2():
    strategy = run_gerding_k_smooth_fictitious_play(
        2, MatchingPennies(cuda = cuda),
        iterations = 1000
        )[2][:]

    for i, strat in enumerate(strategy):
        assert torch.allclose(
            strat,
            torch.tensor([0.5,0.5],device = device),
            atol = 0.03
            ), "Invalid strategy for player {}".format(i)

def test_gerding_k_smooth_fictitious_play_with_BattleOfTheSexes_2x2():
    # -> All initial beliefs which are mirrored lead to mixed Nash
    strategy = run_gerding_k_smooth_fictitious_play(
        2,
        BattleOfTheSexes(cuda = cuda),
        initial_beliefs = [torch.tensor([0.5,0.5], device = device),
                           torch.tensor([0.5,0.5], device = device)],
        iterations = 3000
        )[2][:]

    assert torch.allclose(strategy[0], torch.tensor([0.6,0.4], device = device),atol = 0.01)
    assert torch.allclose(strategy[1], torch.tensor([0.4,0.6], device = device),atol = 0.01)

    # Test pure Nash
    _, _, strategy = run_gerding_k_smooth_fictitious_play(
        2,
        BattleOfTheSexes(cuda = cuda),
        initial_beliefs = [torch.tensor([0.8,0.2], device = device),
                           torch.tensor([0.8,0.2], device = device)],
        iterations = 3000
        )

    for i, strat in enumerate(strategy):
        assert torch.allclose(
            strat,
            torch.tensor([1.,0],device = device),
            atol = 0.01
            ), "Invalid strategy for player {}".format(i)

def test_gerding_k_smooth_fictitious_play_with_PaulTestGame_2x2x2():
    _, expected_valuation, _ = run_gerding_k_smooth_fictitious_play(
        3,
        PaulTestGame(cuda = cuda),
        initial_beliefs = [torch.tensor([0.1,0.9], device = device),
                           torch.tensor([0.2,0.8], device = device),
                           torch.tensor([0.3,0.7], device = device)],
        iterations = 1
        )

    assert torch.allclose(
        expected_valuation[0][-1], torch.tensor([1.98,0.98], device = device),
        atol = 0.01)
    assert torch.allclose(
        expected_valuation[1][-1], torch.tensor([1.3,4.89], device = device),
        atol = 0.01)
    assert torch.allclose(expected_valuation[2][-1], torch.tensor([1.2,4.56], device = device),atol = 0.01)

# Test Fudenberg Fictitious Play

def test_fudenberg_fictitious_play_with_PrisonersDilemmy_2x2():
    _, _, strategy = run_fudenberg_fictitious_play(2, PrisonersDilemma(cuda = cuda),
                                   iterations = 1000)

    for i, strat in enumerate(strategy):
        assert torch.allclose(strat,
                              torch.tensor([0.,1.], device = device),
                              atol = 0.1), \
            "Unexpected strategy found for player {}".format(i)

def test_fudenberg_fictitious_play_with_MatchingPennies_2x2():
    strategy = run_fudenberg_fictitious_play(2, MatchingPennies(cuda = cuda),
                                   iterations = 1000)[2][:]

    for i, strat in enumerate(strategy):
        assert torch.allclose(strat,
                              torch.tensor([0.5,0.5], device = device),
                              atol = 0.1), \
            "Unexpected strategy found for player {}".format(i)

def test_fudenberg_fictitious_play_with_BattleOfTheSexes_2x2():
    strategy = run_fudenberg_fictitious_play(
        2,
        BattleOfTheSexes(cuda = cuda),
        initial_beliefs = [
            torch.tensor([1.,0], device = device),
            torch.tensor([0.,1], device = device)
            ],
        iterations = 1000
        )[2][:]

    for i, strat in enumerate(strategy):
        assert torch.allclose(
            strat,
            torch.tensor([1.,0], device = device),
            atol = 0.1
            ), "Invalid strategy for player {}".format(i)

    strategy = run_fudenberg_fictitious_play(
        2,
        BattleOfTheSexes(cuda = cuda),
        initial_beliefs = [
            torch.tensor([1.,0], device = device),
            torch.tensor([1.,0], device = device)],
        iterations = 1000)[2][:]

    for i, strat in enumerate(strategy):
        assert torch.allclose(
            strat,
            torch.tensor([1.,0], device = device),
            atol = 0.1
            ), "Invalid strategy for player {}. Expected [1,0], got {}".format(i, strat)

def test_fudenberg_fictitious_play_with_BattleOfTheSexes_Mod_3x2():
    strategy = run_fudenberg_fictitious_play(2, BattleOfTheSexes_Mod(cuda = cuda),
                                   iterations = 1000
                                  )[2][:]

    assert torch.allclose(strategy[0], torch.tensor([1.,0,0], device = device),atol = 0.1)
    assert torch.allclose(strategy[1], torch.tensor([1.,0], device = device),atol = 0.1)

def test_fudenberg_fictitious_play_with_PaulTestGame_2x2x2():
    strategy = run_fudenberg_fictitious_play(
        3, PaulTestGame(cuda = cuda),
        initial_beliefs = [torch.tensor([0.,1], device = device),
                           torch.tensor([0.,1], device = device),
                           torch.tensor([0.,1], device = device)],
        iterations = 100
        )[2][:]

    assert torch.allclose(strategy[0], torch.tensor([1.,0], device = device),atol = 0.1)
    assert torch.allclose(strategy[1], torch.tensor([0.,1], device = device),atol = 0.1)
    assert torch.allclose(strategy[2], torch.tensor([0.,1], device = device),atol = 0.1)

# Test fudenberg smooth fictitious play
def test_fudenberg_smooth_fictitious_play_with_BattleOfTheSexes_2x2():
    
    # Pure nash because both P1 and P2 play action 0 by chance.
    _, _, strategy = run_fudenberg_smooth_fictitious_play(
        2,
        BattleOfTheSexes(cuda = cuda),
        initial_beliefs = [torch.tensor([6,4], device = device),
                           torch.tensor([4,6], device = device)],
        iterations = 3000
        )

    assert torch.allclose(strategy[0], 
                          torch.tensor([0,1.], device = device),
                          atol = 0.01), \
                          "Invalid strategy {} for player {}".format(strategy[0],0)
    assert torch.allclose(strategy[1], 
                          torch.tensor([0,1.], device = device),
                          atol = 0.01), \
                          "Invalid strategy {} for player {}".format(strategy[1],1)

def test_fudenberg_smooth_fictitious_play_with_MatchingPennies_2x2():
    # Test starting with mixed Nash
    _, _, strategy = run_fudenberg_smooth_fictitious_play(
        2,
        MatchingPennies(cuda = cuda),
        initial_beliefs = [torch.tensor([1,1], device = device),
                           torch.tensor([1,1], device = device)],
        iterations = 2000
        )

    for i, strat in enumerate(strategy):
        assert torch.allclose(
            strat,
            torch.tensor([0.5,0.5],device = device),
            atol = 0.01
            ), "Invalid strategy for player {}".format(i)
    

# Test smooth fictitious play with known function
def test_smooth_fictitious_play_known_fct_with_MatchingPennies_2x2():
    # Test starting with mixed Nash
    _, _, strategy = run_smooth_fictitious_play_known_fct(
        2,
        MatchingPennies(cuda = cuda),
        initial_beliefs = [torch.tensor([1,1], device = device),
                           torch.tensor([1,1], device = device)],
        iterations = 2000
        )

    for i, strat in enumerate(strategy):
        assert torch.allclose(
            strat,
            torch.tensor([0.5,0.5],device = device),
            atol = 0.01
            ), "Invalid strategy for player {}. Expected [0.5,0.5], got {}".format(i, strat)

def test_smooth_fictitious_play_known_fct_with_BattleOfTheSexes_2x2():
    
    # Pure nash because both P1 and P2 play action 0 by chance.
    _, _, strategy = run_smooth_fictitious_play_known_fct(
        2,
        BattleOfTheSexes(cuda = cuda),
        initial_beliefs = [torch.tensor([6.,4], device = device),
                           torch.tensor([4.,6], device = device)],
        iterations = 3000
        )

    assert torch.allclose(strategy[0], 
                          torch.tensor([0.6,0.4], device = device),
                          atol = 0.01), \
                          "Invalid strategy {} for player {}".format(strategy[0],0)
    assert torch.allclose(strategy[1], 
                          torch.tensor([0.4,0.6], device = device),
                          atol = 0.01), "Invalid strategy {} for player {}".format(strategy[1],1)