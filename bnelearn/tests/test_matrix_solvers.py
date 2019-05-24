import warnings
import pytest
import torch
from bnelearn.mechanism import PrisonersDilemma, BattleOfTheSexes, BattleOfTheSexes_Mod, MatchingPennies, RockPaperScissors, PaulTestGame, JordanGame
from bnelearn.environment import MatrixGameEnvironment


# Shared objects
cuda = torch.cuda.is_available()
device = 'cuda' if cuda else 'cpu'
batch_size = 64

# Smooth Fictitious Play
def strat_to_player(strategy, batch_size, player_position=None):
    return MatrixGamePlayer(
        strategy,
        batch_size = batch_size,
        n_players=n_players,
        player_position=player_position
        )

def run_smooth_fictitious_play(n_players, game, initial_beliefs=None, iterations=1000):
    game = game
    env = MatrixGameEnvironment(game, 
                agents=[],
                max_env_size=1,
                n_players=n_players,
                batch_size=batch_size,
                strategy_to_bidder_closure=strat_to_player)
    
    return env.solve_with_smooth_fictitious_play(
        dev = device,
        initial_beliefs=initial_beliefs,
        iterations = iterations)
    
def run_fictitious_play(n_players, game, initial_beliefs=None, iterations=1000):
    game = game
    env = MatrixGameEnvironment(game, 
                agents=[],
                max_env_size=1,
                n_players=n_players,
                batch_size=batch_size,
                strategy_to_bidder_closure=strat_to_player)
    
    return env.solve_with_fictitious_play(
        dev = device,
        initial_beliefs=initial_beliefs,
        iterations = iterations)
    
    # Test Smooth Fictitious Play
def test_smooth_fictitious_play_with_PrisonersDilemmy_2x2():    
    sol = run_smooth_fictitious_play(2, PrisonersDilemma(),
                              iterations = 1000)[2][:]
                               
    for i in range(len(sol)-1):
        assert torch.allclose(sol[i],
                          torch.tensor([0,1],dtype = torch.float,device = device),atol = 0.001)
  
def test_smooth_fictitious_play_with_MatchingPennies_2x2():
    sol = run_smooth_fictitious_play(2, MatchingPennies(),
                              iterations = 1000)[2][:]
                               
    for i in range(len(sol)-1):
        assert torch.allclose(sol[i], 
                          torch.tensor([0.5,0.5],dtype = torch.float, device = device),atol = 0.01) 
  
def test_smooth_fictitious_play_with_BattleOfTheSexes_2x2():
    # -> All initial beliefs which are mirrored lead to mixed Nash
    sol = run_smooth_fictitious_play(2, BattleOfTheSexes(),
                                     initial_beliefs = [torch.tensor([0.5,0.5], dtype = torch.float, device = device),
                                                        torch.tensor([0.5,0.5], dtype = torch.float, device = device)],
                                     iterations = 3000)[2][:]
                                                      
    assert torch.allclose(sol[0], torch.tensor([0.6,0.4], dtype = torch.float, device = device),atol = 0.01)
    assert torch.allclose(sol[1], torch.tensor([0.4,0.6], dtype = torch.float, device = device),atol = 0.01)                                                
       
    # Test pure Nash
    sol = run_smooth_fictitious_play(2, BattleOfTheSexes(),
                                     initial_beliefs = [torch.tensor([0.8,0.2], dtype = torch.float, device = device),
                                                        torch.tensor([0.8,0.2], dtype = torch.float, device = device)],
                                     iterations = 3000)[2][:]
    for i in range(len(sol)-1):
        assert torch.allclose(sol[i], 
                          torch.tensor([1,0],dtype = torch.float, device = device),atol = 0.01)                                                  


def test_smooth_fictitious_play_with_PaulTestGame_2x2x2():
    sol = run_smooth_fictitious_play(3, PaulTestGame(),
                                     initial_beliefs = [torch.tensor([0.1,0.9], dtype = torch.float, device = device),
                                                        torch.tensor([0.2,0.8], dtype = torch.float, device = device),
                                                        torch.tensor([0.3,0.7], dtype = torch.float, device = device)],
                                     iterations = 1)[1]
                                                     
    assert torch.allclose(sol[0][-1], torch.tensor([1.98,0.98], dtype = torch.float, device = device),atol = 0.01)      
    assert torch.allclose(sol[1][-1], torch.tensor([1.3,4.89], dtype = torch.float, device = device),atol = 0.01)      
    assert torch.allclose(sol[2][-1], torch.tensor([1.2,4.56], dtype = torch.float, device = device),atol = 0.01)    


    # Test Fictitious Play
def test_fictitious_play_with_PrisonersDilemmy_2x2():    
    sol = run_fictitious_play(2, PrisonersDilemma(),
                              iterations = 100)[2][:]
                              
    for i in range(len(sol)-1):
        assert torch.allclose(sol[i],
                          torch.tensor([0,1],dtype = torch.float,device = device),atol = 0.001)

def test_fictitious_play_with_MatchingPennies_2x2():
    sol = run_fictitious_play(2, MatchingPennies(),
                              iterations = 100)[2][:]
                              
    for i in range(len(sol)-1):
        assert torch.allclose(sol[i], 
                          torch.tensor([0.5,0.5],dtype = torch.float, device = device),atol = 0.01) 
    
def test_fictitious_play_with_BattleOfTheSexes_2x2():
    sol = run_fictitious_play(2, BattleOfTheSexes(),
                              initial_beliefs = [torch.tensor([1,0], dtype = torch.float, device = device), torch.tensor([0,1], dtype = torch.float, device = device)],
                              iterations = 100)[2][:]
                                              
    for i in range(len(sol)-1):
        assert torch.allclose(sol[i],                                       
                           torch.tensor([0,1], dtype = torch.float, device = device),atol = 0.01)
     
    sol = run_fictitious_play(2, BattleOfTheSexes(),
                              initial_beliefs = [torch.tensor([1,0], dtype = torch.float, device = device), torch.tensor([1,0], dtype = torch.float, device = device)],
                              iterations = 100)[2][:], 
    
    for i in range(len(sol)-1):
        assert torch.allclose(sol[i],                                       
                           torch.tensor([1,0], dtype = torch.float, device = device),atol = 0.01)
        
def test_fictitious_play_with_BattleOfTheSexes_Mod_3x2():
    sol = run_fictitious_play(2, BattleOfTheSexes_Mod(),
                              iterations = 100)[2][:]
                                              
    assert torch.allclose(sol[0], torch.tensor([1,0,0], dtype = torch.float, device = device),atol = 0.01)
    assert torch.allclose(sol[1], torch.tensor([1,0], dtype = torch.float, device = device),atol = 0.01)
      
      
def test_fictitious_play_with_PaulTestGame_2x2x2():
    sol = run_fictitious_play(3, PaulTestGame(),
                              initial_beliefs = [torch.tensor([0,1], dtype = torch.float, device = device),
                                                 torch.tensor([0,1], dtype = torch.float, device = device),
                                                 torch.tensor([0,1], dtype = torch.float, device = device)],
                              iterations = 10)[2][:]
                              
    assert torch.allclose(sol[0], torch.tensor([1,0], dtype = torch.float, device = device),atol = 0.01) 
    assert torch.allclose(sol[1], torch.tensor([0,1], dtype = torch.float, device = device),atol = 0.01)
    assert torch.allclose(sol[2], torch.tensor([0,1], dtype = torch.float, device = device),atol = 0.01)
    
   