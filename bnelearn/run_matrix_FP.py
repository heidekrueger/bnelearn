import os
import sys
import time

import torch
from torch.utils.tensorboard import SummaryWriter

from bnelearn.bidder import MatrixGamePlayer
from bnelearn.mechanism import (BattleOfTheSexes, JordanGame, MatchingPennies,
                                PrisonersDilemma, RockPaperScissors)
from bnelearn.strategy import (FictitiousPlayMixedStrategy,
                               FictitiousPlaySmoothStrategy,
                               FictitiousPlayStrategy, MatrixGameStrategy)

global root_path                            

cuda = torch.cuda.is_available()
device = 'cuda' if cuda else 'cpu'

####################################Setup#######################################################

# Wrapper transforming a strategy to bidder, used by the optimizer
# this is a dummy, valuation doesn't matter
def strat_to_player(strategy, batch_size, player_position=None):
    return MatrixGamePlayer(strategy, batch_size = batch_size, player_position=player_position)

def main(args):
    #################################Read parameters#######################
    setting = [args[1],args[0]]
    param_tau = args[2] #[0.00001,10,0.99] #[0.,10,0.9]
    initial_beliefs = None#args[3]# torch.Tensor([[500,500],[500,500]]).to(device) #initial_beliefs = torch.Tensor([[59.5,40.5],[40.5,59.5]]).to(device)

    options = {"FP": FictitiousPlayStrategy,
               "FPS": FictitiousPlaySmoothStrategy,
               "FPM": FictitiousPlayMixedStrategy,
               "PD": PrisonersDilemma,
               "MP": MatchingPennies,
               "BoS": BattleOfTheSexes,
               "JG": JordanGame,
              "RPS": RockPaperScissors}

    run_name = time.strftime('{}_%Y-%m-%d %a %H:%M:%S'.format(setting[0]))
    game_name = setting[1]
    logdir = os.path.join(args[5], 'test_experiments/notebooks', 'matrix', game_name, run_name)
    logdir

    ## Experiment setup
    epoch = args[4]

    ## Environment settings
    #Dummies here
    batch_size = 1
    input_length = 1

    # optimization params
    '''
    All with 10000 epoch
    PD:
    FP [] - Converges to 0 quickly
    FPS [0.5, 10, 0.99]: - Converges to [0.12,0.88]
    FPS [0.0 ,10, 0.90] - Converges to 0 quickly
    FPM [0.5, 10, 0.99] - Converges to [0.27,0.73]
    FPM [0.0, 10, 0.90] - Converges to [0.27,0.73]

    MP:
    FP [] -
    FPS [0.5, 10, 0.99] - Converges to
    FPS [0.0 ,10, 0.90] - Converges to
    FPM [0.5, 10, 0.99] - Converges to
    FPM [0.0, 10, 0.90] - Converges to

    BoS:
    FP -
    FPS - When tau_minimum too small (<0.5) and updates too extreme (<0.9) and too often (<10), FPS escapes MNE and runs to PNE
    FPS [0.5,10,0.99] - Sometimes find MNE with initial_beliefs = torch.Tensor([[600,400],[400,600]]).to(device))
    FPM - Find MNE with FPM: parameters: 0., 10, 0.9 and initial_beliefs = torch.Tensor([[6,4],[4,6]]).to(device))
    FPM - TODO: Check why Equilibrium is [0.55,0.45] here.

    Jordan Game:
    FP - Cycles but historical distribution nicely converges
    FPS [0, 10, 0.9] - Cycles with:
    FPS [0.5,10,0.99] - Converges to [0.5,0.5,0.5]
    FPM - Converges at [0.5,0.5,0.5] TODO: Is this equilibrium for all?
    '''
    # FP Parameters
    tau_minimum = param_tau[0]
    tau_update_interval =  param_tau[1]
    tau_update =  param_tau[2]


    param = "tau_minimum: {} \ntau_update_interval: {} \ntau_update: {}".format(tau_minimum,tau_update_interval,tau_update)

    
    game = options[setting[1]]()

    initial_beliefs = None
    if args[3] is not None:
        initial_beliefs = torch.Tensor(args[3]).to(device)
    ##############################End Read parameters#######################
    # init strategies
    strats = [None] * game.n_players
    players = [None] * game.n_players
    if setting[0] == "FP" or setting[0] == "FPS":
        for i in range(game.n_players):
            strats[i] = options[setting[0]](game = game, initial_beliefs = initial_beliefs)
    else:
        strat0 = options[setting[0]](game = game, initial_beliefs = initial_beliefs)
        for i in range(game.n_players):
            strats[i] = strat0   

    # init players
    for i in range(game.n_players):
        players[i] = strat_to_player(strats[i], batch_size = batch_size, player_position = i)
    ########################################Training################################################
    # Parallel updating
    print(param)
    with SummaryWriter(log_dir=logdir, flush_secs=30) as writer:
        writer.add_text('hyperparams/hyperparameter', param, 0)
        # Log init_beliefs for replicability
        for i,strat in enumerate(strats):
            writer.add_text('strategy {}/init_beliefs'.format(i), str(strat.historical_actions), 0)
        torch.cuda.empty_cache()
        for e in range(epoch):
            actions = [None] * len(players)
            for i,playr in enumerate(players):
                actions[i] = playr.get_action()

            if e%1000 == 0:
                print(actions)

            for _,strategy in enumerate(strats):
                strategy.update_observations(actions)
                strategy.update_beliefs()
                if ((setting[0] == "FPS" or setting[0] == "FPM") and
                    e > 0 and e%tau_update_interval == 0 and strategy.tau >= tau_minimum):
                    strategy.update_tau(tau_update)

            # Logging
            for i,playr in enumerate(players):
                # Historical probability for actions
                writer.add_histogram('eval/p{}_action_distribution'.format(i), actions[i].view(-1).cpu().numpy(), e)
                for a in range(len(playr.strategy.probs[i])-1):
                    # Historical probability for actions
                    writer.add_scalar('eval_player_{}/hist_prob_action_{}'.format(i,a), playr.strategy.probs[i][a], e)
                    # Current period actions 
                    if setting[0] == "FPM":
                        writer.add_scalar('eval_player_{}/prob_action_{}'.format(i,a), actions[i][a], e)
                    else:
                        writer.add_scalar('eval_player_{}/prob_action_{}'.format(i,a), playr.strategy.probs_self[a], e)

                # Expected Utility
                if setting[0] == "FPM":
                    writer.add_scalar('eval_player_{}/exp_utility'.format(i), (playr.strategy.exp_util[i] * actions[i]).sum(), e)
                else:
                    writer.add_scalar('eval_player_{}/exp_utility'.format(i), playr.strategy.exp_util[actions[i]], e)

            # Actual Utility
            _actions = [torch.zeros(strats[0].n_actions[i], dtype = torch.float, device = game.device)
                          for i in range(strats[0].n_players)
                         ]
            for player,action in enumerate(actions):
                if setting[0] == "FPM":
                    _actions[player] = action
                else:
                    _actions[player][action] = 1
            for i in range(len(players)):
                if setting[0] == "FPM":
                    writer.add_scalar('eval_player_{}/utility'.format(i), (game.calculate_expected_action_payoffs(_actions, i) * actions[i]).sum(), e)
                else:
                    writer.add_scalar('eval_player_{}/utility'.format(i), (game.calculate_expected_action_payoffs(_actions, i)[actions[i]]), e)





    '''
    # Sequential updating
    for e in range(epoch):
        for i,playr in enumerate(player):
            actions = [None,None]
            actions[i] = playr.get_action()
            print(actions)
            for _,strategy in enumerate(strat):
                strategy.update(actions)
    '''