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
    logdir = os.path.join(args[5], 'experiments/notebooks', 'matrix', game_name, run_name)
    logdir

    ## Experiment setup
    epoch = args[4]

    ## Environment settings
    #Dummies here
    batch_size = 1

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
    for i in range(game.n_players):
        strats[i] = options[setting[0]](game = game, initial_beliefs = initial_beliefs)

    # init players
    for i in range(game.n_players):
        players[i] = strat_to_player(strats[i], batch_size = batch_size, player_position = i)
    ########################################Training################################################
    # Tracking
    hist_probs = [torch.Tensor([0] * game.outcomes.shape[i]).to(device) for i in range(game.n_players)]

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
                hist_probs[i] = (e * hist_probs[i] + playr.strategy.probs_self)/(e+1)
                for a in range(len(playr.strategy.probs[i])-1):
                    # Current period actions
                    writer.add_scalar('eval_player_{}/prob_action_{}'.format(i,a), playr.strategy.probs_self[a], e)
                    # Historical probability for actions
                    writer.add_scalar('eval_player_{}/hist_prob_action_{}'.format(i,a), hist_probs[i][a], e)

                # Expected Utility
                if setting[0] == "FPM":
                    writer.add_scalar('eval_player_{}/exp_utility'.format(i), (playr.strategy.exp_util * actions[i]).sum(), e)
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