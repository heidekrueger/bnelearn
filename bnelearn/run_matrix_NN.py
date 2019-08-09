import os, sys, time
import torch
from torch.nn.utils import parameters_to_vector, vector_to_parameters

from copy import deepcopy
from bnelearn.strategy import MatrixGameStrategy, Strategy
from bnelearn.bidder import Bidder, Player, MatrixGamePlayer
from bnelearn.mechanism import PrisonersDilemma, BattleOfTheSexes, MatchingPennies, RockPaperScissors, JordanGame
from bnelearn.optimizer import ES
from bnelearn.environment import Environment, AuctionEnvironment, MatrixGameEnvironment

from torch.utils.tensorboard import SummaryWriter
import numpy as np
import matplotlib.pyplot as plt
cuda = torch.cuda.is_available()
device = 'cuda' if cuda else 'cpu'

specific_gpu = 5
if cuda and specific_gpu:
    torch.cuda.set_device(specific_gpu)
########################Setup#############################
# Wrapper transforming a strategy to bidder, used by the optimizer
# this is a dummy, valuation doesn't matter
def strat_to_player(strategy, batch_size, player_position=None):
    return MatrixGamePlayer(strategy, batch_size = batch_size,  player_position=player_position)

def main(args):
    #################################Read parameters#######################
    # args: [game, learner, param, belief, epoch]
    setting = ["None",args[0]]
    weight_normalization = True

    options = {"PD": PrisonersDilemma,
               "MP": MatchingPennies,
               "BoS": BattleOfTheSexes,
               "JG": JordanGame,
              "RPS": RockPaperScissors}

    run_name = time.strftime('NSP_%Y-%m-%d %a %H:%M:%S')
    game_name = setting[1]
    logdir = os.path.join(args[5], 'test_experiments/notebooks', 'matrix', game_name, run_name)
    logdir

    ## Experiment setup
    epoch = args[4]

    ## Environment settings
    #training batch size
    batch_size = args[2][0] #2**10
    input_length = 1

    # optimization params
    # NN Parameters
    learning_rate = args[2][1]
    lr_decay = args[2][2]
    lr_decay_every = args[2][3]
    lr_decay_factor = args[2][4]

    sigma = args[2][5] #ES noise parameter
    n_perturbations = args[2][6]

    game = options[setting[1]]()

    initial_beliefs = None
    if args[3] is not None:
        initial_beliefs = [None] * game.n_players
        for player in range(len(initial_beliefs)):
            actions = [None] * game.outcomes.shape[player]
            for action in range(game.outcomes.shape[player]):
                actions[action] = [args[3][player][action]]
            initial_beliefs[player] = torch.Tensor(actions).to('cpu')
    ##############################End Read parameters#######################
    strats = [None] * game.n_players
    strats_copies = [None] * game.n_players
    players = [None] * game.n_players
    hist_utility = [0] * game.n_players
    hist_probs = [0] * game.n_players
    for i in range(game.n_players):
        strats[i] = MatrixGameStrategy(n_actions=game.outcomes.shape[i],
                                       init_weights = initial_beliefs[i],
                                       init_weight_normalization = weight_normalization).cuda()

    env = MatrixGameEnvironment(game, agents=[deepcopy(a) for a in strats],
                     n_players=game.n_players,
                     batch_size=batch_size,
                     strategy_to_player_closure=strat_to_player
                     )

    for i in range(game.n_players):
        players[i] = ES(model=strats[i], environment = env, lr = learning_rate, sigma=sigma, 
                        n_perturbations=n_perturbations, gradient_normalization=weight_normalization, 
                        strat_to_player_kwargs={'player_position':i})
        print(strats[i].distribution.probs)

    def log_hyperparams(writer):
        writer.add_scalar('hyperparams/batch_size', batch_size)
        writer.add_scalar('hyperparams/learning_rate', learning_rate)
        writer.add_scalar('hyperparams/sigma', sigma)
        writer.add_scalar('hyperparams/n_perturbations', n_perturbations) 
    ############################Training#################################
    with SummaryWriter(log_dir=logdir) as writer:
        torch.cuda.empty_cache()
        log_hyperparams(writer)

        for e in range(epoch+1):    

            # lr decay?
            if lr_decay and e % lr_decay_every == 0 and e > 0:
                learning_rate = learning_rate * lr_decay_factor
                writer.add_scalar('hyperparams/learning_rate', learning_rate, e)
                for optimizer in players:
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = learning_rate

            # always: do optimizer step
            utility = [None] * game.n_players
            for i in range(game.n_players):
                utility[i] = -players[i].step()

            #env.agents = [env._strategy_to_player(agent, batch_size, player_position) if isinstance(agent, Strategy) else agent
            #    for player_position, agent in enumerate([deepcopy(a) for a in strats])]


            for i in range(game.n_players):
                params = parameters_to_vector(strats[i].parameters()).clone()
                vector_to_parameters(params, env.agents[i].strategy.parameters())

            for i in range(game.n_players):
                hist_utility[i] = (e * hist_utility[i] + utility[i])/ (e+1) 
                hist_probs[i] = (e * hist_probs[i] + strats[i].distribution.probs)/ (e+1)

            # Logging
            for i,strat in enumerate(strats):
                # Historical probability for actions
                writer.add_histogram('eval/p{}_action_distribution'.format(i), env.agents[i].get_action().view(-1).cpu().numpy(), e)
                for a in range(len(strat.distribution.probs)-1):
                    # Historical probability for actions
                    writer.add_scalar('eval_player_{}/hist_prob_action_{}'.format(i,a), hist_probs[i][a], e)
                    # Current period actions 
                    writer.add_scalar('eval_player_{}/prob_action_{}'.format(i,a), strat.distribution.probs[a], e)
                    # Expected Utility
                    writer.add_scalar('eval_player_{}/utility'.format(i), utility[i], e)
                    # Expected Historical Utility
                    writer.add_scalar('eval_player_{}/hist_utility'.format(i), hist_utility[i], e)
            if not e % 50: print(e)

if __name__ == '__main__':
    main(sys.argv[1:])

