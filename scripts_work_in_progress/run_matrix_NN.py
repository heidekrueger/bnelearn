import os, sys, time
import torch
from torch.nn.utils import parameters_to_vector, vector_to_parameters

from copy import deepcopy
from bnelearn.strategy import MatrixGameStrategy, Strategy, FictitiousNeuralPlayStrategy
from bnelearn.bidder import Bidder, Player, MatrixGamePlayer
from bnelearn.mechanism import PrisonersDilemma, BattleOfTheSexes, MatchingPennies, RockPaperScissors, JordanGame
from bnelearn.learner import ESPGLearner
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

    # Set seeds
    torch.manual_seed(args[6])
    torch.cuda.manual_seed(args[6])
    np.random.seed(args[6])


    # FP Parameters
    tau_minimum = 0.5
    tau_update_interval =  10
    tau_update =  0.95

    setting = ["None",args[0]]
    weight_normalization = True

    options = {"PD": PrisonersDilemma,
               "MP": MatchingPennies,
               "BoS": BattleOfTheSexes,
               "JG": JordanGame,
              "RPS": RockPaperScissors}

    run_name = time.strftime('NSP_%Y-%m-%d %a %H:%M:%S')
    game_name = setting[1]
    logdir = os.path.join(args[5], 'experiments', 'matrix', game_name, run_name)
    logdir

    ## Experiment setup
    epoch = args[4]

    ## Environment settings
    #training batch size
    batch_size =  args[2][0]
    input_length = 1
    # optimization params
    # NN Parameters
    hyperparams = {"population_size": args[2][6],
                    "sigma": args[2][5],
                    "scale_sigma_by_model_size": False,
                    "normalize_gradients": False,
                    "baseline": 'mean_reward'}
    #######################################################
    ####################optimizer_type#####################
    # Adam doesn't work well, it starts oscillating (because it's learning is too extreme!?)
    optimizer_hyperparams = {"lr": args[2][1]}
    optimizer_type = torch.optim.SGD
    
    learning_rate = args[2][1]
    lr_decay = args[2][2]
    lr_decay_every = args[2][3]
    lr_decay_factor = args[2][4]

    sigma = args[2][5] #ES noise parameter
    #n_perturbations = args[2][6]

    game = options[setting[1]]()

    initial_beliefs = None
    if args[3] is not None:
        initial_beliefs = [None] * game.n_players
        for player in range(len(initial_beliefs)):
            actions = [None] * game.outcomes.shape[player]
            for action in range(game.outcomes.shape[player]):
                actions[action] = [args[3][player][action]]
            initial_beliefs[player] = torch.Tensor(actions).t().to('cpu')
    ##############################End Read parameters#######################
    strats = [None] * game.n_players
    strats_copies = [None] * game.n_players
    players = [None] * game.n_players
    hist_utility = [0] * game.n_players
    hist_probs = [torch.Tensor([0] * game.outcomes.shape[i]).to(device) for i in range(game.n_players)]
    for i in range(game.n_players):
        strats[i] = FictitiousNeuralPlayStrategy(n_actions=game.outcomes.shape[i],
                                       beliefs = initial_beliefs[i],
                                       init_weight_normalization = weight_normalization).cuda()

    env = MatrixGameEnvironment(game, agents= [deepcopy(a) for a in strats],
                     n_players=game.n_players,
                     batch_size=batch_size,
                     strategy_to_player_closure=strat_to_player
                     )

    for i in range(game.n_players):
        players[i] = ESPGLearner(model=strats[i], environment = env, hyperparams = hyperparams, 
                        optimizer_type = optimizer_type, optimizer_hyperparams = optimizer_hyperparams, strat_to_player_kwargs={'player_position':i})
        print(strats[i].distribution.probs)

    def log_hyperparams(writer):
        writer.add_scalar('hyperparams/batch_size', batch_size)
        writer.add_scalar('hyperparams/learning_rate', learning_rate)
        writer.add_scalar('hyperparams/sigma', sigma)
        writer.add_scalar('hyperparams/n_perturbations', hyperparams["population_size"])
    ############################Training#################################
    with SummaryWriter(log_dir=logdir) as writer:
        torch.cuda.empty_cache()
        log_hyperparams(writer)

        for e in range(epoch+1):
            # lr decay?
            if lr_decay and e % lr_decay_every == 0 and e > 0:
                learning_rate = learning_rate * lr_decay_factor
                writer.add_scalar('hyperparams/learning_rate', learning_rate, e)
                for p in players:
                    p.optimizer.param_groups[0]['lr'] = learning_rate

            # always: do optimizer step
            utility = [None] * game.n_players
            for i in range(game.n_players):
                utility[i] = -players[i].update_strategy_and_evaluate_utility()

            for i in range(game.n_players):
                params = parameters_to_vector(strats[i].parameters()).clone()
                vector_to_parameters(params, env.agents[i].strategy.parameters())
                env.agents[i].strategy.beliefs = strats[i].beliefs.clone()

            for i in range(game.n_players):
                hist_utility[i] = (e * hist_utility[i] + utility[i])/ (e+1)
                hist_probs[i] = (e * hist_probs[i] + strats[i].distribution.probs)/ (e+1)

            # Logging
            for i,strat in enumerate(strats):
                if (False and e > 0 
                    and e%tau_update_interval == 0 
                    and strat.temperature >= tau_minimum):
                        strat.temperature = strat.temperature * tau_update
                        print("updated temperature")

                for a in range(len(strat.distribution.probs)-1):
                    # Historical probability for actions
                    writer.add_scalar('eval_player_{}/hist_prob_action_{}'.format(i,a), hist_probs[i][a], e)
                    # Current period actions
                    writer.add_scalar('eval_player_{}/prob_action_{}'.format(i,a), strat.distribution.probs[a], e)



                    # Expected Utility
                    writer.add_scalar('eval_player_{}/utility'.format(i), utility[i], e)
                    # Expected Historical Utility
                    writer.add_scalar('eval_player_{}/hist_utility'.format(i), hist_utility[i], e)
            
            # Update beliefs
            for i,strat in enumerate(strats):
                strat.beliefs = hist_probs[-i] #strats[-i].distribution.probs

            if not e % 50: 
                print(e)

if __name__ == '__main__':
    main(sys.argv[1:])

