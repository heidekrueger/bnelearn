import os
import sys
root_path = os.path.abspath(os.path.join('..'))
if root_path not in sys.path:
    sys.path.append(root_path)
    
import torch
from bnelearn.strategy import MatrixGameStrategy
from bnelearn.bidder import Bidder, Player, MatrixGamePlayer
from bnelearn.mechanism import PrisonersDilemma, BattleOfTheSexes, MatchingPennies
from bnelearn.optimizer import ES
from bnelearn.environment import Environment, AuctionEnvironment

from tensorboardX import SummaryWriter
import numpy as np
import matplotlib.pyplot as plt


torch.cuda.is_available()

###
## Experiment setup
n_players = 2

## Environment settings
#training batch size
batch_size = 64
input_length = 1


# optimization params
epoch = 25
learning_rate = 1
lr_decay = False
lr_decay_every = 1000
lr_decay_factor = 0.8

sigma = 5 #ES noise parameter
n_perturbations = 8

name = 16
namestr = './pd/{}'.format(name)

# Wrapper transforming a strategy to bidder, used by the optimizer
# this is a dummy, valuation doesn't matter
def strat_to_player(strategy, batch_size):
    return MatrixGamePlayer(strategy, batch_size = batch_size, n_players=2)

model = MatrixGameStrategy(n_actions=2)
game = PrisonersDilemma()

env = AuctionEnvironment(game, 
                 agents=[],
                 max_env_size =1,
                 n_players=2,
                 batch_size=batch_size,
                 strategy_to_bidder_closure=strat_to_player
                 )

optimizer = ES(model=model, environment = env, lr = learning_rate, sigma=sigma, n_perturbations=n_perturbations)

def log_hyperparams(writer):
    writer.add_scalar('hyperparams/batch_size', batch_size)
    writer.add_scalar('hyperparams/learning_rate', learning_rate)
    writer.add_scalar('hyperparams/sigma', sigma)
    writer.add_scalar('hyperparams/n_perturbations', n_perturbations)


torch.cuda.empty_cache()
writer = SummaryWriter(log_dir=namestr)
log_hyperparams(writer)

for e in range(epoch+1):    
    
    # lr decay?
    if lr_decay and e % lr_decay_every == 0 and e > 0:
        learning_rate = learning_rate * lr_decay_factor
        for param_group in optimizer.param_groups:
            param_group['lr'] = learning_rate
        writer.add_scalar('hyperparams/learning_rate', learning_rate, e)
        
    # always: do optimizer step
    utility = -optimizer.step()
    writer.add_scalar('eval/utility', utility, e) 
    writer.add_scalar('eval/prob_action_0', model.distribution.probs[0], e)    
    #print(list(model.named_parameters()))
    print(e)
        
torch.cuda.empty_cache()
writer.close()