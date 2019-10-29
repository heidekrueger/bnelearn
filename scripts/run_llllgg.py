##############################Imports###################################
import os
import sys
root_path = os.path.join(os.path.expanduser('~'), 'bnelearn')
if root_path not in sys.path:
    sys.path.append(root_path)

import time
from timeit import default_timer as timer
from functools import partial

import torch
import torch.nn as nn
import torch.nn.utils as ut
from torch.optim.optimizer import Optimizer, required

from bnelearn.strategy import NeuralNetStrategy, ClosureStrategy
from bnelearn.bidder import Bidder
from bnelearn.mechanism import LLLLGGAuction, CombinatorialAuction
from bnelearn.learner import ESPGLearner
from bnelearn.environment import AuctionEnvironment

from torch.utils.tensorboard import SummaryWriter
import numpy as np
import matplotlib.pyplot as plt

# 3D plotting
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import LinearLocator, FormatStrFormatter

# Set seeds
torch.manual_seed(2)
torch.cuda.manual_seed(2)
np.random.seed(2)

# set up matplotlib
is_ipython = 'inline' in plt.get_backend()
if is_ipython:
    from IPython import display
plt.rcParams['figure.figsize'] = [10, 7]
    
cuda = torch.cuda.is_available()
device = 'cuda' if cuda else 'cpu'

# Use specific cuda gpu if desired (i.e. for running multiple experiments in parallel)
specific_gpu = 5
if cuda and specific_gpu:
    torch.cuda.set_device(specific_gpu)

print(device)
if cuda: print(torch.cuda.current_device())

##################################Settings##################################
# log in notebook folder
# alternative for shared access to experiments:
#log_root = os.path.abspath('/srv/bnelearn/experiments')
log_root = os.path.abspath('.')
run_comment = 'espg'
save_figure_data_to_disc = False
save_figure_to_disc = True

## Experiment setup
n_players = 6
n_items = 8
# valuation distribution
# both players should have same lower bound
u_lo = 0.0
u0_hi = 1.
u1_hi = 2.
u_his = [u0_hi, u0_hi, u0_hi, u0_hi, u1_hi, u1_hi]

def strat_to_bidder(strategy, batch_size, player_position):
    return Bidder.uniform(u_lo, u_his[player_position], strategy, n_items = 2, player_position=player_position, batch_size = batch_size, cuda = cuda)

## Environment settings
n_threads = 1
model_sharing = True
#training batch size
batch_size = 2**17
eval_batch_size = 2**25
epoch = 7000

# strategy model architecture
input_length = 2
hidden_nodes = [5, 5]
hidden_activations = [nn.SELU(), nn.SELU()]


learner_hyperparams = {
    'population_size':32,
    'sigma': 1.,
    'scale_sigma_by_model_size': True
}

### Optimizer hyperparams
# SGD standards
    #'lr': 1e-3,
    #'momentum': 0.7
# Adam standards:
    # 'lr': 1e-3
    # 'betas': (0.9, 0.999), #coefficients for running avgs of grad and square grad
    # 'eps': 1e-8 , # added to denominator for numeric stability
    # 'weight_decay': 0, #L2-decay
    # 'amsgrad': False #whether to use amsgrad-variant
optimizer_type = torch.optim.Adam
optimizer_hyperparams ={    
    #'lr': 1e-3,
    #'momentum': 0.9
}

# plot and log training options
plot_epoch = 100
plot_points = min(100, batch_size)

plot_xmin = u_lo
plot_xmax = u1_hi
plot_ymin = u_lo
plot_ymax = u1_hi
plot_zmin = u_lo
plot_zmax = u1_hi * 2

############################Setting up the Environment##########################
# for evaluation
# helper constant
c = 1 / (u0_hi - u_lo)**2 - 1 / (u1_hi - u_lo)**2
        
def plot_bid_function(fig, valuations, bids, writer=None, e=None,
                      plot_points=plot_points,
                      save_vectors_to_disc=save_figure_data_to_disc,
                      save_png_to_disc = False):
                      
    # subsample points and plot    
    v_print = [None] * 2
    b_print = [None] * 2
    for k in range(len(valuations)):
        for k2 in range(len(valuations[k][0])):
            if k==0:
                v_print[k2] = valuations[k][:,k2].detach().cpu().numpy()[:plot_points]
                b_print[k2] = bids[k][:,k2].detach().cpu().numpy()[:plot_points]
            else:
                v_print[k2] = np.concatenate((v_print[k2],valuations[k][:,k2].detach().cpu().numpy()[:plot_points]))
                b_print[k2] = np.concatenate((b_print[k2],bids[k][:,k2].detach().cpu().numpy()[:plot_points]))
        
    fig = plt.gcf()
    plt.cla()
    plt.xlim(plot_xmin, plot_xmax)
    plt.ylim(plot_ymin, plot_ymax)
    plt.xlabel('valuation')
    plt.ylabel('bid')
    plt.text(plot_xmin + 1, plot_ymax - 1, 'iteration {}'.format(e))
    plt.plot(v_print[0], b_print[0], 'bo')
    #plt.plot(v_print[0][0], b_print[0][0], 'bo', v_print[1][0], b_print[1][0], 'go', v_print[2][0], b_print[2][0], 'ro', v_print[3][0], b_print[3][0], 'yo', v_print[4][0], b_print[4][0], 'g-', v_print[5][0], b_print[5][0], 'b-')
    #plt.plot(v_print[0][1], b_print[0][1], 'bo', v_print[1][1], b_print[1][1], 'go', v_print[2][1], b_print[2][1], 'ro', v_print[3][1], b_print[3][1], 'yo', v_print[4][1], b_print[4][1], 'g-', v_print[5][1], b_print[5][1], 'b-')
    if is_ipython:
        display.clear_output(wait=True)
    if save_png_to_disc:
        plt.savefig(os.path.join(logdir, 'png', f'_{e:05}_1.png'))
    #display.display(fig)
    plt.show()
    plt.cla()
    plt.plot(v_print[1], b_print[1], 'bo')
    if save_png_to_disc:
        plt.savefig(os.path.join(logdir, 'png', f'_{e:05}_2.png'))
    #display.display(fig)
    plt.show()
    
    if writer:
        writer.add_figure('eval/bid_function', fig, e)  
        

        
def plot_bid_function_3d(writer, e, save_figure_to_disc=False):
    assert input_length == 2, 'Only case of n_items equals 2 can be plotted.'
    #$$$differentiate local and global in model and valuation

    lin_local = torch.linspace(u_lo, u0_hi, plot_points)
    lin_global = torch.linspace(u_lo, u1_hi, plot_points)
    xv = [None] * 2
    yv = [None] * 2
    xv[0], yv[0] = torch.meshgrid([lin_local, lin_local])
    xv[1], yv[1] = torch.meshgrid([lin_global, lin_global])
    valuations = torch.zeros(plot_points**2, len(models), input_length, device=device)
    models_print = [None] * len(models)

    for model_idx in range(len(models)):
        valuations[:,model_idx,0] = xv[model_idx].reshape(plot_points**2)
        valuations[:,model_idx,1] = yv[model_idx].reshape(plot_points**2)
        models_print[model_idx] = models[model_idx].play(valuations[:,model_idx,:])

    fig = plt.figure()
    for model_idx in range(len(models)):
        for input_idx in range(input_length):
            ax = fig.add_subplot(len(models), 2, model_idx*input_length+input_idx+1, projection='3d')
            ax.plot_trisurf(
                xv[model_idx].reshape(plot_points**2).detach().cpu().numpy(),
                yv[model_idx].reshape(plot_points**2).detach().cpu().numpy(),
                models_print[model_idx][:,input_idx].reshape(plot_points**2).detach().cpu().numpy(),
                cmap = 'plasma',
                # linewidth = 0,
                antialiased = True
            )
            # Axis labeling
            if model_idx == 0:
                ax.set_xlim(plot_xmin, plot_xmax-(u1_hi-u0_hi)); ax.set_ylim(plot_ymin, plot_ymax-(u1_hi-u0_hi)); ax.set_zlim(plot_zmin, plot_zmax-(u1_hi-u0_hi))
            else:
                ax.set_xlim(plot_xmin, plot_xmax); ax.set_ylim(plot_ymin, plot_ymax); ax.set_zlim(plot_zmin, plot_zmax)
            ax.set_xlabel('bundle 0 value'); ax.set_ylabel('bundle 1 value')#; ax.set_zlabel('bid')
            ax.zaxis.set_major_locator(LinearLocator(10))
            ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
            ax.set_title('model {} biddings for bundle {}'.format(model_idx, input_idx))
            ax.view_init(20, -135)
    fig.suptitle('iteration {}'.format(e), size=16)
    fig.tight_layout()

    if save_figure_to_disc:
        plt.savefig(os.path.join(logdir, 'png', f'_{e:05}_3d.png'))
    #if writer:
    #    writer.add_figure(log_name + ' eval/plot_3d', fig, e)

    plt.show()


##########################################################################
# initialize models
if model_sharing:
    models = [None] * 2 
else:
    models = [None] * n_players

sample = torch.tensor([[float(u0_hi), float(u0_hi)],
             [float(u0_hi), float(u1_hi)],
             [float(u1_hi), float(u1_hi)]])
    
for k,v in enumerate(models):
    models[k] = NeuralNetStrategy(input_length,                            
                            hidden_nodes = hidden_nodes,
                            hidden_activations = hidden_activations,
                            ensure_positive_output = sample,
                            output_length = 2).to(device)

n_parameters = [sum([p.numel() for model in models for p in model.parameters()]),sum([p.numel() for model in models for p in model.parameters()])]

bidders = [None] * n_players

for k,v in enumerate(bidders):
    if (model_sharing & k < n_players - 2):
        bidders[k] = strat_to_bidder(models[0], batch_size, player_position=k)
    elif model_sharing & k >= n_players - 2:
        bidders[k] = strat_to_bidder(models[1], batch_size, player_position=k)
    else:
        bidders[k] = strat_to_bidder(models[k], batch_size, player_position=k)

mechanism = LLLLGGAuction(batch_size = batch_size, rule = 'vcg', cuda = cuda)
env = AuctionEnvironment(mechanism,
                  agents = bidders,
                  batch_size = batch_size,
                  n_players =n_players,
                  strategy_to_player_closure = strat_to_bidder
                 )

learners = [None] * len(models)
for k,v in enumerate(learners):
        learners[k] = ESPGLearner(model = models[k],
                                environment = env,
                                hyperparams = learner_hyperparams,
                                optimizer_type = optimizer_type,
                                optimizer_hyperparams = optimizer_hyperparams)

###################################################Logging####################################################
print(log_root)

if os.name == 'nt': raise ValueError('The run_name may not contain : on Windows! (change datetime format to fix this)') 
run_name = time.strftime('%Y-%m-%d %a %H:%M')
if run_comment:
    run_name = run_name + ' - ' + str(run_comment)
logdir = os.path.join(log_root, 'LLLLGG', 'asymmetric', 'uniform', str(n_players) + 'p', run_name)
print(logdir)
os.makedirs(logdir, exist_ok=True)
if save_figure_to_disc:
    os.mkdir(os.path.join(logdir, 'png'))

plt.rcParams['figure.figsize'] = [10, 7]

print('Total parameters: ' + str(n_parameters))
if True:
    v = [None] * len(bidders)
    b = [None] * len(bidders)
    for k, bidder in enumerate(bidders):
        bidder.draw_valuations_()
        v[k] = bidder.valuations.squeeze(0)
        b[k] = bidder.get_action().squeeze(0)

    fig = plt.figure()
    plot_bid_function(fig, v, b, writer=None,e=0,plot_points = plot_points, save_png_to_disc = save_figure_to_disc) 
    plot_bid_function_3d(writer=None,e=0,save_figure_to_disc = save_figure_to_disc) 

###################################################Training####################################################

with SummaryWriter(logdir, flush_secs=60) as writer:
    
    overhead_mins = 0
    torch.cuda.empty_cache()
    fig = plt.figure()

    for e in range(epoch+1):
        print(e)

        # always: do optimizer step
        utilities = [None] * len(learners)
        for k,v in enumerate(learners):
            utilities[k] = learners[k].update_strategy_and_evaluate_utility()
        
        #logging 
        start_time = timer()
            
        # plot current function output
        if e%plot_epoch == 0:
            v = [None] * len(bidders)
            b = [None] * len(bidders)
            for k, bidder in enumerate(bidders):
                bidder.draw_valuations_()
                v[k] = bidder.valuations#.squeeze(0)
                b[k] = bidder.get_action()#.squeeze(0)
            fig = plt.figure()

            print(('Epoch: {}: Model utility in learning env:'+'\t{:.5f}'*len(models)).format(e, *utilities))            
    
            plot_bid_function(fig, v, b, writer,e,plot_points = plot_points,
                                  save_png_to_disc=save_figure_to_disc)  
            plot_bid_function_3d(writer=None,e=e,save_figure_to_disc = save_figure_to_disc) 
        
        elapsed = timer() - start_time
        overhead_mins = overhead_mins + elapsed/60
        writer.add_scalar('debug/overhead_mins', overhead_mins, e)