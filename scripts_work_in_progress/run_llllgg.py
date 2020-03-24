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
from torch.optim.optimizer import required

from bnelearn.strategy import NeuralNetStrategy, ClosureStrategy
from bnelearn.bidder import Bidder
from bnelearn.mechanism import LLLLGGAuction, CombinatorialAuction
from bnelearn.learner import ESPGLearner
from bnelearn.environment import AuctionEnvironment
import bnelearn.util.metrics as metrics

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
specific_gpu = 2
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
save_figure_data_to_disc = True
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
core_solver = 'no_core' #no_core #gurobi
pricing_rule =  'first_price'#'first_price'#nearest-vcg'
model_sharing = True
#training batch size (2**17 - 2**18 for vcg)
batch_size = 2**11
regret_bid_size = 2**7
eval_batch_size = 2**25
epoch = 100000


# strategy model architecture
input_length = 2
hidden_nodes = [16, 16]#[8,8, 8]#, 8,8, 8, 8] #8, 8,8, 8, 8]#, 128]#3, 5, 2]
hidden_activations = [nn.SELU(),nn.SELU()]#,nn.SELU()]#, nn.SELU(),nn.SELU(), nn.SELU(), nn.SELU()] #nn.Tanh(), nn.Tanh(),


learner_hyperparams = {
    'population_size':128,
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
    'lr': 1e-2,
    #'momentum': 0.6
}

# plot and log training options
plot_epoch = 10
write_epoch = 1000
plot_points = 100 #min(100, batch_size)
# For verification writing
write_points = 200 #min(200, batch_size)


plot_xmin = u_lo
plot_xmax = u1_hi
plot_ymin = u_lo
plot_ymax = u1_hi
plot_zmin = u_lo
plot_zmax = u1_hi * 1

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
    if writer:
        writer.add_figure('eval/bid_function_2d_0', fig, e)  

    plt.cla()
    plt.plot(v_print[1], b_print[1], 'bo')
    if save_png_to_disc:
        plt.savefig(os.path.join(logdir, 'png', f'_{e:05}_2.png'))
    #display.display(fig)
    plt.show()
    if writer:
        writer.add_figure('eval/bid_function_2d_1', fig, e)  
        

        
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
    models_print_wf = [None] * len(models)

    for model_idx in range(len(models)):
        if len(models) > 2:
            valuations[:,model_idx,0] = xv[0].reshape(plot_points**2)
            valuations[:,model_idx,1] = yv[0].reshape(plot_points**2)
            if model_idx>3:
                valuations[:,model_idx,0] = xv[1].reshape(plot_points**2)
                valuations[:,model_idx,1] = yv[1].reshape(plot_points**2)    
    for model_idx in range(len(models)):
        if len(models) > 2:
            valuations[:,model_idx,0] = xv[0].reshape(plot_points**2)
            valuations[:,model_idx,1] = yv[0].reshape(plot_points**2)
            if model_idx>3:
                valuations[:,model_idx,0] = xv[1].reshape(plot_points**2)
                valuations[:,model_idx,1] = yv[1].reshape(plot_points**2)
        else:
            valuations[:,model_idx,0] = xv[model_idx].reshape(plot_points**2)
            valuations[:,model_idx,1] = yv[model_idx].reshape(plot_points**2)
        models_print[model_idx] = models[model_idx].play(valuations[:,model_idx,:])
        models_print_wf[model_idx] = models_print[model_idx].view(plot_points,plot_points,2)

    fig = plt.figure()
    for model_idx in range(len(models)):
        for input_idx in range(input_length):
            ax = fig.add_subplot(len(models), 2, model_idx*input_length+input_idx+1, projection='3d')
            ax.plot_trisurf(
                
                valuations[:,model_idx,0].detach().cpu().numpy(),
                valuations[:,model_idx,1].detach().cpu().numpy(),
                models_print[model_idx][:,input_idx].reshape(plot_points**2).detach().cpu().numpy(),
                color = 'yellow',
                linewidth = 0.2,
                antialiased = True
            )
            # ax.plot_wireframe(
            #     xv[model_idx].detach().cpu().numpy(),
            #     yv[model_idx].detach().cpu().numpy(),
            #     models_print_wf[model_idx][:,:,input_idx].detach().cpu().numpy(),
            #     rstride=4, cstride=4
            # )
            # Axis labeling
            if len(models)>2:
                if model_idx < 4:
                    ax.set_xlim(plot_xmin, plot_xmax-(u1_hi-u0_hi)); ax.set_ylim(plot_ymin, plot_ymax-(u1_hi-u0_hi)); ax.set_zlim(plot_zmin, plot_zmax-(u1_hi-u0_hi))
                else:
                    ax.set_xlim(plot_xmin, plot_xmax); ax.set_ylim(plot_ymin, plot_ymax); ax.set_zlim(plot_zmin, plot_zmax)
            else:
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
    if writer:
        writer.add_figure('eval/bid_function_3d', fig, e)

    plt.show()

def write_bid_function_3d():
    assert input_length == 2, 'Only case of n_items equals 2 can be plotted'
    write_local = (int)(write_points/2) +1
    write_global = write_points + 1

    lin_local = torch.linspace(u_lo, u0_hi, write_local)
    lin_global = torch.linspace(u_lo, u1_hi, write_global)

    prepare_model_output_3d(lin_local,0)
    prepare_model_output_3d(lin_global,1)


def prepare_model_output_3d(points, model_idx):
    num_points = len(points)
    xv, yv = torch.meshgrid([points, points])
    valuations = torch.zeros(num_points**2, input_length, device=device)

    valuations[:,0] = xv.reshape(num_points**2)
    valuations[:,1] = yv.reshape(num_points**2)
    models_print = models[model_idx].play(valuations)

    #Adjust for precision loss when converting to numpy
    np.savetxt(os.path.join(logdir, 'data_%s.csv' %model_idx), torch.cat((valuations.squeeze(0) + 0.00001,
               models_print), 1).detach().cpu().numpy(), delimiter=',')#, fmt='%1.6f')



##########################################################################
# initialize models
if model_sharing:
    models = [None] * 2 
else:
    models = [None] * n_players

# Generate sample
tmp_plot_points = plot_points
plot_points = 100
lin_local = torch.linspace(u_lo, u0_hi, plot_points)
lin_global = torch.linspace(u_lo, u1_hi, plot_points)
xv = [None] * 2
yv = [None] * 2
xv[0], yv[0] = torch.meshgrid([lin_local, lin_local])
xv[1], yv[1] = torch.meshgrid([lin_global, lin_global])
valuations = torch.zeros(plot_points**2, len(models), input_length, device=device)
models_print = [None] * len(models)
for model_idx in range(len(models)):
    if len(models) > 2:
        valuations[:,model_idx,0] = xv[0].reshape(plot_points**2)
        valuations[:,model_idx,1] = yv[0].reshape(plot_points**2)
        if model_idx>3:
            valuations[:,model_idx,0] = xv[1].reshape(plot_points**2)
            valuations[:,model_idx,1] = yv[1].reshape(plot_points**2)    
    else:
        valuations[:,model_idx,0] = xv[model_idx].reshape(plot_points**2)
        valuations[:,model_idx,1] = yv[model_idx].reshape(plot_points**2)

sample = torch.tensor([[float(u0_hi), float(u0_hi)],
             [float(u0_hi), float(u1_hi)],
             [float(u1_hi), float(u1_hi)]])
    

for k,v in enumerate(models):
    models[k] = NeuralNetStrategy(input_length,                            
                            hidden_nodes = hidden_nodes,
                            hidden_activations = hidden_activations,
                            ensure_positive_output = sample,
                            output_length = 2).to(device)
    models[k].pretrain(valuations[:,k,:],100)

plot_points = tmp_plot_points

n_parameters = [sum([p.numel() for model in models for p in model.parameters()]),sum([p.numel() for model in models for p in model.parameters()])]

bidders = [None] * n_players

for k,v in enumerate(bidders):
    if (model_sharing & k < n_players - 2):
        bidders[k] = strat_to_bidder(models[0], batch_size, player_position=k)
    elif model_sharing & k >= n_players - 2:
        bidders[k] = strat_to_bidder(models[1], batch_size, player_position=k)
    else:
        bidders[k] = strat_to_bidder(models[k], batch_size, player_position=k)

mechanism = LLLLGGAuction(rule = pricing_rule, 
                         cuda = cuda, core_solver = core_solver, parallel = n_threads)
env = AuctionEnvironment(mechanism,
                  agents = bidders,
                  batch_size = batch_size,
                  n_players =n_players,
                  strategy_to_player_closure = strat_to_bidder
                 )

learners = [None] * len(models)
for k,v in enumerate(learners):
        if len(learners) == 2:
            player_position_tmp = k*4
        else:
            player_position_tmp = k
        learners[k] = ESPGLearner(model = models[k],
                                environment = env,
                                hyperparams = learner_hyperparams,
                                optimizer_type = optimizer_type,
                                optimizer_hyperparams = optimizer_hyperparams,
                                strat_to_player_kwargs = {'player_position': player_position_tmp})

###################################################Logging####################################################
print(log_root)

if os.name == 'nt': raise ValueError('The run_name may not contain : on Windows! (change datetime format to fix this)') 
run_name = time.strftime('%Y-%m-%d %a %H:%M')
if run_comment:
    run_name = run_name + ' - ' + str(run_comment)
logdir = os.path.join(log_root, 'experiments', 'LLLLGG', str(n_players) + 'p', run_name)
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
expect_counter = 0
with SummaryWriter(logdir, flush_secs=60) as writer:
    #TODO: Write parameters
    prep_nodes = ''
    prep_activations = ''
    for i in range(len(hidden_nodes)):
        prep_nodes += str(hidden_nodes[i])
        prep_activations += str(hidden_activations[i])
    parameter_setting = 'device: %s,\r\n \
                         pricing_rule: %s,\r\n \
                         core_solver: %s,\n \
                         model_sharing: %s, \
                         batch_size: %s, \
                                            \
                         input_length: %s, \
                         hidden_nodes_num: %s, \
                         hidden_nodes: %s, \
                         hidden_activities: %s, \
                                                    \
                         population_size: %s, \
                         sigma: %s, \
                         scale_sigma_by_model_size: %s, \
                                                    \
                         optimizer_type: %s,    \
                         optimizer_hyperparams: %s, \
                                                \
                         plot_epoch: %s' \
                         %(device, pricing_rule, core_solver, model_sharing,
                         batch_size, input_length, len(hidden_nodes), prep_nodes, prep_activations,  
                         learner_hyperparams['population_size'], learner_hyperparams['sigma'], 
                         learner_hyperparams['scale_sigma_by_model_size'], optimizer_type,
                         optimizer_hyperparams, plot_epoch)

    writer.add_text('parameter_setting', parameter_setting)


    overhead_mins = 0
    torch.cuda.empty_cache()
    fig = plt.figure()

    for e in range(epoch+1):
        #try:
        #    if expect_counter>5:
        #        sys.exit()
        if e % 100 == 0:
            print(e)    
        # always: do optimizer step
        utilities = [None] * len(learners)
        for k,v in enumerate(learners):
            utilities[k] = learners[k].update_strategy_and_evaluate_utility()
        #logging 
        start_time = timer()
        #TODO: Write 1. Utility of each,
        if model_sharing:
            writer.add_scalars('utilities',
                            {'l': utilities[0],
                             'g': utilities[1],
                            }, e)
        else:
            writer.add_scalars('utilities',
                            {'l1': utilities[0],
                                'l2': utilities[1],
                                'l3': utilities[2],
                                'l4': utilities[3],
                                'g1': utilities[4],
                                'g2': utilities[5]
                            }, e)
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
            #plot_bid_function(fig, v, b, writer,e,plot_points = plot_points,
            #                      save_png_to_disc=save_figure_to_disc)  
            #plot_bid_function_3d(writer=writer,e=e,save_figure_to_disc = save_figure_to_disc)

            bid_i = torch.linspace(u_lo, u1_hi, regret_bid_size)

            player_position = 4

            val = env.agents[player_position].valuations
            agent_bid = env.agents[player_position].get_action()
            action_length = agent_bid.shape[1]
            bid_profile = torch.zeros(env.batch_size, env.n_players, action_length,
                                          dtype=agent_bid.dtype, device = env.mechanism.device)

            counter = 1
            for opponent_pos, opponent_bid in env._generate_agent_actions(exclude = set([player_position])):
                    # since auction mechanisms are symmetric, we'll define 'our' agent to have position 0
                    if opponent_pos is None:
                        opponent_pos = counter
                    bid_profile[:, opponent_pos, :] = opponent_bid
                    counter = counter + 1
            print("Calculating regret...")
            torch.cuda.empty_cache()
            regret = metrics.ex_interim_regret(mechanism, bid_profile, player_position, val,
                                    agent_bid, bid_i)
            
            print("agent {} can improve by, avg: {}, max: {}".format(player_position,
                                                                 torch.mean(regret),
                                                                 torch.max(regret)))



            out = torch.cat((val, regret.view(batch_size,1)),1)
            np.savetxt(os.path.join(logdir, 'regret.txt'), out.detach().cpu().numpy(), delimiter=',')
            

        if e%write_epoch == 0:
            write_bid_function_3d()
        elapsed = timer() - start_time
        overhead_mins = overhead_mins + elapsed/60
        writer.add_scalar('debug/overhead_mins', overhead_mins, e)

        #except:
        #    print("------------------A RuntimeError occured by I keep moooving oooon!-----------------------\n")
        #    except_counter += 1