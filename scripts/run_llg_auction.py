""" The run function of this script runs multiple runs of training on single_item auctions """

import os
import sys
import warnings
from timeit import default_timer as timer

import matplotlib.pyplot as plt
import numpy as np
import scipy.integrate as integrate
import torch
import torch.nn as nn
from functools import partial

root_path = os.path.join(os.path.expanduser('~'), 'bnelearn')
if root_path not in sys.path:
    sys.path.append(root_path)

# pylint: disable=wrong-import-position

from bnelearn.bidder import Bidder
from bnelearn.environment import AuctionEnvironment
from bnelearn.experiment import Experiment
from bnelearn.learner import ESPGLearner
from bnelearn.mechanism import FirstPriceSealedBidAuction, VickreyAuction, LLGAuction
from bnelearn.strategy import ClosureStrategy, NeuralNetStrategy

##%%%%%%%%%%%%%%%%%%%%%%%%%%    Settings
# device and seed
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

n_runs = 10
seeds = list(range(n_runs))
epochs = 1000

# Logging and plotting
logging_options = dict(
    log_root = os.path.join(root_path, 'experiments'),
    save_figure_to_disc_png = True,
    save_figure_to_disc_svg = False, #for publishing. better quality but a pain to work with
    plot_epoch = 100,
    show_plot_inline = True,
    save_figure_data_to_disc = True
)

# Experiment setting parameters
n_players = 3


auction_mechanism = LLGAuction # FirstPriceSealedBidAuction, VickreyAuction, LLGAuction
payment_rule =  'vcg'            #'first_price', 'vcg', 'nearest-vcg,...'
gamma = 0
valuation_prior = 'uniform' # for now, one of 'uniform' / 'normal', specific params defined in script
risk = 1.0

if risk == 1.0:
    risk_profile = 'risk_neutral'
elif risk == 0.5:
    risk_profile = 'risk_averse'
else:
    risk_profile = 'other'

u_lo = 0
u_hi_local = 1.0
u_hi_global = 2.0

u_his = [u_hi_local, u_hi_local, u_hi_global]

# Learning
model_sharing = True
pretrain_iters = 500
batch_size = 2**18
## ES
learner_hyperparams = {
    'population_size': 64,
    'sigma': 1.,
    'scale_sigma_by_model_size': True
}
## Optimizer
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
    'lr': 3e-3
}

# plot and log training options
plot_epoch = 100
plot_points = min(100, batch_size)

# in single item auctions there's only a single input
### strategy model architecture
input_length = 1
hidden_nodes = [5, 5, 5]
hidden_activations = [nn.SELU(), nn.SELU(), nn.SELU()]

# Evaluation
eval_batch_size = 2**22
cache_eval_actions = True
n_processes_optimal_strategy = 44 if valuation_prior != 'uniform' and auction_mechanism != 'second_price' else 0

######################### No settings beyond this point ######################

### Set up experiment domain and bidders
def strat_to_bidder(strategy, batch_size, player_position):
    return Bidder.uniform(u_lo, u_his[player_position], strategy, player_position=player_position, batch_size = batch_size)

def setup_bidders(self, local_model_sharing = True):
    print('Setting up models and bidders...')
    model_l1 = NeuralNetStrategy(input_length,
                                 hidden_nodes = hidden_nodes,
                                 hidden_activations = hidden_activations,
                                 ensure_positive_output = torch.tensor([float(u_hi_local)])
                                 ).to(device)

    if not local_model_sharing:
        model_l2 = NeuralNetStrategy(input_length,
                                     hidden_nodes = hidden_nodes,
                                     hidden_activations = hidden_activations,
                                     ensure_positive_output = torch.tensor([float(u_hi_local)])
                                     ).to(device)

    # global player
    model_g = NeuralNetStrategy(input_length,
                                hidden_nodes = hidden_nodes,
                                hidden_activations = hidden_activations,
                                ensure_positive_output = torch.tensor([float(u_hi_global)])
                                ).to(device)
    bidder_l1 = strat_to_bidder(model_l1, batch_size, player_position=0)
    bidder_l2 = strat_to_bidder(model_l1, batch_size, player_position=1) \
                    if local_model_sharing else strat_to_bidder(model_l2, batch_size, player_position=1)
    bidder_g  = strat_to_bidder(model_g,  batch_size, player_position=2)
    
    self.bidders = [bidder_l1, bidder_l2, bidder_g]
    self.n_parameters = [sum([p.numel() for p in model.parameters()]) for model in [b.strategy for b in self.bidders]]

    if pretrain_iters > 0:
        print('\tpretraining...')
        for bidder in self.bidders:
            bidder.strategy.pretrain(bidder.valuations, pretrain_iters)

### Setup Learning Environment and Learner(s)
def setup_learning_environment(self):
    self.env = AuctionEnvironment(self.mechanism,
                                  agents = self.bidders,
                                  batch_size = batch_size,
                                  n_players =n_players,
                                  strategy_to_player_closure = strat_to_bidder)
def setup_learner(self):
    learner_l1 = ESPGLearner(model = self.bidders[0].strategy,
                             environment = self.env,
                             hyperparams = learner_hyperparams,
                             optimizer_type = optimizer_type,
                             optimizer_hyperparams = optimizer_hyperparams,
                             strat_to_player_kwargs={"player_position":0}
                 )

    learner_l2 = ESPGLearner(model = self.bidders[1].strategy,
                             environment = self.env,
                             hyperparams = learner_hyperparams,
                             optimizer_type = optimizer_type,
                             optimizer_hyperparams = optimizer_hyperparams,
                             strat_to_player_kwargs={"player_position":1}
                 )

    learner_g = ESPGLearner(model = self.bidders[2].strategy,
                            environment = self.env,
                            hyperparams = learner_hyperparams,
                            optimizer_type = optimizer_type,
                            optimizer_hyperparams = optimizer_hyperparams,
                            strat_to_player_kwargs={"player_position":2}
                )
    self.learners = [learner_l1, learner_l2, learner_g]



### Setup Evaluation
# for evaluation
def optimal_bid(valuation: torch.Tensor or np.ndarray or float, player_position: int) -> torch.Tensor:
    if not isinstance(valuation, torch.Tensor):
        valuation = torch.tensor(valuation)

    # all core-selecting rules are strategy proof for global player:
    if player_position == 2:
        return valuation

    # local bidders:
    if payment_rule == 'vcg':
        return valuation
    if payment_rule in ['proxy', 'nearest_zero']:
        bid_if_positive = 1 + torch.log(valuation * (1.0-gamma) + gamma)/(1.0-gamma)
        return torch.max( torch.zeros_like(valuation), bid_if_positive) 
    if payment_rule == 'nearest_bid':
        return  (np.log(2) - torch.log(2.0 - (1. - gamma) * valuation))/ (1.- gamma)
    if payment_rule == 'nearest_vcg':
        bid_if_positive = 2. / (2. + gamma) * (valuation - (3. - np.sqrt(9 - (1. - gamma)**2)) / (1. - gamma) ) 
        return torch.max(torch.zeros_like(valuation), bid_if_positive)
    
    raise ValueError('optimal bid not implemented for other rules')

bne_strategies = [
    ClosureStrategy(partial(optimal_bid, player_position=i))
    for i in range(n_players)
]

global_bne_env = AuctionEnvironment(
    auction_mechanism(cuda = cuda, rule = payment_rule),
    agents = [strat_to_bidder(bne_strategies[i], player_position=i, batch_size=eval_batch_size)
              for i in range(n_players)],
    n_players = n_players,
    batch_size = eval_batch_size,
    strategy_to_player_closure = strat_to_bidder
)


#print("Utility in BNE (analytical): \t{:.5f}".format(bne_utility))
global_bne_utility_sampled = torch.tensor([global_bne_env.get_reward(a, draw_valuations = True) for a in global_bne_env.agents])
print(('Utilities in BNE (sampled):'+ '\t{:.5f}'*n_players + '.').format(*global_bne_utility_sampled))


eps_abs = lambda us: global_bne_utility_sampled - us
eps_rel = lambda us: 1- us/global_bne_utility_sampled

def setup_eval_environment(self):
    # environment filled with optimal players for logging
    # use higher batch size for calculating optimum
    self.bne_env = global_bne_env
    #TODO:@Stefan: Check if sampled is ok.
    self.bne_utility = global_bne_utility_sampled


### Setup Plotting
vl1_opt = np.linspace(u_lo, u_hi_local, 25)
bl1_opt = optimal_bid(vl1_opt, 0).numpy()
vl2_opt = np.linspace(u_lo, u_hi_local, 25)
bl2_opt = optimal_bid(vl2_opt, 0).numpy()
vg_opt = np.linspace(u_lo, u_hi_global, 50)
bg_opt = optimal_bid(vg_opt, 2).numpy() 

def plot_bid_function(self, fig, v, b, writer=None, e=None):
    # subsample points and plot
    for i in range(len(v)):
        v[i] = v[i].detach().cpu().numpy()[:plot_points]
        b[i] = b[i].detach().cpu().numpy()[:plot_points]

    fig = plt.gcf()
    plt.cla()
    plt.xlim(0, 2)
    plt.ylim(0, 2)
    plt.xlabel('valuation')
    plt.ylabel('bid')
    plt.text(0 + 1, 2 - 1, 'iteration {}'.format(e))
    plt.plot(v[0],b[0], 'bo', vl1_opt, bl1_opt, 'b--', v[1],b[1], 'go', vl2_opt, bl2_opt, 'g--', v[2],b[2], 'ro', vg_opt,bg_opt, 'r--')
    self._process_figure(fig, writer, e)


def log_once(self, writer, e):
    """Everything that should be logged only once on initialization."""
    for i in range(n_players):
        writer.add_scalar('debug_players/p{}_model_parameters'.format(i), self.n_parameters[i], e)
    writer.add_scalar('debug/model_parameters', sum(self.n_parameters), e)
    writer.add_scalar('debug/eval_batch_size', eval_batch_size, e)
    for a in self.bidders:
        writer.add_text('hyperparams/neural_net_spec', str(a.strategy), 0)
    #writer.add_scalar('debug/eval_batch_size', eval_batch_size, e)
    #Currently only the first grad is printed. TODO: Generalize
    writer.add_graph(self.bidders[0].strategy, self.env.agents[0].valuations)

def log_metrics(self, writer, utility, utilities_vs_bne, e):
    """log scalar for each player. Tensor should be of shape n_players"""
    epsilons_rel = eps_rel(utilities_vs_bne)
    epsilons_abs = eps_abs(utilities_vs_bne)
    
    # redundant logging of utlities for multiline
    for i in range(n_players):
        ## Note: multiline chart capture all tags that match the given beginning of the tag_name,
        ## i.e. eval/utility will match all of  eval/utility, eval/utility_sp and eval/utlity_vs_bne
        ## thus self play utility should be named utility_sp to be able to capture it by itself later.
        writer.add_scalar('eval_players/p{}_utility_sp'.format(i), utility[i], e)
        writer.add_scalar('eval_players/p{}_utility_vs_bne'.format(i), utilities_vs_bne[i], e)
        writer.add_scalar('eval_players/p{}_epsilon_absolute'.format(i), epsilons_abs[i], e)
        writer.add_scalar('eval_players/p{}_epsilon_relative'.format(i), epsilons_rel[i], e)
    
    writer.add_scalar('eval/epsilon_relative', epsilons_rel.mean(),e)
    writer.add_scalar('debug/epsilon_absolute', epsilons_abs.mean(),e)

# TODO: deferred until writing logger
def log_hyperparams(self, writer, e):
    """Everything that should be logged on every learning_rate updates"""
    writer.add_scalar('hyperparams/batch_size', batch_size, e)
    writer.add_scalar('hyperparams/learning_rate', optimizer_hyperparams['lr'], e)
    #writer.add_scalar('hyperparams/momentum', optimizer_hyperparams['momentum'], e)
    writer.add_scalar('hyperparams/sigma', learner_hyperparams['sigma'], e)
    writer.add_scalar('hyperparams/n_perturbations', learner_hyperparams['population_size'], e)

def setup_custom_scalar_plots(writer):    
    ## define layout first, then call add_custom_scalars once
    layout = {'eval':
        {
            #'Loss vs BNE relative': ['Multiline',
            #                         ['eval_players/p{}_epsilon_relative'.format(i) for i in range(n_players)]]
            #'How to make a margin chart': ['Margin', ['tag_mean', 'tag_min', 'tag_max']]
        }
    }    
    writer.add_custom_scalars(layout) 

## Define Training Loop
def training_loop(self, writer, e):
    
    # plot current function output
    v = []
    b = []
    for index in range(self.n_players): #TODO: Change this permantely to this!?: self.players_sharing_model_index:
        self.bidders[index].draw_valuations_()
        v.append(self.bidders[index].valuations)
        b.append(self.bidders[index].get_action())
    if e == 0:       
        self.plot(self.fig, v, b, writer,e)     
    # always: do optimizer step
    utilities = [
        learner.update_strategy_and_evaluate_utility()
        for learner in self.learners
    ]
    
    #logging 
    start_time = timer()
    utilities = torch.tensor(utilities)
    utilities_vs_bne = torch.tensor([global_bne_env.get_strategy_reward(a.strategy, player_position=i) for i,a in enumerate(self.env.agents)])

    
    self.log_metrics(writer, utilities, utilities_vs_bne, e)
    if e % self._logging_options['plot_epoch'] == 0 and e > 0:
        # plot current function output
        v = []
        b = []
        for index in range(n_players):
            self.bidders[index].draw_valuations_()
            v.append(self.bidders[index].valuations)
            b.append(self.bidders[index].get_action())

        
        print(('Epoch: {}: Model utility in learning env:'+'\t{:.5f}'*n_players).format(e, *utilities))            
        #print("Epoch {}: \tutilities: \t p0: {:.3f} \t p1: {:.3f}".format(e, utility_0, utility_1))

        self.plot(self.fig, v, b, writer,e)            
    
    elapsed = timer() - start_time
    overhead_mins = self.overhead_mins + elapsed/60
    writer.add_scalar('debug/overhead_mins', overhead_mins, e)

# Define Experiment Class
class AuctionExperiment(Experiment):
    setup_players = setup_bidders
    setup_learning_environment = setup_learning_environment
    setup_learners = setup_learner
    equilibrium_strategy = optimal_bid
    setup_eval_environment = setup_eval_environment
    plot = plot_bid_function
    log_once = log_once
    log_metrics = log_metrics
    log_hyperparams = log_hyperparams
    training_loop = training_loop

def run(seed, run_comment, epochs):
    ## Create the experiment
    ### Set up random seeds
    if seed is not None:
        torch.random.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    exp = AuctionExperiment(
        name = ['single_item', auction_mechanism.name, valuation_prior,
                'symmetric', risk_profile, str(n_players)+'p'],
        mechanism = auction_mechanism(cuda = cuda, rule = payment_rule),
        n_players = n_players,
        logging_options = logging_options)

    #setup_custom_scalar_plots(writer) <- do we still need this?
    overhead_mins = 0

    exp.run(epochs, run_comment)

    del exp
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    if torch.cuda.memory_allocated() > 0:
        warnings.warn('Theres a memory leak')



#### run the experiments

for seed in seeds:
    print('Running experiment {}'.format(seed))
    run(seed, str(seed), epochs)
