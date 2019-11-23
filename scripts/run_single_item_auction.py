""" The run function of this script runs multiple runs of training on single_item auctions """

import os, sys, time, warnings
from timeit import default_timer as timer
root_path = os.path.join(os.path.expanduser('~'), 'bnelearn')
if root_path not in sys.path:
    sys.path.append(root_path)

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.utils as ut
from torch.optim.optimizer import Optimizer, required #pylint: disable=no-name-in-module
from torch.utils.tensorboard import SummaryWriter

from bnelearn.strategy import NeuralNetStrategy, ClosureStrategy
from bnelearn.bidder import Bidder
from bnelearn.mechanism import FirstPriceSealedBidAuction, VickreyAuction
from bnelearn.learner import ESPGLearner
from bnelearn.environment import AuctionEnvironment
from bnelearn.experiment import Experiment

##%%%%%%%%%%%%%%%%%%%%%%%%%%    Settings
# device and seed
cuda = True
specific_gpu = 3

n_runs = 3
seeds = list(range(n_runs))
epochs = 100

# Logging and plotting
logging_options = dict(
    log_root = os.path.join(root_path, 'experiments'),
    save_figure_to_disc_png = False,
    save_figure_to_disc_svg = False, #for publishing. better quality but a pain to work with
    plot_epoch = 100,
    show_plot_inline = True
)

# Experiment setting parameters
n_players = 2
auction_mechanism = 'first_price' # one of 'first_price', 'second_price'
valuation_prior = 'uniform' # for now, one of 'uniform' / 'normal', specific params defined in script


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

# in single item auctions there's only a single input
### strategy model architecture
input_length = 1
hidden_nodes = [5, 5, 5]
hidden_activations = [nn.SELU(), nn.SELU(), nn.SELU()]

# Evaluation
eval_batch_size = 2**15
cache_eval_actions = True
n_processes_optimal_strategy = 44 if valuation_prior != 'uniform' else 0

######################### No settings beyond this point ######################


# set up matplotlib
is_ipython = 'inline' in plt.get_backend()
if is_ipython:
    from IPython import display
plt.rcParams['figure.figsize'] = [8, 5]

### set device settings
if cuda and not torch.cuda.is_available():
    warnings.warn('Cuda not available. Falling back to CPU!')
    cuda = False
device = 'cuda' if cuda else 'cpu'

if cuda and specific_gpu:
    torch.cuda.set_device(specific_gpu)

### Game setup
if auction_mechanism == 'first_price' :
    mechanism = FirstPriceSealedBidAuction(cuda = cuda)
elif auction_mechanism == 'second_price':
    mechanism = VickreyAuction(cuda = cuda)

### Set up experiment domain and bidders



if valuation_prior == 'uniform':
    u_lo =0
    u_hi =10
    risk = 1 # risk parameter for agent <-- not implemented in bidder yet but used in calculation of optimal utility
    positive_output_point = u_hi
    def strat_to_bidder(strategy, batch_size=batch_size, player_position=None, cache_actions=False):
        return Bidder.uniform(u_lo, u_hi, strategy, batch_size = batch_size,
                              player_position=player_position, cache_actions=cache_actions)
    plot_xmin = u_lo
    plot_xmax = u_hi
    plot_ymin = 0
    plot_ymax = 10
elif valuation_prior == 'normal':
    valuation_mean = 10.0
    valuation_std = 5.0
    positive_output_point = valuation_mean

    plot_xmin = int(max(0, valuation_mean - 3*valuation_std))
    plot_xmax = int(valuation_mean + 3*valuation_std)
    plot_ymin = 0
    plot_ymax = 20
    def strat_to_bidder(strategy, batch_size=batch_size, player_position=None, cache_actions=False):
        return Bidder.normal(valuation_mean,valuation_std, strategy,
                             batch_size = batch_size,
                             player_position=player_position,
                             cache_actions=cache_actions)
else:
    raise ValueError('Only normal and uniform prios supported by this script.')

def setup_bidders(self, model_sharing = True):
    if model_sharing:
        print('Setting up bidders with model Sharing...')
        self.model = NeuralNetStrategy(
            input_length, hidden_nodes = hidden_nodes,hidden_activations = hidden_activations,
            ensure_positive_output = torch.tensor([float(positive_output_point)])
            ).to(device)


        self.bidders = [strat_to_bidder(self.model, batch_size, player_position)
                   for player_position in range(n_players)]
        if pretrain_iters > 0:
            print('\tpretraining...')
            self.model.pretrain(self.bidders[0].valuations, pretrain_iters)

### Setup Learning Environment and Learner(s)
def setup_learning_environment(self): self.env = AuctionEnvironment(self.mechanism, agents = self.bidders,
                                  batch_size = batch_size, n_players =n_players,
                                  strategy_to_player_closure = strat_to_bidder)
def setup_learner(self): self.learner = ESPGLearner(
        model = self.model, environment = self.env, hyperparams = learner_hyperparams,
        optimizer_type = optimizer_type, optimizer_hyperparams = optimizer_hyperparams)


### Setup Evaluation
# for evaluation
if valuation_prior == 'uniform':
    def optimal_bid(valuation):
        return valuation * (n_players - 1) / n_players
elif valuation_prior == 'normal':
    import scipy.integrate as integrate
    common_dist = torch.distributions.normal.Normal(loc = valuation_mean, scale = valuation_std)
    def optimal_bid(valuation: torch.Tensor or np.ndarray or float) -> torch.Tensor:
        # For float and numpy --> convert to tensor
        if not isinstance(valuation, torch.Tensor):
            valuation = torch.tensor(valuation, dtype = torch.float)
        # For float / 0d tensors --> unsqueeze to allow list comprehension below
        if valuation.dim() == 0:
            valuation.unsqueeze_(0)

        # shorthand notation for F^(n-1)
        Fpowered = lambda v: torch.pow(common_dist.cdf(v), n_players - 1)

        # do the calculations
        numerator = torch.tensor(
                [integrate.quad(Fpowered, 0, v)[0] for v in valuation],
                device = valuation.device
            ).reshape(valuation.shape)
        return valuation - numerator / Fpowered(valuation)

bneStrategy = ClosureStrategy(optimal_bid, parallel=n_processes_optimal_strategy)

if valuation_prior == 'uniform':
    def setup_eval_environment(self):
        # environment filled with optimal players for logging
        # use higher batch size for calculating optimum
        self.bne_env = AuctionEnvironment(self.mechanism,
                                    agents = [strat_to_bidder(bneStrategy,
                                                            player_position= i,
                                                            batch_size = eval_batch_size,
                                                            cache_actions=cache_eval_actions)
                                            for i in range(n_players)],
                                    batch_size = eval_batch_size,
                                    n_players=n_players,
                                    strategy_to_player_closure = strat_to_bidder
                                )
        self.bne_utility = risk/(n_players - 1 + risk)*(u_hi - u_lo)/(n_players+1)

elif valuation_prior == 'normal':
    ## define bne agents once then use them in all runs
    global_bne_env = AuctionEnvironment(mechanism,
                                agents = [strat_to_bidder(bneStrategy,
                                                          player_position= i,
                                                          batch_size = eval_batch_size,
                                                          cache_actions=cache_eval_actions)
                                          for i in range(n_players)],
                                batch_size = eval_batch_size,
                                n_players=n_players,
                                strategy_to_player_closure = strat_to_bidder
                               )
    with warnings.catch_warnings(): 
        warnings.simplefilter('ignore')
        # don't print scipy accuracy warnings
        global_bne_utility, analytical_error = integrate.dblquad(
            lambda x,v: common_dist.cdf(x)**(n_players - 1) * common_dist.log_prob(v).exp(),
            0, float('inf'), # outer boundaries
            lambda v: 0, lambda v: v) # inner boundaries
        global_bne_utility_sampled = global_bne_env.get_reward(global_bne_env.agents[0], draw_valuations = True)
        if analytical_error > 1e-7:
            warnings.warn('Error in optimal utility might not be negligible')
    print("Utility in BNE (analytical): \t{:.5f}".format(global_bne_utility))
    print('Utility in BNE (sampled): \t{:.5f}'.format(global_bne_utility_sampled))

    def setup_eval_environment(self):
        # environment filled with optimal players for logging
        # use higher batch size for calculating optimum
        self.bne_env = global_bne_env
        self.bne_utility = global_bne_utility



### Setup Plotting
plot_points = min(150, batch_size)
v_opt = np.linspace(plot_xmin, plot_xmax, 100)
b_opt = optimal_bid(v_opt)
def plot_bid_function(self, fig, plot_data, writer=None, e=None):
    v,b = plot_data
    v = v.detach().cpu().numpy()[:plot_points]
    b= b.detach().cpu().numpy()[:plot_points]

    # create the plot
    fig = plt.gcf()
    plt.cla()
    plt.xlim(plot_xmin, plot_xmax)
    plt.ylim(plot_ymin, plot_ymax)
    plt.xlabel('valuation')
    plt.ylabel('bid')
    plt.text(plot_xmin + 0.05*(plot_xmax - plot_xmin),
             plot_ymax - 0.05*(plot_ymax - plot_ymin),
             'iteration {}'.format(e))
    plt.plot(v,b, 'o', v_opt, b_opt, 'r--')

    #show and/or log
    self._process_figure(fig, writer, e)

## Setup logging
def log_once(self, writer, e):
    """Everything that should be logged only once on initialization."""
    #writer.add_scalar('debug/total_model_parameters', n_parameters, e)
    #writer.add_text('hyperparams/neural_net_spec', str(self.model), 0)
    #writer.add_scalar('debug/eval_batch_size', eval_batch_size, e)
    writer.add_graph(self.model, self.env.agents[0].valuations)

def log_metrics(self, writer, e):
    writer.add_scalar('eval/utility', self.utility, e)
    writer.add_scalar('debug/norm_parameter_update', self.update_norm, e)
    writer.add_scalar('eval/utility_vs_bne', self.utility_vs_bne, e)
    writer.add_scalar('eval/epsilon_relative', self.epsilon_relative, e)
    writer.add_scalar('eval/epsilon_absolute', self.epsilon_absolute, e) # debug because only interesting to see if numeric precision is a problem, otherwise same as relative but scaled.

# TODO: deferred until writing logger
def log_hyperparams(self, writer, e):
    """Everything that should be logged on every learning_rate updates"""
#     writer.add_scalar('hyperparams/batch_size', batch_size, e)
#     writer.add_scalar('hyperparams/learning_rate', learning_rate, e)
#     writer.add_scalar('hyperparams/momentum', momentum, e)
#     writer.add_scalar('hyperparams/sigma', sigma, e)
#     writer.add_scalar('hyperparams/n_perturbations', n_perturbations, e)

## Define Training Loop
def training_loop(self, writer, e):

    ### do in every iteration ###
    # save current params to calculate update norm
    prev_params = torch.nn.utils.parameters_to_vector(self.model.parameters())
    #update model
    self.utility = self.learner.update_strategy_and_evaluate_utility()

    ## everything after this is logging --> measure overhead
    start_time = timer()

    # calculate infinity-norm of update step
    new_params = torch.nn.utils.parameters_to_vector(self.model.parameters())
    self.update_norm = (new_params - prev_params).norm(float('inf'))
    # calculate utility vs bne
    self.utility_vs_bne = self.bne_env.get_reward(strat_to_bidder(self.model, batch_size = eval_batch_size), draw_valuations=False)
    self.epsilon_relative = 1 - self.utility_vs_bne / self.bne_utility
    self.epsilon_absolute = self.bne_utility - self.utility_vs_bne

    self.log_metrics(writer, e)

    if e % self._logging_options['plot_epoch'] == 0:
        # plot current function output
        #bidder = strat_to_bidder(model, batch_size)
        #bidder.draw_valuations_()
        v = self.bidders[0].valuations
        b = self.bidders[0].get_action()
        plot_data = (v,b)

        print("Epoch {}: \tcurrent utility: {:.3f},\t utility vs BNE: {:.3f}, \tepsilon (abs/rel): ({:.5f}, {:.5f})".format(
            e, self.utility, self.utility_vs_bne, self.epsilon_absolute, self.epsilon_relative))
        self.plot(self.fig, plot_data ,writer,e)

    elapsed = timer() - start_time
    self.overhead_mins = self.overhead_mins + elapsed/60
    writer.add_scalar('debug/overhead_mins', self.overhead_mins, e)

# Define Experiment Class
class SymmetricSingleItemAuctionExperiment(Experiment):
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

    exp = SymmetricSingleItemAuctionExperiment(
        name = ['single_item', auction_mechanism, valuation_prior, 'symmetric', str(n_players)+'p'],
        mechanism = mechanism,
        n_players = n_players,
        logging_options = logging_options)

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