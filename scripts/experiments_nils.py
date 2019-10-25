"""
Author Nils Kohring
Date   Oct 2019
Desc   Testing of learning in multi unit auction formats
"""


# ## Imports
import sys
import os
import time
import random
from functools import partial
import warnings

import torch
import torch.nn as nn
# import torch.nn.utils as ut
# from torch.optim.optimizer import Optimizer, required

sys.path.append(os.path.realpath('.'))
from bnelearn.strategy import NeuralNetStrategy, ClosureStrategy
from bnelearn.bidder import Bidder
from bnelearn.mechanism import MultiItemDiscriminatoryAuction, \
    MultiItemUniformPriceAuction, MultiItemVickreyAuction
from bnelearn.learner import ESPGLearner
from bnelearn.environment import AuctionEnvironment

from torch.utils.tensorboard import SummaryWriter
import numpy as np
import matplotlib.pyplot as plt
from itertools import product


# set up matplotlib
plot_epoch = 100
plt.rcParams['figure.figsize'] = [10, 7]

cuda = torch.cuda.is_available()
device = 'cuda' if cuda else 'cpu'

# Use specific cuda gpu if desired (i.e. for running multiple experiments in parallel)
specific_gpu = 7
if cuda and specific_gpu:
    torch.cuda.set_device(specific_gpu)

print('device:', device, end=' ')
if cuda:
    print(torch.cuda.current_device())

# Set random seeds
# seed = 69
# random.seed(seed)
# torch.manual_seed(seed)
# torch.cuda.manual_seed(seed)
# np.random.seed(seed)


# ## Settings

# auction type
mechanism = MultiItemVickreyAuction(cuda=True)

## Experiment setup
n_players = 2
n_items = 2
item_interest_limit = None#2
model_sharing = False

# log in folder
log_root = os.path.abspath('/home/kohring/bnelearn/experiments')
save_figure_data_to_disc = False
save_figure_to_disc = True

auction_type_str = str(type(mechanism))
auction_type_str = str(auction_type_str[len(auction_type_str) \
                       - auction_type_str[::-1].find('.'):-2])
log_name = auction_type_str + '_' + str(n_players) + 'players_' + str(n_items) + 'items'

# valuation distribution
u_lo = 0
u_hi = 1

def strat_to_bidder(strategy, batch_size, player_position):
    return Bidder.uniform(
            u_lo, u_hi, strategy,
            n_items = n_items,
            item_interest_limit = item_interest_limit,
            descending_valuations = True,
            player_position = player_position,
            batch_size = batch_size
        )

## Environment settings
batch_size = 2**17
epoch = 4000
epo_n = 8

# strategy model architecture
input_length = n_items
hidden_nodes = [5, 5]
hidden_activations = [nn.SELU(), nn.SELU()]

hyperparams = {
        'population_size':           [64],
        'sigma':                     [1.0],
        'scale_sigma_by_model_size': [True],
        'normalize_gradients':       [True],
        'lr':                        [1e-2],
        'momentum':                  [0.9]
    }


for vals in product(*hyperparams.values()):
    population_size, sigma, scale_sigma_by_model_size, normalize_gradients, lr, momentum = vals

    print('\nhyperparams\n-----------')
    for k in hyperparams.keys():
        print('{}: {}'.format(k, eval(k)))
    print('-----------\n')

    learner_hyperparams = {
            'population_size': population_size,
            'sigma': sigma,
            'scale_sigma_by_model_size': scale_sigma_by_model_size,
            'normalize_gradients': normalize_gradients
        }

    optimizer_type = torch.optim.SGD
    # lr_scheduler = ReduceLROnPlateau(optimizer, 'min')
    optimizer_hyperparams = {
            'lr': lr,
            # 'weight_decay': 0.,
            # 'lr_decay': 0.97,
            'momentum': momentum
        }


    # ## Setting up the Environment

    def optimal_bid(
            valuation: torch.Tensor or np.ndarray or float,
            player_position: int = 0
        ) -> torch.Tensor:

        if not isinstance(valuation, torch.Tensor):
            valuation = torch.tensor(valuation, dtype=torch.float, device=device)

        # unsqueeze if simple float
        if valuation.dim() == 0:
            valuation.unsqueeze_(0)

        if isinstance(mechanism, MultiItemDiscriminatoryAuction): # is inefficient
            warnings.warn("No explict BNE for MultiItemDiscriminatoryAuction known.", Warning)
            return valuation
        elif isinstance(mechanism, MultiItemUniformPriceAuction): # is inefficient
            if n_players == 2 and n_items == 3 and u_lo == 0 \
                and u_hi == 1 and item_interest_limit == 2:
                opt_bid = torch.clone(valuation)
                opt_bid[1,:] = opt_bid[1,:] ** 2
                opt_bid[2,:] = 0
                return opt_bid
            else:
                warnings.warn("No explict BNE for MultiItemUniformPriceAuction known.", Warning)
                return valuation
        elif isinstance(mechanism, MultiItemVickreyAuction): # is efficient
            return valuation
        else:
            raise ValueError('Unknown optimal strategy for auction type', type(mechanism))


    # def setup_custom_scalar_plots(writer):
    #     ## define layout first, then call add_custom_scalars once
    #     layout = {
    #         'eval': {
    #             'Utilities (SP and BNE)': ['Multiline',
    #                                        ['eval_players/p{}_utility'.format(i) for i in range(n_players)]],
    #             'Utilities Self Play':  ['Multiline',
    #                                      ['eval_players/p{}_utility_sp'.format(i) for i in range(n_players)]],
    #             'Utilities vs BNE':     ['Multiline',
    #                                      ['eval_players/p{}_utility_vs_bne'.format(i) for i in range(n_players)]],
    #             'Loss vs BNE absolute': ['Multiline',
    #                                      ['eval_players/p{}_epsilon_absolute'.format(i) for i in range(n_players)]],
    #             'Loss vs BNE relative': ['Multiline',
    #                                      ['eval_players/p{}_epsilon_relative'.format(i) for i in range(n_players)]]
    #             #'How to make a margin chart': ['Margin', ['tag_mean', 'tag_min', 'tag_max']]
    #         }
    #     }
    #     writer.add_custom_scalars(layout)

    def log_once(writer, e):
        """Everything that should be logged only once on initialization."""
        for agent in range(n_players):
            writer.add_scalar(log_name + ' hyperparameters/p{}_model_parameters'.format(agent),
                              n_parameters[agent], e)
        writer.add_scalar(log_name + ' hyperparameters/model_parameters', sum(n_parameters), e)
        writer.add_text(log_name + ' hyperparameters/neural_net_spec', str(models[0]), 0)
        writer.add_scalar(log_name + ' hyperparameters/batch_size', batch_size, e)
        writer.add_scalar(log_name + ' hyperparameters/epochs', epoch, e)
        for key, value in learner_hyperparams.items():
            writer.add_scalar(log_name + ' hyperparameters/' + str(key), value, e)
        writer.add_text(log_name + ' hyperparameters/optimizer', str(optimizer_type), e)
        for key, value in optimizer_hyperparams.items():
            writer.add_scalar(log_name + ' hyperparameters/' + str(key), value, e)
        writer.add_graph(models[0], env.agents[0].valuations)

    def log_metrics(writer, utilities, bne_utilities, e):
        """log scalar for each player. Tensor should be of shape n_players"""
        u, bne_u = iter(utilities), iter(bne_utilities)
        u_vs_bne = iter(utilities - bne_utilities)
        writer.add_scalars(log_name + ' eval/utilities',
                           dict(zip(['player_{}'.format(i) for i in range(n_players)], u)), e)
        writer.add_scalars(log_name + ' eval/bne_utilities',
                           dict(zip(['player_{}'.format(i) for i in range(n_players)], bne_u)), e)
        writer.add_scalars(log_name + ' eval/utilities_vs_bne',
                           dict(zip(['player_{}'.format(i) for i in range(n_players)], u_vs_bne)), e)
        writer.add_scalar(log_name + ' eval/total_bne_utilities', sum(bne_utilities), e)
        writer.add_scalar(log_name + ' eval/total_utilites', sum(utilities), e)
        writer.add_scalar(log_name + ' eval/total_utilities_vs_bne', sum(utilities - bne_utilities), e)


    v_opt = torch.linspace(u_lo, u_hi, 25).repeat(n_items, 1)
    b_opt = optimal_bid(v_opt).cpu().numpy()

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
              '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
              '#bcbd22', '#17becf']

    def batched_index_select(input, dim, index):
        """
        Extends the torch ´index_select´ function to be used for multiple batches
        at once.

        author:
            dashesy @ https://discuss.pytorch.org/t/batched-index-select/9115/11

        args:
            input: Tensor which is to be indexed
            dim: Dimension
            index: Index tensor which proviedes the seleting and ordering.

        returns/yields:
            Indexed tensor
        """
        for ii in range(1, len(input.shape)):
            if ii != dim:
                index = index.unsqueeze(ii)
        expanse = list(input.shape)
        expanse[0] = -1
        expanse[dim] = -1
        index = index.expand(expanse)
        return torch.gather(input, dim, index)

    def plot_bid_function(
            valuations, actions, writer=None, e=None,
            sort_by_bids = False,
            save_vectors_to_disc = save_figure_data_to_disc,
            save_fig_to_disc = False
        ):
        """Method for plotting"""

        plot_points = min(100, batch_size)

        if save_vectors_to_disc:
            pass
        #     np.savez(
        #             os.path.join(logdir, 'figure_data.npz'),
        #             v0_opt = v0_opt,
        #             b0_opt = b0_opt,
        #             v1_opt = v1_opt,
        #             b1_opt = b1_opt,
        #             v0=v0, b0=b0, v1=v1, b1=b1
        #         )

        # sorting of points, s.t. 1st plot coresponds to 1st item, etc.
        # (from sorted values to sorted bids)
        vals, acts = list(), list()
        for val, act in zip(valuations, actions):
            if sort_by_bids:
                sorted_idx = torch.sort(act, dim=1, descending=True)[1]
                vals.append(batched_index_select(val, 1, sorted_idx).detach().cpu().numpy()[:plot_points])
                acts.append(batched_index_select(act, 1, sorted_idx).detach().cpu().numpy()[:plot_points])
            else:
                vals.append(val.detach().cpu().numpy()[:plot_points])
                acts.append(act.detach().cpu().numpy()[:plot_points])

        fig, axs = plt.subplots(n_items, sharex=True)
        plt.cla()

        if not isinstance(axs, np.ndarray): # only one item/plot
            axs = [axs]

        for item in range(n_items):
            axs[item].plot(v_opt[item,:], b_opt[item,:], '--', color='grey')
            for agent_idx in range(n_players):
                zeros = acts[agent_idx][:,item] < 1e-9
                axs[item].plot(
                    vals[agent_idx][:,item][~zeros], acts[agent_idx][:,item][~zeros], 'o',
                    color = colors[agent_idx],
                    label = 'agent ' + str(agent_idx)
                )
                axs[item].plot(
                    vals[agent_idx][:,item][zeros], acts[agent_idx][:,item][zeros], 'x',
                    color = colors[agent_idx],
                )
            axs[item].set_title('item ' + str(item))
            if item == n_items - 1:
                axs[item].set_xlabel('valuation')
            axs[item].set_ylabel('bid')
            axs[item].set_xlim([u_lo, u_hi])
            axs[item].set_ylim([u_lo, u_hi])
            if item == 0:
                axs[item].legend(title='iteration {}'.format(e), loc='upper left')
            axs[item].grid(True)
        fig.tight_layout()

        if save_fig_to_disc:
            plt.savefig(os.path.join(logdir, 'plots', f'_{e:05}.png'))
        if writer:
            writer.add_figure(log_name + ' eval/plot', fig, e)

        plt.show()

    def plot_bid_function_3d(writer, e, save_fig_to_disc=False):
        assert n_items == 2, 'Only case of n_items equals 2 can be plotted.'

        plot_points = min(10, batch_size)

        from mpl_toolkits.mplot3d import Axes3D
        from matplotlib.ticker import LinearLocator, FormatStrFormatter

        lin = torch.linspace(u_lo, u_hi, plot_points)
        x, y = torch.meshgrid([lin, lin])

        plot_valuations = torch.zeros(plot_points**2, n_players, n_items, device=device)
        for agent_idx in range(n_players):
            plot_valuations[:,agent_idx,0] = x.reshape(plot_points**2)
            plot_valuations[:,agent_idx,1] = y.reshape(plot_points**2)
            strategy = bidders[agent_idx].strategy.play(
                    plot_valuations[:,agent_idx,:].view(plot_points**2, -1)
                )

        x, y = x.detach().cpu().numpy(), y.detach().cpu().numpy()
        mask = x >= y

        fig = plt.figure()
        for agent_idx in range(n_players):
            for item_idx in range(n_items):
                ax = fig.add_subplot(n_players, 2, agent_idx*n_items+item_idx+1, projection='3d')
                b = strategy[:,item_idx].reshape(plot_points, plot_points).detach().cpu().numpy()
                ax.plot_trisurf(x[mask], y[mask], b[mask],
                        cmap = 'plasma',
                        # linewidth = 0,
                        antialiased = True
                    )
                ax.set_xlim(u_lo, u_hi); ax.set_ylim(u_lo, u_hi); ax.set_zlim(u_lo, u_hi+.1*(u_hi-u_lo))
                ax.set_xlabel('item 0 value'); ax.set_ylabel('item 1 value')#; ax.set_zlabel('bid')
                ax.zaxis.set_major_locator(LinearLocator(10))
                ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
                ax.set_title('agent {} bidding for her {}. item'.format(agent_idx, item_idx))
                ax.view_init(20, -135)
        fig.suptitle('iteration {}'.format(e), size=16)
        fig.tight_layout()

        if save_fig_to_disc:
            plt.savefig(os.path.join(logdir, 'plots', f'_{e:05}_3d.png'))
        if writer:
            writer.add_figure(log_name + ' eval/plot_3d', fig, e)

        plt.show()


    # initialize models
    ensure_positive_output = torch.zeros(epo_n, n_items).uniform_(u_lo, u_hi).sort(dim=1, descending=True)[0]
    if model_sharing:
        model = NeuralNetStrategy(input_length,
                hidden_nodes = hidden_nodes,
                hidden_activations = hidden_activations,
                ensure_positive_output = ensure_positive_output,
                output_length = input_length
            ).to(device)
        models = [model for _ in range(n_players)]
    else:
        models = list()
        for _ in range(n_players):
            models.append(
                    NeuralNetStrategy(input_length,
                            hidden_nodes = hidden_nodes,
                            hidden_activations = hidden_activations,
                            ensure_positive_output = ensure_positive_output,
                            output_length = input_length
                        ).to(device)
                )

    n_parameters = list()
    for model in models:
        n_parameters.append(sum([p.numel() for p in model.parameters()]))

    bidders = [strat_to_bidder(model, batch_size, i)
               for i, model in enumerate(models)]

    env = AuctionEnvironment(mechanism,
            agents = bidders,
            batch_size = batch_size,
            n_players = n_players,
            strategy_to_player_closure = strat_to_bidder
        )

    learners = list()
    for i in range(n_players):
        learners.append(
                ESPGLearner(
                        model = models[i],
                        environment = env,
                        hyperparams = learner_hyperparams,
                        optimizer_type = optimizer_type,
                        optimizer_hyperparams = optimizer_hyperparams,
                        strat_to_player_kwargs = {"player_position": i}
                    )
            )


    # ## Set up equilibrium-environment

    bne_strategies = [ClosureStrategy(partial(optimal_bid, player_position=i))
                      for i in range(n_players)]

    bne_env = AuctionEnvironment(
            mechanism,
            agents = [strat_to_bidder(bne_strategies[i], batch_size, i)
                    for i in range(n_players)],
            n_players = n_players,
            batch_size = batch_size,
            strategy_to_player_closure = strat_to_bidder
        )

    # bne_utilities_sampled = torch.tensor([bne_env.get_reward(a, draw_valuations=True) for a in bne_env.agents])
    # print(('Utilities in BNE (sampled):'+ '\t{:.5f}'*n_players + '.').format(*bne_utilities_sampled))
    # eps_abs = lambda us: bne_utilities_sampled - us
    # eps_rel = lambda us: 1 - us/bne_utilities_sampled
    # utilities_vs_bne = torch.tensor(
    #         [bne_env.get_strategy_reward(a.strategy, player_position=i)
    #          for i, a in enumerate(env.agents)]
    #     )
    # print(('Model utility vs BNE: \t'+'\t{:.5f}'*n_players).format(*utilities_vs_bne))
    # utilities_learning_env = torch.tensor(
    #         [env.get_strategy_reward(a.strategy, player_position=i, draw_valuations=True)
    #          for i,a in enumerate(env.agents)]
    #     )
    # print(('Model utility in learning env:'+'\t{:.5f}'*n_players).format(*utilities_learning_env))


    for bidder in bidders:
        bidder.draw_valuations_()


    # ## Training
    run_name = str(time.strftime('%Y%m%d_%H%M%S', time.gmtime()))
    logdir = os.path.join(log_root, 'expiriments_nils', auction_type_str,
                          str(n_players) + 'players_' + str(n_items) + 'items', run_name)
    print('logdir:', logdir)
    os.makedirs(logdir, exist_ok=False)

    if save_figure_to_disc:
        os.mkdir(os.path.join(logdir, 'plots'))

    with SummaryWriter(logdir, flush_secs=60) as writer:

        # setup_custom_scalar_plots(writer)

        overhead_mins = 0
        torch.cuda.empty_cache()
        log_once(writer, 0)

        for e in range(epoch + 1):

            start_time = time.time()
            # torch.cuda.reset_max_memory_allocated(device=device)

            # do optimizer step and record utilities
            utilities = torch.tensor([learner.update_strategy_and_evaluate_utility()
                                      for learner in learners])

            # calculate utility vs BNE
            bne_utilities = torch.tensor([bne_env.get_strategy_reward(a.strategy, player_position=i)
                                          for i, a in enumerate(env.agents)])

            # logging
            log_metrics(writer, utilities, bne_utilities, e)

            if e % plot_epoch == 0:

                # plot current function output
                valuations, actions = list(), list()
                for bidder in bidders:
                    valuations.append(bidder.draw_valuations_())
                    actions.append(bidder.get_action())

                # plot_bid_function(valuations, actions, writer, e,
                #                   save_fig_to_disc=save_figure_to_disc)
                plot_bid_function_3d(writer, e, save_fig_to_disc=save_figure_to_disc)

            elapsed = time.time() - start_time
            overhead_mins += elapsed
            # memory = torch.cuda.max_memory_allocated(device=device) * (2**-17)

            # print('epoch {}:\t{}s\t({}mb)'.format(e, round(elapsed, 4), int(memory)))
            print('epoch {}:\t{}s'.format(e, round(elapsed, 4)))
            writer.add_scalar(log_name + ' eval/overhead_mins', elapsed, e)
