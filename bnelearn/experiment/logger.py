import os
import sys
import time
from abc import ABC, abstractmethod
from typing import Callable
from copy import deepcopy
from timeit import default_timer as timer
import bnelearn.util.metrics as metrics
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import torch
from torch.utils.tensorboard import SummaryWriter

import bnelearn
from bnelearn.experiment.learning_configuration import LearningConfiguration
# TODO: can't import experiment because of circular import :/
#from .experiment import Experiment
#from bnelearn.experiment.MultiUnitExperiment import FPSBSplitAwardAuction2x2

# TODO: can't use type hint for experiment :/
class Logger(ABC):
    def __init__(self, exp, base_dir, save_figure_to_disk_png: bool = True,
                 save_figure_to_disk_svg: bool = True,
                 plot_epoch: int = 100, show_plot_inline: bool = True, save_figure_data_to_disk: bool = False):
        root_path = os.path.join(os.path.expanduser('~'), 'bnelearn')
        if root_path not in sys.path:
            sys.path.append(root_path)

        self.logging_options = dict(
            log_root=os.path.join(root_path, 'experiments'),
            save_figure_to_disk_png=save_figure_to_disk_png,
            save_figure_to_disk_svg=save_figure_to_disk_svg,  # for publishing. better quality but a pain to work with
            plot_epoch=plot_epoch,
            show_plot_inline=show_plot_inline
        )

        self.exp = exp
        self.l_config = exp.l_config
        self.experiment_params = exp.experiment_params


        self.base_dir = base_dir
        self.log_dir = None
        self.fig = None
        self.writer = None

        # plotting
        self.plot_epoch = plot_epoch
        self.save_figure_to_disk_svg = save_figure_to_disk_svg
        self.save_figure_to_disk_png = save_figure_to_disk_png
        self.plot_points = None
        self.max_epochs = None

        self.overhead_mins = 0.0

    # Doesn't seem to be needed
    # def __del__(self):
    #    self.writer.close()

    @abstractmethod
    def log_experiment(self, run_comment, max_epochs: int):
        pass

    # ToDo Make a signature take a single dictionary parameter, as signatures would differ in each class
    @abstractmethod
    def log_training_iteration(self, prev_params, epoch, bne_env, strat_to_bidder, eval_batch_size, bne_utility,
                               utility, log_params: dict):
        pass

    @abstractmethod
    def _plot(self, fig, plot_data, writer: SummaryWriter or None, e=None):
        """This method should implement a vizualization of the experiment at the current state"""
        pass

    def _process_figure(self, fig, writer=None, epoch=None, name='epoch_'):
        """displays, logs and/or saves figure built in plot method"""

        if self.logging_options['save_figure_to_disk_png']:
            plt.savefig(os.path.join(self.log_dir, 'png', f'{name}{epoch:05}.png'))

        if self.logging_options['save_figure_to_disk_svg']:
            plt.savefig(os.path.join(self.log_dir, 'svg', f'{name}{epoch:05}.svg'),
                        format='svg', dpi=1200)
        if writer:
            writer.add_figure('eval/bid_function', fig, epoch)
        if self.logging_options['show_plot_inline']:
            # display.display(plt.gcf())
            plt.show()

    @abstractmethod
    def _log_once(self):
        """Everything that should be logged only once on initialization."""
        pass

    @abstractmethod
    def _log_metrics(self, writer, epoch, utility, update_norm, utility_vs_bne, epsilon_relative, epsilon_absolute,
                     L_2, L_inf, param_group_postfix = '', metric_prefix = ''):
        pass

    @abstractmethod
    def _log_hyperparams(self, writer, epoch):
        pass

# TODO: Allow multiple utilities and params (for multiple learners)
class SingleItemAuctionLogger(Logger):
    def __init__(self, exp, base_dir):
        super().__init__(exp, base_dir)

    def log_experiment(self, run_comment, max_epochs: int):

        # setting up plotting
        self.plot_points = min(100, self.exp.l_config.batch_size)
        self.v_opt = [np.linspace(self.exp.plot_xmin, self.exp.plot_xmax, 100)] * len(self.exp.models)
        # TODO: presumes existence of optimal bid --> should not be assumed here
        self.b_opt = [self.exp._optimal_bid(self.v_opt[i], player_position=model.connected_bidders[0])
                        for i,model in enumerate(self.exp.models)]

        is_ipython = 'inline' in plt.get_backend()
        if is_ipython:
            from IPython import display
        plt.rcParams['figure.figsize'] = [8, 5]

        if os.name == 'nt':
            raise ValueError('The run_name may not contain : on Windows!')
        run_name = time.strftime('%Y-%m-%d %a %H:%M:%S')
        if run_comment:
            run_name = run_name + ' - ' + str(run_comment)

        self.log_dir = os.path.join(self.logging_options['log_root'], self.base_dir, run_name)
        os.makedirs(self.log_dir, exist_ok=False)
        if self.logging_options['save_figure_to_disk_png']:
            os.mkdir(os.path.join(self.log_dir, 'png'))
        if self.logging_options['save_figure_to_disk_svg']:
            os.mkdir(os.path.join(self.log_dir, 'svg'))

        print('Started run. Logging to {}'.format(self.log_dir))
        self.fig = plt.figure()

        self.writer = SummaryWriter(self.log_dir, flush_secs=30)
        self._log_once()
        self._log_hyperparams()

    #TODO: Have to get bne_utilities for all models instead of bne_utoility of only one!?
    def log_training_iteration(self, prev_params, epoch, strat_to_bidder, bne_utilities,
                               utilities, log_params: dict):
        # TODO It is by no means nice that there is so much specific logic in here
        start_time = timer()
        plot_data = []


        model_is_global = len(self.exp.models) == 1

        # TODO: a lot of overhead is currently not counted correctly!
        # TODO: update logging to follow following strategy:
        #       1. calculate ALL updates (if we have multiple models)
        #       2. caclulate ALL metrics as lists
        #       3. log everything (using a single function call)

        for i, model in enumerate(self.exp.models):
            group_postfix = '' if model_is_global else f'_p{i}'
            metric_prefix = ''


            ## TODO: no knowledge of bneenve should be assumed here! Might have settings without bne

            # calculate infinity-norm of update step
            new_params = torch.nn.utils.parameters_to_vector(model.parameters())
            update_norm = (new_params - prev_params[i]).norm(float('inf'))
            # calculate utility vs bne
            utility_vs_bne = self.exp.bne_env.get_reward(
                strat_to_bidder(model, batch_size=self.exp.l_config.eval_batch_size),
                draw_valuations=False)  # False because expensive for normal priors
            epsilon_relative = 1 - utility_vs_bne / bne_utilities[i]
            epsilon_absolute = bne_utilities[i] - utility_vs_bne
            L_2 = metrics.norm_strategy_and_actions(model, self.exp.bne_env.agents[i].get_action(),
                                                    self.exp.bne_env.agents[i].valuations, 2)
            L_inf = metrics.norm_strategy_and_actions(model, self.exp.bne_env.agents[i].get_action(),
                                                      self.exp.bne_env.agents[i].valuations, float('inf'))
            self._log_metrics(writer=self.writer, epoch=epoch, utility=utilities[i], update_norm=update_norm,
                              utility_vs_bne=utility_vs_bne, epsilon_relative=epsilon_relative,
                              epsilon_absolute=epsilon_absolute, L_2=L_2, L_inf=L_inf,
                              param_group_postfix=group_postfix, metric_prefix=metric_prefix)

        if epoch % self.logging_options['plot_epoch'] == 0:
            bidders = [strat_to_bidder(model, self.exp.l_config.batch_size, model.connected_bidders[0]) for model in self.exp.models]
            v = [bidder.valuations for bidder in bidders]
            b = [bidder.get_action() for bidder in bidders]
            plot_data = ((v, b))
            print(
                "Epoch {}: \tcurrent utility: {:.3f},\t vs BNE: {:.3f}, \tepsilon (abs/rel): ({:.5f}, {:.5f})".format(
                    epoch, utilities[i], utility_vs_bne, epsilon_absolute, epsilon_relative))
            self._plot(self.fig, plot_data, self.writer, epoch)

        elapsed = timer() - start_time
        self.overhead_mins = self.overhead_mins + elapsed / 60
        self.writer.add_scalar('debug/overhead_mins', self.overhead_mins, epoch)


    # TODO: rename u_lo, u_hi --> these have NOTHING to do with normal distribution.
    def log_ex_interim_regret(self, epoch, mechanism, env, learners, u_lo, u_hi, regret_batch_size, regret_grid_size):

        original_batch_size = env.agents[0].batch_size

        bid_profile = torch.zeros(regret_batch_size, env.n_players, env.agents[0].n_items,
                                          dtype=env.agents[0].valuations.dtype, device = env.mechanism.device)
        for agent in env.agents:
            agent.batch_size = regret_batch_size
            agent.draw_valuations_new_batch_(regret_batch_size)
            bid_profile[:, agent.player_position, :] = agent.get_action()

        regrets = []
        valuations = []
        max_regret = 0
        for learner in learners:
            player_position = learner.strat_to_player_kwargs['player_position']

            regret_grid = torch.linspace(u_lo[player_position], u_hi[player_position], regret_grid_size)

            #print("Calculating regret...")
            torch.cuda.empty_cache()
            regret = metrics.ex_interim_regret(mechanism, bid_profile, player_position, env.agents[player_position].valuations, regret_grid)

            print("agent {} ex ante/ex interim regrat: avg: {:.3f}, max: {:.3f}".format(player_position,
                                                                 torch.mean(regret),
                                                                 torch.max(regret)))

            self.writer.add_scalar('eval/max_ex_interim_regret', torch.max(regret), epoch)
            self.writer.add_scalar('eval/ex_ante_regret', torch.mean(regret), epoch)
            regrets.append(regret)

            valuations.append(env.agents[player_position].valuations)

            max_regret = max(max_regret, torch.max(regret))

        # if isinstance(self, LLLLGGAuctionLogger):
        #     valuations_tensor = torch.tensor([t.cpu().numpy() for t in valuations]).permute(1,0,2)
        #     regrets_tensor = torch.tensor([t.cpu().numpy() for t in regrets]).view(len(learners), regret_batch_size, 1)
        #     fig, _ = self._plot_3d((valuations_tensor, regrets_tensor), epoch, [self.plot_xmin, self.plot_xmax], [self.plot_ymin, self.plot_ymax],
        #                         [0, max_regret.detach().cpu().numpy()], input_length=1, x_label="valuation", y_label="regret")
        # else:
        fig, _ = self._plot_2d((valuations, regrets), epoch, [self.exp.plot_xmin, self.exp.plot_xmax],
                            [0, max_regret.detach().cpu().numpy()], x_label="valuation", y_label="regret")
        self._process_figure(fig, self.writer, epoch, name="regret_epoch")

        for agent in env.agents:
            agent.batch_size = original_batch_size
            agent.draw_valuations_new_batch_(original_batch_size)


    def _plot(self, fig, plot_data, writer: SummaryWriter or None, e=None):
        """This method should implement a vizualization of the experiment at the current state"""
        fig, plt = self._plot_2d(plot_data, e, [self.exp.plot_xmin, self.exp.plot_xmax], [self.exp.plot_ymin,self.exp.plot_ymax])
        cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']

        for i in range(len(plot_data[0])):
            #TODO: Plottet noch scheiße!
            #TODO: Not working yet for LLG
            plt.plot(self.v_opt[i], self.b_opt[i], color=cycle[i], linestyle = '--')#linestyle = '--')
        # show and/or log
        self._process_figure(fig, writer, e)

    #TODO: Do I need "fig" here?
    def _plot_2d(self, plot_data, epoch, xlim: list,
                 ylim: list, x_label="valuation", y_label="bid"):
        """This implements plotting simple 2d data"""
        plot_xmin = xlim[0]
        plot_xmax = xlim[1]
        plot_ymin = ylim[0]
        plot_ymax = ylim[1]
        x = [None] * len(plot_data[0])
        y = [None] * len(plot_data[0])

        for i in range(len(x)):
            x[i] = plot_data[0][i].detach().cpu().numpy()[:self.plot_points]
            y[i] = plot_data[1][i].detach().cpu().numpy()[:self.plot_points]

        # create the plot
        fig = plt.gcf()
        plt.cla()
        plt.xlim(plot_xmin, plot_xmax)
        plt.ylim(plot_ymin, plot_ymax)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.text(plot_xmin + 0.05 * (plot_xmax - plot_xmin),
                 plot_ymax - 0.05 * (plot_ymax - plot_ymin),
                 'iteration {}'.format(epoch))
        cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
        for i in range(len(x)):
            plt.plot(x[i], y[i], color=cycle[i], marker='o', linestyle = 'None')
        return fig, plt


    # Setup logging
    def _log_once(self):
        """Everything that should be logged only once on initialization."""
        # writer.add_scalar('debug/total_model_parameters', n_parameters, epoch)
        # writer.add_text('hyperparams/neural_net_spec', str(self.model), 0)
        # writer.add_scalar('debug/eval_batch_size', eval_batch_size, epoch)
        self.writer.add_graph(self.exp.models[0], self.exp.env.agents[0].valuations)

    def _log_metrics(self, writer, epoch, utility, update_norm, utility_vs_bne, epsilon_relative, epsilon_absolute,
                     L_2, L_inf, param_group_postfix = '', metric_prefix = ''):

        def log_metric(group, name, value):
            writer.add_scalar(
                f'{group}{param_group_postfix}/{metric_prefix}{name}',
                value, epoch
                )

        log_metric('debug', 'update_norm', update_norm)

        log_metric('eval', 'utility', utility)
        log_metric('eval', 'utility_vs_bne', utility_vs_bne)
        log_metric('eval', 'epsilon_relative', epsilon_relative)
        log_metric('eval', 'epsilon_absolute', epsilon_absolute)
        log_metric('eval', 'L_2', L_2)
        log_metric('eval', 'L_inf', L_inf)

    # TODO: deferred until writing logger
    def _log_hyperparams(self):
        """Everything that should be logged on every learning_rate updates"""
    #     writer.add_scalar('hyperparams/batch_size', batch_size, e)
    #     writer.add_scalar('hyperparams/learning_rate', learning_rate, e)
    #     writer.add_scalar('hyperparams/momentum', momentum, e)
    #     writer.add_scalar('hyperparams/sigma', sigma, e)
    #     writer.add_scalar('hyperparams/n_perturbations', n_perturbations, e

class LLGAuctionLogger(SingleItemAuctionLogger):
    # TODO: Inherit from Logger
    def __init__(self, l_config: LearningConfiguration):
        super().__init__(None, l_config)

    #TODO: Delete!?
    def plot_bid_function(self, fig, v, b, writer=None, e=None):
        # subsample points and plot
        for i in range(len(v)):
            if i in [0,1]:
                #plot fewer points for local bidders
                v[i] = v[i].detach().cpu().numpy()[:self.plot_points]
                b[i] = b[i].detach().cpu().numpy()[:self.plot_points]
            else:
                v[i] = v[i].detach().cpu().numpy()[:self.plot_points]
                b[i] = b[i].detach().cpu().numpy()[:self.plot_points]

        fig = plt.gcf()
        plt.cla()
        plt.xlim(0, 2)
        plt.ylim(0, 2)
        plt.xlabel('valuation')
        plt.ylabel('bid')
        plt.text(0 + 0.5, 2 - 0.5, 'iteration {}'.format(e))
        plt.plot(v[0],b[0], 'bo', self.v_opt[0], self.b_opt[0], 'b--', v[1],b[1], 'go', self.v_opt[1], self.b_opt[1], 'g--', v[2],b[2], 'ro', self.v_opt[2],self.b_opt[2], 'r--')
        self._process_figure(fig, writer, e)

class LLLLGGAuctionLogger(SingleItemAuctionLogger):
    # TODO: Inherit from Logger
    def __init__(self, l_config: LearningConfiguration):
        super().__init__(None, l_config)

    def log_training_iteration(self, prev_params, epoch, strat_to_bidder, eval_batch_size, utilities, log_params: dict):
        # TODO It is by no means nice that there is so much specific logic in here
        #TODO: Change similar to single_item
        start_time = timer()
        for i, model in enumerate(self.exp.models):
            # calculate infinity-norm of update step
            new_params = torch.nn.utils.parameters_to_vector(model.parameters())
            update_norm = (new_params - prev_params[i]).norm(float('inf'))

            self._log_metrics(writer=self.writer, epoch=epoch, utility=utilities[i], update_norm=update_norm)

        if epoch % self.logging_options['plot_epoch'] == 0:
            [print("Epoch {}: \tcurrent utility: {:.3f}".format(epoch, utilities[i]))
                for i in range(len(self.exp.models))]
            self._plot(self.fig, self.exp.models, self.writer, epoch)
        elapsed = timer() - start_time
        self.overhead_mins = self.overhead_mins + elapsed / 60
        self.writer.add_scalar('debug/overhead_mins', self.overhead_mins, epoch)

    def _log_metrics(self, writer, epoch, utility, update_norm):
        writer.add_scalar('eval/utility', utility, epoch)
        writer.add_scalar('debug/norm_parameter_update', update_norm, epoch)

    #TODO: Implement
    def _plot(self, fig, models, writer: SummaryWriter or None, e=None):
        input_length = 2
        plot_points = self.plot_points
        lin_local = torch.linspace(self.experiment_params['u_lo'][0], self.experiment_params['u_hi'][0], plot_points)
        lin_global = torch.linspace(self.experiment_params['u_lo'][4], self.experiment_params['u_hi'][4], plot_points)
        xv = [None] * 2
        yv = [None] * 2
        xv[0], yv[0] = torch.meshgrid([lin_local, lin_local])
        xv[1], yv[1] = torch.meshgrid([lin_global, lin_global])
        valuations = torch.zeros(plot_points**2, len(models), input_length, device=self.gpu_config)
        models_print = [None] * len(models)
        models_print_wf = [None] * len(models)

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
            models_print_wf[model_idx] = models_print[model_idx].view(plot_points,plot_points,input_length)

        fig, plt = self._plot_3d([valuations, models_print], e, [self.plot_xmin, self.plot_xmax],
                                 [self.plot_ymin, self.plot_ymax], [self.plot_ymin, self.plot_ymax])

        self._process_figure(fig, writer, e)

    #TODO: Fix output (currently overpallping)
    def _plot_3d(self, plot_data, epoch, xlim: list, ylim: list, zlim:list=[None,None],
                 input_length=2, x_label="valuation_0", y_label="valuation_1", z_label="bid"):
        """This implements plotting simple 2d data"""
        batch_size, n_models, n_items = plot_data[0].shape
        valuations = plot_data[0]
        bids = plot_data[1]


        plot_xmin = xlim[0]
        plot_xmax = xlim[1]
        plot_ymin = ylim[0]
        plot_ymax = ylim[1]
        plot_zmin = zlim[0]
        plot_zmax = zlim[1]

        # create the plot
        fig = plt.figure()
        for model_idx in range(n_models):
            for input_idx in range(input_length):
                ax = fig.add_subplot(n_models, input_length, model_idx*input_length+input_idx+1, projection='3d')
                ax.plot_trisurf(
                    valuations[:,model_idx,0].detach().cpu().numpy(),
                    valuations[:,model_idx,1].detach().cpu().numpy(),
                    bids[model_idx][:,input_idx].reshape(batch_size).detach().cpu().numpy(),
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
                if n_models>2:
                    if model_idx < 4:
                        ax.set_xlim(plot_xmin, plot_xmax-(self.experiment_params['u_hi'][4] - self.experiment_params['u_hi'][0]))
                        ax.set_ylim(plot_ymin, plot_ymax-(self.experiment_params['u_hi'][4] - self.experiment_params['u_hi'][0]))
                        if plot_zmin==None:
                            ax.set_zlim(plot_zmin, self.experiment_params['u_hi'][0])
                        else:
                            ax.set_zlim(plot_zmin, plot_zmax)
                    else:
                        ax.set_xlim(plot_xmin, plot_xmax)
                        ax.set_ylim(plot_ymin, plot_ymax)
                        if plot_zmin==None:
                            ax.set_zlim(plot_zmin, self.experiment_params['u_hi'][4])
                        else:
                            ax.set_zlim(plot_zmin, plot_zmax)
                else:
                    if model_idx == 0:
                        ax.set_xlim(plot_xmin, plot_xmax-(self.experiment_params['u_hi'][4] - self.experiment_params['u_hi'][0]))
                        ax.set_ylim(plot_ymin, plot_ymax-(self.experiment_params['u_hi'][4] - self.experiment_params['u_hi'][0]))
                        if plot_zmin==None:
                            ax.set_zlim(plot_zmin, self.experiment_params['u_hi'][0])
                        else:
                            ax.set_zlim(plot_zmin, plot_zmax)
                    else:
                        ax.set_xlim(plot_xmin, plot_xmax)
                        ax.set_ylim(plot_ymin, plot_ymax)
                        if plot_zmin==None:
                            ax.set_zlim(plot_zmin, self.experiment_params['u_hi'][4])
                        else:
                            ax.set_zlim(plot_zmin, plot_zmax)

                ax.set_xlabel(x_label); ax.set_ylabel(y_label); ax.set_zlabel(z_label)
                ax.zaxis.set_major_locator(LinearLocator(10))
                ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
                ax.set_title('model {}, bundle {}'.format(model_idx, input_idx))
                ax.view_init(20, -135)
        fig.suptitle('iteration {}'.format(epoch), size=16)
        fig.tight_layout()

        return fig, plt

class MultiUnitAuctionLogger(Logger):
    def __init__(self, exp, base_dir, plot_epoch: int=100):
        self.colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
                       '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
                       '#bcbd22', '#17becf']
        self.colors_warm = ['maroon', 'firebrick', 'red', 'salmon',
                            'coral', 'lightsalmon', 'mistyrose', 'lightgrey',
                            'white']
        super().__init__(exp=exp, base_dir = base_dir, plot_epoch=plot_epoch)

    def log_experiment(self, run_comment, max_epochs: int):

        # TODO: rewrite to get fields from self.exp instead of parameters
        self.models = models
        self.env = env
        self.experiment_params = experiment_params
        self.gpu_config= gpu_config

        self.max_epochs = max_epochs
        if os.name == 'nt':
            raise ValueError('The run_name may not contain : on Windows!')
        run_name = time.strftime('%Y-%m-%d %a %H:%M:%S')
        if run_comment:
            run_name = run_name + ' - ' + str(run_comment)

        self.log_dir = os.path.join(self.logging_options['log_root'], self.base_dir, run_name)
        os.makedirs(self.log_dir, exist_ok=False)
        if self.logging_options['save_figure_to_disk_png']:
            os.mkdir(os.path.join(self.log_dir, 'png'))
        if self.logging_options['save_figure_to_disk_svg']:
            os.mkdir(os.path.join(self.log_dir, 'svg'))

        print('Started run. Logging to {}'.format(self.log_dir))
        self.fig = plt.figure()

        # TODO: this never shuts down the writer! (that's the memory leak probably)
        self.writer = SummaryWriter(self.log_dir, flush_secs=30)

        self._log_once()

    def log_training_iteration(self, epoch, bidders, log_params: dict):

        # ToDO I know it's ugly, this is to avoid importing the FPSBSplitAwardAuction2x2 class to use in isinstance
        # and creating a circular dependency. For sure type checking needs to be dispensed with altogether.
        is_FPSBSplitAwardAuction = "efficiency_parameter" in self.experiment_params.keys()

        # ToDo Are bounds the same for all bidders? (used just [0] in all three cases below)
        # plotting
        if epoch % self.plot_epoch == 0:
            self.plot_bid_function(
                bidders,
                log_params['optima_bid'],
                log_params['optima_bid_2'],
                epoch=epoch,
                bounds=[self.experiment_params['u_lo'][0], self.experiment_params["u_hi"][0]],
                split_award={
                    'split_award': True,
                    "efficiency_parameter": self.experiment_params['efficiency_parameter'],
                    "input_length": self.experiment_params["input_length"]
                } if is_FPSBSplitAwardAuction else None,
            )

            # if param_dict["n_items"] == 2 \
            # and param_dict["n_players"] < 4 \
            # and param_dict["exp_no"] != 6 \
            # or param_dict["exp_no"] == 2:
            #     plot_bid_function_3d(
            #         writer, e, param_dict["exp_no"],
            #         param_dict["n_items"], log_name, logdir, bidders,
            #         batch_size, device, #bounds=[param_dict["u_lo"], param_dict["u_hi"]],
            #         split_award = param_dict["exp_no"]==6,
            #         save_fig_to_disk = save_figure_to_disk
            #     )

        policy_metrics = dict()
        bne_idx = 1
        while True:
            key = "BNE{}".format(bne_idx)
            if key in self.experiment_params.keys():
                policy_metrics[key] = torch.tensor([
                    self._policy_metric(
                        model.forward,
                        log_params['optima_bid'],
                        self.experiment_params["n_items"],
                        selection={'split_award': True,
                                "efficiency_parameter": self.experiment_params['efficiency_parameter'],
                                "input_length": self.experiment_params["input_length"]
                            } if is_FPSBSplitAwardAuction else 'random',
                        bounds=[self.experiment_params["u_lo"][0], self.experiment_params["u_hi"][0]],
                        item_interest_limit=self.experiment_params["item_interest_limit"] if \
                            "item_interest_limit" in self.experiment_params.keys() else None,
                        eval_points_max=2 ** 18,
                        device = self.gpu_config.device
                    )
                    for model in self.models], device=self.gpu_config.device)
                bne_idx += 1
            else:
                break

        metrics_dict = log_params
        metrics_dict['rel_utility_loss'] = [
            1 - u / bne_u for u, bne_u
            in zip(log_params['against_bne_utilities'], log_params['bne_utilities'])
        ]
        self._log_metrics(epoch, metrics_dict)

        print('epoch {}:\t{}s'.format(epoch, round(log_params['elapsed'], 2)))

        if epoch == self.max_epochs:
            for i, model in enumerate(self.models):
                torch.save(model.state_dict(), os.path.join(self.log_dir, 'saved_model_' + str(i) + '.pt'))

    def _plot(self, fig, plot_data, writer: SummaryWriter or None, e=None):
        pass

    def _process_figure(self, fig, writer=None, epoch=None):
        pass

    def _log_once(self):
        epoch = 0
        n_parameters = self.experiment_params['n_parameters']
        for agent in range(len(self.models)):
            self.writer.add_scalar('hyperparameters/p{}_model_parameters'.format(agent),
                                   n_parameters[agent], epoch)
        self.writer.add_scalar('hyperparameters/model_parameters', sum(n_parameters), epoch)

        for i, model in enumerate(self.models):
            self.writer.add_text('hyperparameters/neural_net_spec', str(model), epoch)
            self.writer.add_graph(model, self.env.agents[i].valuations)

        self.writer.add_scalar('hyperparameters/batch_size', self.l_config.batch_size, epoch)
        self.writer.add_scalar('hyperparameters/epochs', self.max_epochs, epoch)
        self.writer.add_scalar(
            'hyperparameters/pretrain_iters',
            self.l_config.pretrain_iters,
            epoch
        )
        # TODO log seeds

        # for key, value in self.l_config.learner_hyperparams.items():
        #     self.writer.add_scalar('hyperparameters/' + str(key), value, epoch)

        self.writer.add_text('hyperparameters/optimizer', str(self.l_config.optimizer), epoch)
        for key, value in self.l_config.optimizer_hyperparams.items():
            self.writer.add_scalar('hyperparameters/' + str(key), value, epoch)

    def _log_metrics(self, epoch, metrics_dict: dict):
        """Log scalar for each player"""

        agent_name_list = ['agent_{}'.format(i) for i in range(self.experiment_params['n_players'])]

        for metric_key, metric_val in metrics_dict.items():
            if isinstance(metric_val, float):
                self.writer.add_scalar('eval/' + str(metric_key), metric_val, epoch)
            elif isinstance(metric_val, list):
                self.writer.add_scalars(
                    'eval/' + str(metric_key),
                    dict(zip(agent_name_list, metric_val)),
                    epoch
                )
            elif isinstance(metric_val, dict):
                for key, val in metric_val.items():
                    self.writer.add_scalars(
                        'eval/' + str(metric_key),
                        dict(zip([name + '/' + str(key) for name in agent_name_list], val)),
                        epoch
                    )

        # log model parameters
        model_paras = [torch.norm(torch.nn.utils.parameters_to_vector(model.parameters()), p=2)
                       for model in self.models]
        self.writer.add_scalars('eval/weight_norm', dict(zip(agent_name_list, model_paras)), epoch)

    def _log_hyperparams(self, writer, epoch):
        pass

    def _policy_metric(self, policy_1: Callable, policy_2: Callable, dim: int, bounds=[0, 1],
                       eval_points_max: int = 2 ** 7,
                       selection='random', item_interest_limit=None, dim_of_interest=None, device=None):
        """
        Calculate the p-norm on a grid between the two policy functions.

        TODO: Consider sampleing according to underlying distribution instead of
            the uniform grid sampleing!
        """
        valuations = self.multi_unit_valuations(device if device!=None else self.gpu_config.device,
                                                bounds, dim, eval_points_max, selection, item_interest_limit)

        policy_1_bidding = policy_1(valuations).detach()
        policy_2_bidding = policy_2(valuations).detach()

        if dim_of_interest is not None:
            policy_1_bidding = policy_1_bidding[:, dim_of_interest]
            policy_2_bidding = policy_2_bidding[:, dim_of_interest]

        metric = self.rmse(policy_1_bidding, policy_2_bidding)

        return metric.detach()

    @staticmethod
    def multi_unit_valuations(
            device = None,
            bounds = [0, 1],
            dim = 2,
            batch_size = 100,
            selection = 'random',
            item_interest_limit = None,
            sort = False,
        ):
        """Returns uniformly sampled valuations for multi unit auctions."""
        # for uniform vals and 2 items <=> F1(v)=v**2, F2(v)=2v-v**2

        eval_points_per_dim = round((2*batch_size) ** (1/dim))
        valuations = torch.zeros(eval_points_per_dim ** dim, dim, device=device)

        if selection == 'random':
            valuations.uniform_(bounds[0], bounds[1])
            valuations = valuations.sort(dim=1, descending=True)[0]

        elif 'split_award' in selection.keys():
            if 'linspace' in selection.keys() and selection['linspace']:
                valuations[:,0] = torch.linspace(
                    bounds[0], bounds[1],
                    eval_points_per_dim ** dim, device=device
                )
            else:
                valuations.uniform_(bounds[0], bounds[1])
            valuations[:,1] = selection['efficiency_parameter'] * valuations[:,0]
            # if 'input_length' in selection.keys():
            #     valuations = valuations[:,:selection['input_length']]

        else:
            lin = torch.linspace(bounds[0], bounds[1], eval_points_per_dim, device=device)
            mesh = torch.meshgrid([lin] * dim)
            for n in range(dim):
                valuations[:,n] = mesh[n].reshape(eval_points_per_dim ** dim)

            mask = valuations.sort(dim=1, descending=True)[0]
            mask = (mask == valuations).all(dim=1)
            valuations = valuations[mask]

        if item_interest_limit is not None:
            valuations[:,item_interest_limit:] = 0
        if sort:
            valuations = valuations.sort(dim=1)[0]

        return valuations

    # TODO: Why is this a method? why does it clone the tensors?
    def rmse(self, y, y_hat):
        """
        Root mean squared error.
        """
        return torch.sqrt(torch.mean((torch.clone(y_hat) - torch.clone(y)) ** 2))

    def plot_bid_function(self, bidders, optimal_bid, optimal_bid_2, epoch=None, format='png', bounds=[0., 1.],
                          split_award=None):
        """Method for plotting"""

        n_items = bidders[0].n_items
        n_players = len(bidders)
        plot_points = 25

        if split_award is not None:
            split_award['linspace'] = True

        valuations = self.multi_unit_valuations(
            self.gpu_config.device, bounds, n_items, plot_points,
            'random' if split_award is None else split_award
        )
        # valuations = deepcopy(bidders[0]).draw_valuations_()[:plot_points,:]

        if split_award is not None:
            valuations, _ = valuations.sort(0)

        b_opt_2 = optimal_bid_2(valuations).cpu().numpy()
        if split_award is None:
            b_opt = optimal_bid(valuations)
            b_opt = b_opt.cpu().numpy()
        else:
            b_opt = optimal_bid(valuations, return_payoff_dominant=False)
            for k, v in b_opt.items():
                b_opt[k] = v.cpu().numpy()
            temp = b_opt
            b_opt = b_opt_2
            b_opt_2 = temp

        actions = list()
        for bidder in bidders:
            try:
                dim = bidder.strategy.input_length
            except:
                try:
                    dim = n_items
                except Exception as exc:
                    print(exc)
            try:
                actions.append(bidder.strategy.play(valuations[:,:dim]))
            except:
                actions.append(bidder.strategy(valuations[:,:dim]))

        # sorting of points, s.t. 1st plot corresponds to 1st item, etc.
        # (from sorted values to sorted bids)
        acts = list()
        for act in actions:
            # if sort_by_bids:
            #     sorted_idx = torch.sort(act, dim=1, descending=True)[1]
            #     acts.append(batched_index_select(act, 1, sorted_idx).detach().cpu().numpy())
            # else:
            acts.append(act.detach().cpu().numpy())

        fig, axs = plt.subplots(nrows=1, ncols=n_items, sharey=True, figsize=[7, 4])
        plt.cla()

        if not isinstance(axs, np.ndarray): # only one item/plot
            axs = [axs]

        if split_award is not None and n_items == 2:
            if valuations.shape[1] == 1:
                valuations = torch.cat(
                    (valuations, split_award["efficiency_parameter"] * valuations), 1
                )

        valuations = valuations.cpu().numpy()

        for item in range(n_items):
            if split_award is not None:
                plot = list(reversed(range(n_items)))[item]
            else:
                plot = item

            for agent_idx in range(n_players):
                if split_award is None:
                    zeros = acts[agent_idx][:, item] < 1e-9
                    axs[plot].scatter(
                        valuations[:, item][~zeros], acts[agent_idx][:, item][~zeros],
                        marker='.',
                        color=self.colors[agent_idx % len(self.colors)],
                        label='agent ' + str(agent_idx + 1),
                    )
                    axs[plot].plot(
                        valuations[:, item][zeros], acts[agent_idx][:, item][zeros],
                        marker="x",
                        color=self.colors[agent_idx % len(self.colors)],
                    )
                else:
                    zeros = acts[agent_idx][:, item] < 1e-9
                    axs[plot].plot(
                        valuations[:, item][~zeros], acts[agent_idx][:, item][~zeros],
                        '.-',
                        color=self.colors[agent_idx % len(self.colors)],
                        label='agent ' + str(agent_idx + 1),
                    )
                    axs[plot].plot(
                        valuations[:, item][zeros], acts[agent_idx][:, item][zeros],
                        marker="x",
                        color=self.colors[agent_idx % len(self.colors)],
                    )

            axs[plot].plot(
                valuations[:, item], b_opt[:, item],
                '.' if split_award is None else '-', color='black',
                label='WTA BNE strategy' if split_award is not None
                else 'BNE strategy'
            )

            if split_award is not None:
                x_label = 'cost'
                axs[plot].yaxis.grid(which="major", linestyle=':')
                axs[plot].set_title('100% share' if item == 0 else '50% share')

                select = 'wta_bounds' if item == 0 else 'sigma_bounds'
                axs[plot].plot(
                    valuations[:, 0 if item == 0 else 1],
                    b_opt_2[select][:, 0], '--',
                    label='pooling BNE bounds',
                    color='black'
                )
                axs[plot].plot(
                    valuations[:, 0 if item == 0 else 1],
                    b_opt_2[select][:, 1], '--',
                    color='black'
                )

            if split_award is not None:
                axs[plot].set_xlim(
                    [bounds[0], bounds[1]] if item == 0 else
                    [split_award["efficiency_parameter"] * bounds[0],
                     split_award["efficiency_parameter"] * bounds[1]]
                )
                axs[plot].set_ylim(
                    [0, 1.9 * bounds[1]] if item == 0 else
                    [0, 5.2*split_award["efficiency_parameter"]*bounds[1]]
                )

            else:
                x_label = 'valuation'
                axs[plot].set_title(str(item + 1) + '. bid')
                axs[plot].set_xlim([bounds[0], bounds[1]])
                axs[plot].set_ylim([bounds[0], bounds[1]])

            axs[plot].set_xlabel(x_label)

            if plot == 0:
                axs[plot].set_ylabel('bid')
                if n_players < 10:
                    axs[plot].legend(loc='upper left')

        axs[plot].locator_params(axis='x', nbins=5)
        fig.tight_layout()

        if self.save_figure_to_disk_png and self.log_dir is not None:
            try:
                os.mkdir(os.path.join(self.log_dir, 'plots'))
            except FileExistsError:
                pass
            print(os.path.join(self.log_dir, 'plots', f'_{epoch:05}.' + format))
            plt.savefig(os.path.join(self.log_dir, 'plots', f'_{epoch:05}.' + format))

        self.writer.add_figure('plot/plot', fig, epoch)
