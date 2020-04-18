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
from bnelearn.experiment.configurations import LearningConfiguration
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
        self.learning_config = exp.learning_config
        self.experiment_config = exp.experiment_config

        # metrics
        self.log_opt = self.exp.known_bne
        self.log_regret = None
        self.log_L2 = None
        self.log_rmse = None

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

        self.overhead = 0.0

    # Doesn't seem to be needed
    # def __del__(self):
    #    self.writer.close()

    # ToDo Make a signature take a single dictionary parameter, as signatures would differ in each class
    @abstractmethod
    def log_training_iteration(self, prev_params, epoch, bne_env, strat_to_bidder,
                               utility, log_params: dict):
        pass

    # TODO: Could be moved outside as a static method
    def _plot(self, fig, plot_data, writer: SummaryWriter or None, epoch=None,
              xlim: list=None, ylim: list=None, labels: list=None,
              x_label="valuation", y_label="bid", fmts=['o'],
              figure_name: str='bid_function', plot_points=100):
        """
        This implements plotting simple 2D data.

        Args
            fig: matplotlib.figure, TODO might not be needed
            plot_data: tuple of two pytorch tensors first beeing for x axis, second for y.
                Both of dimensions (batch_size, n_strategies, n_bundles)
            writer: could be replaced by self.writer
            epoch: int, epoch or iteration number
            xlim: list of floats, x axis limits for all n_bundles dimensions
            ylim: list of floats, y axis limits for all n_bundles dimensions
            labels: list of str lables for legend
            fmts: list of str for matplotlib markers and lines
            figure_name: str, for seperate plot saving of e.g. bids and regret,
            plot_point: int of number of ploting points for each strategy in each subplot
        """

        x = plot_data[0].detach().cpu().numpy()
        y = plot_data[1].detach().cpu().numpy()
        n_batch, n_players, n_bundles = y.shape

        n_batch = min(plot_points, n_batch)
        x = x[:n_batch,:,:]
        y = y[:n_batch,:,:]

        # create the plot
        fig, axs = plt.subplots(nrows=1, ncols=n_bundles, sharey=True)
        plt.cla()
        if not isinstance(axs, np.ndarray):
            axs = [axs] # one plot only

        cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']

        # actual plotting
        for plot_idx in range(n_bundles):
            for agent_idx in range(n_players):
                axs[plot_idx].plot(
                    x[:,agent_idx,plot_idx], y[:,agent_idx,plot_idx],
                    fmts[agent_idx % len(fmts)],
                    label=None if labels is None else labels[agent_idx % len(labels)],
                    color=cycle[agent_idx],
                )

            # formating
            axs[plot_idx].set_xlabel(x_label)
            if plot_idx == 0:
                axs[plot_idx].set_ylabel(y_label)
                if n_players < 10 and labels is not None:
                    axs[plot_idx].legend(loc='upper left')
            if xlim is not None:
                axs[plot_idx].set_xlim(xlim[plot_idx][0], xlim[plot_idx][1])
            elif hasattr(self.exp, 'plot_xmin'):
                axs[plot_idx].set_xlim(self.exp.plot_xmin, self.exp.plot_xmax)
            if ylim is not None:
                axs[plot_idx].set_ylim(ylim[plot_idx][0], ylim[plot_idx][1])
            elif hasattr(self.exp, 'plot_xmin'):
                axs[plot_idx].set_ylim(self.exp.plot_ymin, self.exp.plot_ymax)

            axs[plot_idx].locator_params(axis='x', nbins=5)
        title = plt.title if n_bundles == 1 else plt.suptitle
        title('iteration {}'.format(epoch))

        self._process_figure(fig, writer=writer, epoch=epoch, figure_name=figure_name)

        return fig

    def _process_figure(self, fig, writer=None, epoch=None, figure_name='plot', group ='eval', filename=None):
        """displays, logs and/or saves figure built in plot method"""

        if not filename:
            filename = figure_name

        if self.logging_options['save_figure_to_disk_png']:
            plt.savefig(os.path.join(self.log_dir, 'png', f'{filename}_{epoch:05}.png'))

        if self.logging_options['save_figure_to_disk_svg']:
            plt.savefig(os.path.join(self.log_dir, 'svg', f'{filename}_{epoch:05}.svg'),
                        format='svg', dpi=1200)
        if writer:
            writer.add_figure(f'{group}/{figure_name}', fig, epoch)
        if self.logging_options['show_plot_inline']:
            # display.display(plt.gcf())
            plt.show()

    @abstractmethod
    def _log_experimentparams():
        pass

    @abstractmethod
    def _log_metrics(self, writer, epoch, utility, update_norm, utility_vs_bne, epsilon_relative, epsilon_absolute,
                     L_2, L_inf, param_group_postfix = '', metric_prefix = ''):
        pass

    def _log_hyperparams(self):
        """Everything that should be logged on every learning_rate updates"""
        epoch = 0
        #TODO: what is n_parameters?
        #n_parameters = self.experiment_config['n_parameters']
        #for agent in range(len(self.exp.models)):
        #    self.writer.add_scalar('hyperparameters/p{}_model_parameters'.format(agent),
        #                           n_parameters[agent], epoch)
        #self.writer.add_scalar('hyperparameters/model_parameters', sum(n_parameters), epoch)

        for i, model in enumerate(self.exp.models):
            self.writer.add_text('hyperparameters/neural_net_spec', str(model), epoch)
            self.writer.add_graph(model, self.exp.env.agents[i].valuations)

        self.writer.add_scalar('hyperparameters/batch_size', self.learning_config.batch_size, epoch)
        self.writer.add_scalar('hyperparameters/epochs', self.max_epochs, epoch)
        self.writer.add_scalar(
            'hyperparameters/pretrain_iters',
            self.learning_config.pretrain_iters,
            epoch
        )

# TODO: Allow multiple utilities and params (for multiple learners)
class SingleItemAuctionLogger(Logger):
    def __init__(self, exp, base_dir):
        super().__init__(exp, base_dir)

    def log_experiment(self, run_comment, max_epochs):
        self.max_epochs = max_epochs
        # setting up plotting
        self.plot_points = min(100, self.exp.learning_config.batch_size)

        if self.log_opt:
            # TODO: apdapt interval to be model specific! (e.g. for LLG)
            self.v_opt = torch.stack(
                [torch.linspace(self.exp.plot_xmin, self.exp.plot_xmax, self.plot_points,
                 device=self.exp.gpu_config.device) for i in self.exp.models],
                dim=1)[:,:,None]
            self.b_opt = torch.stack(
                [self.exp._optimal_bid(self.v_opt[:,i,:], player_position=self.exp._model2bidder[i][0])
                 for i in range(len(self.exp.models))],
                dim=1)

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
        start_time = timer()
        self._log_once()
        self._log_experimentparams() # TODO: what to use
        self._log_hyperparams()
        elapsed = timer() - start_time
        self.overhead += elapsed

    #TODO: Have to get bne_utilities for all models instead of bne_utoility of only one!?
    def log_training_iteration(self, prev_params, epoch, strat_to_bidder,
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

        # TODO: this logic does not work for LLG model
        for i, model in enumerate(self.exp.models):
            group_postfix = '' if model_is_global else f'_p{i}'
            metric_prefix = ''

            ## TODO: no knowledge of bneenve should be assumed here! Might have settings without bne

            # calculate infinity-norm of update step
            new_params = torch.nn.utils.parameters_to_vector(model.parameters())
            update_norm = (new_params - prev_params[i]).norm(float('inf'))

            if self.exp.known_bne:
                # calculate utility vs bne
                utility_vs_bne = self.exp.bne_env.get_reward(
                    strat_to_bidder(model, batch_size=self.exp.logging_config.eval_batch_size),
                    draw_valuations=False)  # False because expensive for normal priors
                epsilon_relative = 1 - utility_vs_bne / self.exp.bne_utilities[i]
                epsilon_absolute = self.exp.bne_utilities[i] - utility_vs_bne
                L_2 = metrics.norm_strategy_and_actions(model, self.exp.bne_env.agents[i].get_action(),
                                                        self.exp.bne_env.agents[i].valuations, 2)
                L_inf = metrics.norm_strategy_and_actions(model, self.exp.bne_env.agents[i].get_action(),
                                                          self.exp.bne_env.agents[i].valuations, float('inf'))
            else:
                utility_vs_bne, epsilon_relative, epsilon_absolute, L_2, L_inf = None,None,None,None,None

            self._log_metrics(writer=self.writer, epoch=epoch, utility=utilities[i], update_norm=update_norm,
                              utility_vs_bne=utility_vs_bne, epsilon_relative=epsilon_relative,
                              epsilon_absolute=epsilon_absolute, L_2=L_2, L_inf=L_inf,
                              param_group_postfix=group_postfix, metric_prefix=metric_prefix)

        if epoch % self.logging_options['plot_epoch'] == 0:
            bidders = [strat_to_bidder(model, self.exp.learning_config.batch_size, self.exp._model2bidder[i][0])
                       for i, model in enumerate(self.exp.models)]
            v = torch.stack([bidder.valuations for bidder in bidders], dim=1) # shape: n_batch, n_players, n_bundles
            b = torch.stack([bidder.get_action() for bidder in bidders], dim=1)
            print(
                "Epoch {}: \tcurrent utility: {:.3f},\t vs BNE: {:.3f}, \tepsilon (abs/rel): ({:.5f}, {:.5f})".format(
                    epoch, utilities[i], utility_vs_bne, epsilon_absolute, epsilon_relative))

            labels = ['NPGA']
            fmts = ['bo']
            if self.log_opt:
                # TODO: handle case of no opt strategy
                v = torch.cat([v[:self.plot_points,:,:], self.v_opt], dim=1)
                b = torch.cat([b[:self.plot_points,:,:], self.b_opt], dim=1)
                labels.append('BNE')
                fmts.append('b--')
                
            self._plot(fig=self.fig, plot_data=(v, b), writer=self.writer, figure_name='bid_function',
                       epoch=epoch, labels=labels, fmts=fmts)

        elapsed = timer() - start_time
        self.overhead = self.overhead + elapsed
        self.writer.add_scalar('debug/overhead_hours', self.overhead/3600, epoch)


    # TODO: rename u_lo, u_hi --> these have NOTHING to do with normal distribution.
    def log_ex_interim_regret(self, epoch, mechanism, env, learners, u_lo, u_hi, regret_batch_size, regret_grid_size):
        start_time = timer()

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
            regret = metrics.ex_interim_regret(mechanism, bid_profile, player_position,
                                               env.agents[player_position].valuations, regret_grid)

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

        # TODO: as tensors in first place?
        valuations = torch.stack(valuations, dim=1)
        regrets = torch.stack(regrets, dim=1)[:,:,None]

        self.fig = self._plot(
            fig=self.fig, plot_data=(valuations, regrets), writer=self.writer,
            epoch=epoch, xlim=[self.exp.plot_xmin, self.exp.plot_xmax],
            ylim=[0, max_regret.detach().cpu().numpy()],
            x_label="valuation", y_label="regret", figure_name='regret'
        )

        # TODO: why? don't we redraw valuations at beginning of loop anyway.
        for agent in env.agents:
            agent.batch_size = original_batch_size
            agent.draw_valuations_new_batch_(original_batch_size)

        elapsed = timer() - start_time
        self.overhead += elapsed

    # Setup logging
    def _log_once(self):
        """Everything that should be logged only once on initialization."""
        # writer.add_scalar('debug/total_model_parameters', n_parameters, epoch)
        # writer.add_text('hyperparams/neural_net_spec', str(self.model), 0)
        # writer.add_scalar('debug/eval_batch_size', eval_batch_size, epoch)
        self.writer.add_graph(self.exp.models[0], self.exp.env.agents[0].valuations)

    def _log_experimentparams(self):
        #TODO: write out all experiment params (complete dict)
        pass

    def _log_trained_model(self):
        #TODO: write out the trained model at the end of training @Stefan
        pass

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

        self.writer.add_text('hyperparameters/optimizer', str(self.learning_config.optimizer), epoch)
        for key, value in self.learning_config.optimizer_hyperparams.items():
            self.writer.add_scalar('hyperparameters/' + str(key), value, epoch)


class LLGAuctionLogger(SingleItemAuctionLogger):

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
        plt.xlim(self.exp.plot_xmin, self.exp.plot_xmax)
        plt.ylim(self.exp.plot_ymin, self.exp.plot_ymax)
        plt.xlabel('valuation')
        plt.ylabel('bid')
        plt.text(0 + 0.5, 2 - 0.5, 'iteration {}'.format(e))

        if self.exp.known_bne:
            plt.plot(v[0],b[0],'bo',   self.v_opt[0],self.b_opt[0],'b--', 
                     v[1],b[1],'go',   self.v_opt[1],self.b_opt[1],'g--', 
                     v[2],b[2],'ro',   self.v_opt[2],self.b_opt[2],'r--')
        else:
            plt.plot(v[0],b[0],'bo',   v[1],b[1],'go',   v[2],b[2],'ro')

        self._process_figure(fig, writer, e, figure_name='bid_function')


class LLLLGGAuctionLogger(SingleItemAuctionLogger):

    def log_training_iteration(self, prev_params, epoch, strat_to_bidder, utilities, log_params: dict):
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
        self.overhead = self.overhead + elapsed
        self.writer.add_scalar('debug/overhead_hours', self.overhead/3600, epoch)

    def _log_metrics(self, writer, epoch, utility, update_norm):
        writer.add_scalar('eval/utility', utility, epoch)
        writer.add_scalar('debug/norm_parameter_update', update_norm, epoch)

    #TODO: Implement
    def _plot(self, fig, models, writer: SummaryWriter or None, e=None):
        input_length = 2
        plot_points = self.plot_points
        lin_local = torch.linspace(self.exp.u_lo[0], self.exp.u_hi[0], plot_points)
        lin_global = torch.linspace(self.exp.u_lo[4], self.exp.u_hi[4], plot_points)
        xv = [None] * 2
        yv = [None] * 2
        xv[0], yv[0] = torch.meshgrid([lin_local, lin_local])
        xv[1], yv[1] = torch.meshgrid([lin_global, lin_global])
        valuations = torch.zeros(plot_points**2, len(models), input_length, device=self.exp.gpu_config.device)
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

        fig, plt = self._plot_3d([valuations, models_print], e, [self.exp.plot_xmin, self.exp.plot_xmax],
                                 [self.exp.plot_ymin, self.exp.plot_ymax], [self.exp.plot_ymin, self.exp.plot_ymax])

        self._process_figure(fig, writer, e, figure_name='bid_functions')

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
                        ax.set_xlim(plot_xmin, plot_xmax-(self.experiment_config.u_hi[4] - self.experiment_config.u_hi[0]))
                        ax.set_ylim(plot_ymin, plot_ymax-(self.experiment_config.u_hi[4] - self.experiment_config.u_hi[0]))
                        if plot_zmin==None:
                            ax.set_zlim(plot_zmin, self.experiment_config.u_hi[0])
                        else:
                            ax.set_zlim(plot_zmin, plot_zmax)
                    else:
                        ax.set_xlim(plot_xmin, plot_xmax)
                        ax.set_ylim(plot_ymin, plot_ymax)
                        if plot_zmin==None:
                            ax.set_zlim(plot_zmin, self.experiment_config.u_hi[4])
                        else:
                            ax.set_zlim(plot_zmin, plot_zmax)
                else:
                    if model_idx == 0:
                        ax.set_xlim(plot_xmin, plot_xmax-(self.experiment_config.u_hi[4] - self.experiment_config.u_hi[0]))
                        ax.set_ylim(plot_ymin, plot_ymax-(self.experiment_config.u_hi[4] - self.experiment_config.u_hi[0]))
                        if plot_zmin==None:
                            ax.set_zlim(plot_zmin, self.experiment_config.u_hi[0])
                        else:
                            ax.set_zlim(plot_zmin, plot_zmax)
                    else:
                        ax.set_xlim(plot_xmin, plot_xmax)
                        ax.set_ylim(plot_ymin, plot_ymax)
                        if plot_zmin==None:
                            ax.set_zlim(plot_zmin, self.experiment_config.u_hi[4])
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
    def __init__(self, exp, base_dir, save_figure_to_disk_png: bool = True,
                 save_figure_to_disk_svg: bool = True, plot_epoch: int = 100,
                 show_plot_inline: bool = True, save_figure_data_to_disk: bool = False):
        super().__init__(exp, base_dir, save_figure_to_disk_png, save_figure_to_disk_svg,
                         plot_epoch, show_plot_inline, save_figure_data_to_disk)

        self.models = self.exp.models
        self.env = self.exp.env
        self.experiment_config = self.exp.experiment_config
        self.gpu_config = self.exp.gpu_config

    def log_experiment(self, run_comment, max_epochs):
        self.max_epochs = max_epochs

        self.plot_points = min(100, self.exp.learning_config.batch_size)
        if self.log_opt:
            self.v_opt = torch.stack([
                torch.linspace(self.exp.plot_xmin, self.exp.plot_xmax, self.plot_points,
                               device=self.exp.gpu_config.device)
                ] * self.exp.n_units,
                dim = 1
            )
            self.b_opt = self.exp._optimal_bid(self.v_opt) # only one sym. BNE supported

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
        self._log_experimentparams()
        self._log_hyperparams()

    def log_training_iteration(self, epoch, bidders, log_params: dict):

        valuations = list()
        bids = list()
        for bidder in bidders:
            valuations.append(bidder.draw_valuations_())
            bids.append(bidder.get_action())
        valuations = torch.stack(valuations, dim=1)
        bids = torch.stack(bids, dim=1)

        if self.log_opt:
            valuations = torch.cat([
                valuations[:self.plot_points,:,:], self.v_opt[:,None,:],
            ], dim=1)
            bids = torch.cat([
                bids[:self.plot_points,:,:], self.b_opt[:,None,:],
            ], dim=1)

        # plotting
        if epoch % self.plot_epoch == 0:
            labels = ['NPGA'] * len(bidders)
            fmts = ['bo'] * len(bidders)
            if self.log_opt:
                labels.append('BNE')
                fmts.append('b--')
            from bnelearn.experiment.multi_unit_experiment import SplitAwardExperiment
            if isinstance(self.exp, SplitAwardExperiment):
                xlim = [
                    [self.exp.u_lo[0], self.exp.u_hi[0]],
                    [self.exp.efficiency_parameter * self.exp.u_lo[0],
                     self.exp.efficiency_parameter * self.exp.u_hi[0]]
                ]
                ylim = [
                    [0, 2 * self.exp.u_hi[0]],
                    [0, 2 * self.exp.u_hi[0]]
                ]
            else:
                xlim = ylim = None
            super()._plot(fig=self.fig, plot_data=(valuations, bids), writer=self.writer, xlim=xlim, ylim=ylim,
                          figure_name='bid_function', epoch=epoch, labels=labels, fmts=fmts)

        # TODO: dim_of_interest for multiple BNE
        log_params['rmse'] = {
            'BNE1': torch.tensor(
                [bnelearn.util.metrics.norm_actions(bids[:,i,:], self.exp._optimal_bid(valuations[:,i,:]))
                 for i, model in enumerate(self.models)]
            )
        }

        log_params['rel_utility_loss'] = [
            1 - u / bne_u for u, bne_u
            in zip(log_params['against_bne_utilities'], log_params['bne_utilities'])
        ]
        self._log_metrics(epoch, log_params)

        print('epoch {}:\t{}s'.format(epoch, round(log_params['elapsed'], 2)))

        # TODO: unify model saving via switch
        if epoch == self.max_epochs:
            for i, model in enumerate(self.models):
                torch.save(model.state_dict(), os.path.join(self.log_dir, 'saved_model_' + str(i) + '.pt'))

    def _log_experimentparams(self):
        #TODO: write out all experiment params (complete dict)
        pass

    def _log_metrics(self, epoch, metrics_dict: dict):
        """Log scalar for each player"""

        agent_name_list = ['agent_{}'.format(i) for i in range(self.experiment_config.n_players)]

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
