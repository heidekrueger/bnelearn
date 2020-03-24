import os
import sys
import time
from abc import ABC, abstractmethod
from timeit import default_timer as timer
import bnelearn.util.metrics as metrics
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.tensorboard import SummaryWriter


class Logger(ABC):
    def __init__(self, save_figure_to_disc_png: bool = True, save_figure_to_disc_svg: bool = True,
                 plot_epoch: int = 10, show_plot_inline: bool = True, save_figure_data_to_dis: bool = False):
        root_path = os.path.join(os.path.expanduser('~'), 'bnelearn')
        if root_path not in sys.path:
            sys.path.append(root_path)

        self.logging_options = dict(
            log_root=os.path.join(root_path, 'experiments'),
            save_figure_to_disc_png=save_figure_to_disc_png,
            save_figure_to_disc_svg=save_figure_to_disc_svg,  # for publishing. better quality but a pain to work with
            plot_epoch=plot_epoch,
            show_plot_inline=show_plot_inline
        )

        self.base_dir = None
        self.log_dir = None
        self.fig = None
        self.model = None
        self.env = None
        self.writer = None

        # plotting
        self.plot_points = None
        self.v_opt = None
        self.b_opt = None
        self.plot_xmin = None
        self.plot_xmax = None
        self.plot_ymin = None
        self.plot_ymax = None

        self.overhead_mins = 0.0

    # Doesn't seem to be needed
    # def __del__(self):
    #    self.writer.close()

    @abstractmethod
    def log_experiment(self, model, env, run_comment, plot_xmin, plot_xmax, plot_ymin, plot_ymax, batch_size,
                       optimal_bid):
        pass

    @abstractmethod
    def log_training_iteration(self, prev_params, epoch, bne_env, strat_to_bidder, eval_batch_size, bne_utility,
                               bidders, utility):
        pass

    @abstractmethod
    def _plot(self, fig, plot_data, writer: SummaryWriter or None, e=None):
        """This method should implement a vizualization of the experiment at the current state"""
        pass

    @abstractmethod
    def _process_figure(self, fig, writer=None, epoch=None):
        """displays, logs and/or saves figure built in plot method"""
        pass

    @abstractmethod
    def _log_once(self, writer, epoch):
        """Everything that should be logged only once on initialization."""
        pass

    @abstractmethod
    def _log_metrics(self, writer, epoch, utility, update_norm, utility_vs_bne, epsilon_relative, epsilon_absolute,
                     L_2, L_inf):
        pass

    @abstractmethod
    def _log_hyperparams(self, writer, epoch):
        pass


class SingleItemAuctionLogger(Logger):
    def __init__(self):
        super().__init__()

    def log_experiment(self, model, env, run_comment, plot_xmin, plot_xmax, plot_ymin, plot_ymax, batch_size,
                       optimal_bid, player_position=0):
        # setting up plotting
        self.plot_xmin = plot_xmin
        self.plot_xmax = plot_xmax
        self.plot_ymin = plot_ymin
        self.plot_ymax = plot_ymax
        self.plot_points = min(150, batch_size)
        self.v_opt = np.linspace(plot_xmin, plot_xmax, 100)
        self.b_opt = optimal_bid(self.v_opt, player_position=player_position)

        is_ipython = 'inline' in plt.get_backend()
        if is_ipython:
            from IPython import display
        plt.rcParams['figure.figsize'] = [8, 5]

        # TODO: This should rather be represented as a list and plotting all models in that list

        self.model = model
        self.env = env

        if os.name == 'nt':
            raise ValueError('The run_name may not contain : on Windows!')
        run_name = time.strftime('%Y-%m-%d %a %H:%M:%S')
        if run_comment:
            run_name = run_name + ' - ' + str(run_comment)

        self.log_dir = os.path.join(self.logging_options['log_root'], self.base_dir, run_name)
        os.makedirs(self.log_dir, exist_ok=False)
        if self.logging_options['save_figure_to_disc_png']:
            os.mkdir(os.path.join(self.log_dir, 'png'))
        if self.logging_options['save_figure_to_disc_svg']:
            os.mkdir(os.path.join(self.log_dir, 'svg'))

        print('Started run. Logging to {}'.format(self.log_dir))
        self.fig = plt.figure()

        self.writer = SummaryWriter(self.log_dir, flush_secs=30)
        self._log_once(self.writer, 0)
        self._log_hyperparams(self.writer, 0)

    def log_training_iteration(self, prev_params, epoch, bne_env, strat_to_bidder, eval_batch_size, bne_utility,
                               bidders, utility):
        # ToDO It is by no means nice that there is so much specific logic in here
        start_time = timer()

        # calculate infinity-norm of update step
        new_params = torch.nn.utils.parameters_to_vector(self.model.parameters())
        update_norm = (new_params - prev_params).norm(float('inf'))
        # calculate utility vs bne
        utility_vs_bne = bne_env.get_reward(
            strat_to_bidder(self.model, batch_size=eval_batch_size),
            draw_valuations=False)  # False because expensive for normal priors
        epsilon_relative = 1 - utility_vs_bne / bne_utility
        epsilon_absolute = bne_utility - utility_vs_bne
        L_2 = metrics.norm_strategy_and_actions(self.model, bne_env.agents[0].get_action(),
                                                bne_env.agents[0].valuations, 2)
        L_inf = metrics.norm_strategy_and_actions(self.model, bne_env.agents[0].get_action(),

                                                  bne_env.agents[0].valuations, float('inf'))
        self._log_metrics(writer=self.writer, epoch=epoch, utility=utility, update_norm=update_norm,
                          utility_vs_bne=utility_vs_bne, epsilon_relative=epsilon_relative,
                          epsilon_absolute=epsilon_absolute, L_2=L_2, L_inf=L_inf)

        if epoch % self.logging_options['plot_epoch'] == 0:
            # plot current function output
            # bidder = strat_to_bidder(model, batch_size)
            # bidder.draw_valuations_()
            v = bidders[0].valuations
            b = bidders[0].get_action()
            plot_data = (v, b)

            print(
                "Epoch {}: \tcurrent utility: {:.3f},\t utility vs BNE: {:.3f}, \tepsilon (abs/rel): ({:.5f}, {:.5f})".format(
                    epoch, utility, utility_vs_bne, epsilon_absolute, epsilon_relative))
            self._plot(self.fig, plot_data, self.writer, epoch)

        elapsed = timer() - start_time

        self.overhead_mins = self.overhead_mins + elapsed / 60
        self.writer.add_scalar('debug/overhead_mins', self.overhead_mins, epoch)

    def _plot(self, fig, plot_data, writer: SummaryWriter or None, e=None):
        """This method should implement a vizualization of the experiment at the current state"""
        v, b = plot_data
        v = v.detach().cpu().numpy()[:self.plot_points]
        b = b.detach().cpu().numpy()[:self.plot_points]

        # create the plot
        fig = plt.gcf()
        plt.cla()
        plt.xlim(self.plot_xmin, self.plot_xmax)
        plt.ylim(self.plot_ymin, self.plot_ymax)
        plt.xlabel('valuation')
        plt.ylabel('bid')
        plt.text(self.plot_xmin + 0.05 * (self.plot_xmax - self.plot_xmin),
                 self.plot_ymax - 0.05 * (self.plot_ymax - self.plot_ymin),
                 'iteration {}'.format(e))
        plt.plot(v, b, 'o', self.v_opt, self.b_opt, 'r--')

        # show and/or log
        self._process_figure(fig, writer, e)

    def _process_figure(self, fig, writer=None, epoch=None):
        """displays, logs and/or saves figure built in plot method"""

        if self.logging_options['save_figure_to_disc_png']:
            plt.savefig(os.path.join(self.log_dir, 'png', f'epoch_{epoch:05}.png'))

        if self.logging_options['save_figure_to_disc_svg']:
            plt.savefig(os.path.join(self.log_dir, 'svg', f'epoch_{epoch:05}.svg'),
                        format='svg', dpi=1200)
        if writer:
            writer.add_figure('eval/bid_function', fig, epoch)
        if self.logging_options['show_plot_inline']:
            # display.display(plt.gcf())
            plt.show()

    # Setup logging
    def _log_once(self, writer, epoch):
        """Everything that should be logged only once on initialization."""
        # writer.add_scalar('debug/total_model_parameters', n_parameters, epoch)
        # writer.add_text('hyperparams/neural_net_spec', str(self.model), 0)
        # writer.add_scalar('debug/eval_batch_size', eval_batch_size, epoch)
        writer.add_graph(self.model, self.env.agents[0].valuations)

    def _log_metrics(self, writer, epoch, utility, update_norm, utility_vs_bne, epsilon_relative, epsilon_absolute,
                     L_2, L_inf):
        writer.add_scalar('eval/utility', utility, epoch)
        writer.add_scalar('debug/norm_parameter_update', update_norm, epoch)
        writer.add_scalar('eval/utility_vs_bne', utility_vs_bne, epoch)
        writer.add_scalar('eval/epsilon_relative', epsilon_relative, epoch)
        writer.add_scalar('eval/epsilon_absolute', epsilon_absolute, epoch)
        writer.add_scalar('eval/L_2', L_2, epoch)
        writer.add_scalar('eval/L_inf', L_inf, epoch)

    # TODO: deferred until writing logger
    def _log_hyperparams(self, writer, epoch):
        """Everything that should be logged on every learning_rate updates"""

    #     writer.add_scalar('hyperparams/batch_size', batch_size, e)
    #     writer.add_scalar('hyperparams/learning_rate', learning_rate, e)
    #     writer.add_scalar('hyperparams/momentum', momentum, e)
    #     writer.add_scalar('hyperparams/sigma', sigma, e)
    #     writer.add_scalar('hyperparams/n_perturbations', n_perturbations, e)


class MultiUnitAuctionLogger(Logger):
    def __init__(self):
        super().__init__()

    def log_experiment(self, model, env, run_comment, plot_xmin, plot_xmax, plot_ymin, plot_ymax, batch_size,
                       optimal_bid):
        if os.name == 'nt':
            raise ValueError('The run_name may not contain : on Windows!')
        run_name = time.strftime('%Y-%m-%d %a %H:%M:%S')
        if run_comment:
            run_name = run_name + ' - ' + str(run_comment)

        self.log_dir = os.path.join(self.logging_options['log_root'], self.base_dir, run_name)
        os.makedirs(self.log_dir, exist_ok=False)
        if self.logging_options['save_figure_to_disc_png']:
            os.mkdir(os.path.join(self.log_dir, 'png'))
        if self.logging_options['save_figure_to_disc_svg']:
            os.mkdir(os.path.join(self.log_dir, 'svg'))









    def log_training_iteration(self, prev_params, epoch, bne_env, strat_to_bidder, eval_batch_size, bne_utility,
                               bidders, utility):
        pass

    def _plot(self, fig, plot_data, writer: SummaryWriter or None, e=None):
        pass

    def _process_figure(self, fig, writer=None, epoch=None):
        pass

    def _log_once(self, writer, epoch):
        pass

    def _log_metrics(self, writer, epoch, utility, update_norm, utility_vs_bne, epsilon_relative, epsilon_absolute, L_2,
                     L_inf):
        pass

    def _log_hyperparams(self, writer, epoch):
        pass




class CombinatorialAuctionLogger(Logger):
    def log_experiment(self, model, env, run_comment, plot_xmin, plot_xmax, plot_ymin, plot_ymax, batch_size,
                       optimal_bid):
        pass

    def log_training_iteration(self, prev_params, epoch, bne_env, strat_to_bidder, eval_batch_size, bne_utility,
                               bidders, utility):
        pass

    def _plot(self, fig, plot_data, writer: SummaryWriter or None, e=None):
        pass

    def _process_figure(self, fig, writer=None, epoch=None):
        pass

    def _log_once(self, writer, epoch):
        pass

    def _log_metrics(self, writer, epoch, utility, update_norm, utility_vs_bne, epsilon_relative, epsilon_absolute, L_2,
                     L_inf):
        pass

    def _log_hyperparams(self, writer, epoch):
        pass

    def __init__(self):
        super().__init__()
