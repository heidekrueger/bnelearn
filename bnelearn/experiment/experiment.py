"""
This module defines an experiment. It includes logging and plotting since they
can often be shared by specific experiments.
"""

import os
import time
from abc import ABC, abstractmethod
from time import perf_counter as timer
from typing import Iterable, List
from collections import deque

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.ticker import FormatStrFormatter, LinearLocator

from mpl_toolkits.mplot3d import Axes3D  # pylint: disable=unused-import

from torch.utils.tensorboard import SummaryWriter

import bnelearn.util.logging as logging_utils
import bnelearn.util.metrics as metrics
from bnelearn.bidder import Bidder
from bnelearn.environment import AuctionEnvironment, Environment
from bnelearn.experiment.configurations import (ExperimentConfig)
from bnelearn.learner import ESPGLearner, Learner
from bnelearn.mechanism import Mechanism
from bnelearn.strategy import NeuralNetStrategy


# pylint: disable=unnecessary-pass,unused-argument

class Experiment(ABC):
    """Abstract Class representing an experiment"""

    # READ THIS BEFORE TOUCHING THIS SECTION:
    # We are abusing CLASS attributes here to trick the IDE into thinking that INSTANCE attributes exist in the base
    # class. In reality, we set these as INSTANCE attributes in the subclass __init__s or during runtime and our
    # logic guarantees that they will always exist as instance attributes when required.
    # This greatly simplifies readability of the __init__ implementations as it allows simplification of order of calls
    # reduces numbers of passed parameters and avoids repetition of similar concepts in subclasses.
    # DO NOT PUT MUTABLE fields (i.e. instantiated Lists here), otherwise mutating them will hold for all objects
    # of experiment.
    # Make sure everything set here is set to NotImplemented. The actual CLASS attributes should never be accessed!

    # attributes required for general setup logic
    _bidder2model: List[int]  # a list matching each bidder to their Strategy
    n_models: int
    n_items: int
    mechanism: Mechanism
    positive_output_point: torch.Tensor  # shape must be valid model input
    input_length: int

    ## Fields required for plotting
    plot_xmin: float
    plot_xmax: float
    plot_ymin: float
    plot_ymax: float
    _max_util_loss: float
    ## Optional - set only in some settings

    ## Equilibrium environment
    bne_utilities: torch.Tensor or List[float]  # dimension: n_players
    bne_env: AuctionEnvironment or List[AuctionEnvironment]
    _optimal_bid: callable

    def __init__(self, config: ExperimentConfig):
        # Configs, params are duplicated for the ease of usage and brevity
        self.config = config
        self.running = config.running
        self.setting = config.setting
        self.learning = config.learning
        self.logging = config.logging
        self.hardware = config.hardware

        # Global Stuff that should be initiated here
        self.plot_frequency = self.logging.plot_frequency
        self.plot_points = min(self.logging.plot_points, self.learning.batch_size)

        # Everything that will be set up per run initiated with none
        self.run_log_dir = None
        self.writer = None
        self.overhead = 0.0

        self.models: Iterable[torch.nn.Module] = None
        self.bidders: Iterable[Bidder] = None
        self.env: Environment = None
        self.learners: Iterable[Learner] = None

        # These are set on first _log_experiment
        self.v_opt: torch.Tensor = None
        self.b_opt: torch.Tensor = None

        self._hparams_metrics = {}
        ### Save locally - can haves
        # Logging
        if self.logging.util_loss_batch_size is not None:
            self.util_loss_batch_size = self.logging.util_loss_batch_size
        if self.logging.util_loss_grid_size is not None:
            self.util_loss_grid_size = self.logging.util_loss_grid_size
        self.n_parameters = None

        # The following required attrs have already been set in many subclasses in earlier logic.
        # Only set here if they haven't. Don't overwrite.
        if not hasattr(self, 'n_players'):
            self.n_players = self.setting.n_players
        if not hasattr(self, 'payment_rule'):
            self.payment_rule = self.setting.payment_rule
        if not hasattr(self, 'correlation_groups'):
            # TODO Stefan: quick hack, only works properly for LLG
            self.correlation_groups = None
            self.correlation_devices = None

        # sets log dir for experiment. Individual runs will log to subdirectories of this.
        self.experiment_log_dir = os.path.join(self.logging.log_root_dir,
                                               self._get_logdir_hierarchy(),
                                               self.logging.experiment_dir)

        ### actual logic
        # Inverse of bidder --> model lookup table
        self._model2bidder: List[List[int]] = [[] for m in range(self.n_models)]
        for b_id, m_id in enumerate(self._bidder2model):
            self._model2bidder[m_id].append(b_id)
        self._model_names = self._get_model_names()

        self._setup_mechanism()

        self.known_bne = self._check_and_set_known_bne()
        if self.known_bne:
            self._setup_eval_environment()
        else:
            self.logging.log_metrics['opt'] = False

    @abstractmethod
    def _setup_mechanism(self):
        pass

    # TODO: move entire name/dir logic out of logger into run. Assigned to Stefan
    @abstractmethod
    def _get_logdir_hierarchy(self):
        pass

    def _get_model_names(self):
        """Returns a list of names of models for use in logging.
        Defaults to agent{ids of agents that use the model} but may be overwritten by subclasses.
        """
        if self.n_models == 1:
            return []
        return ['bidder' + str(bidders[0]) if len(bidders) == 1 else
                'bidders' + ''.join([str(b) for b in bidders])
                for bidders in self._model2bidder]

    @abstractmethod
    def _strat_to_bidder(self, strategy, batch_size, player_position=None, cache_actions=False):
        pass

    def _setup_learners(self):

        self.learners = [
            ESPGLearner(model=model,
                        environment=self.env,
                        hyperparams=self.learning.learner_hyperparams,
                        optimizer_type=self.learning.optimizer,
                        optimizer_hyperparams=self.learning.optimizer_hyperparams,
                        strat_to_player_kwargs={"player_position": self._model2bidder[m_id][0]})
            for m_id, model in enumerate(self.models)]

    def _setup_bidders(self):
        """
        1. Create and save the models and bidders
        2. Save the model parameters
        """
        print('Setting up bidders...')
        # this method is part of the init workflow, so we #pylint: disable=attribute-defined-outside-init
        self.models = [None] * self.n_models

        for i in range(len(self.models)):
            self.models[i] = NeuralNetStrategy(
                self.input_length,
                hidden_nodes=self.learning.hidden_nodes,
                hidden_activations=self.learning.hidden_activations,
                ensure_positive_output=self.positive_output_point,
                output_length=self.n_items
            ).to(self.hardware.device)

        self.bidders = [
            self._strat_to_bidder(self.models[m_id], batch_size=self.learning.batch_size, player_position=i)
            for i, m_id in enumerate(self._bidder2model)]

        self.n_parameters = [sum([p.numel() for p in model.parameters()]) for model in
                             self.models]

        if self.learning.pretrain_iters > 0:
            print('Pretraining...')

            if hasattr(self, 'pretrain_transform'):
                pretrain_transform = self.pretrain_transform  # pylint: disable=no-member
            else:
                pretrain_transform = None

            for i, model in enumerate(self.models):
                model.pretrain(self.bidders[self._model2bidder[i][0]].valuations,
                               self.learning.pretrain_iters, pretrain_transform)

    def _check_and_set_known_bne(self):
        """Checks whether a bne is known for this experiment and sets the corresponding
           `_optimal_bid` function.
        """
        print("No BNE was found for this experiment.")
        return False

    def _setup_eval_environment(self):
        """Overwritten by subclasses with known BNE.
        Sets up an environment used for evaluation of learning agents (e.g.) vs known BNE"""
        raise NotImplementedError("This Experiment has no implemented BNE. No eval env was created.")

    def _setup_learning_environment(self):
        print(f'Learning env correlation {self.correlation_groups}: {self.correlation_devices}.')
        self.env = AuctionEnvironment(self.mechanism,
                                      agents=self.bidders,
                                      batch_size=self.learning.batch_size,
                                      n_players=self.n_players,
                                      strategy_to_player_closure=self._strat_to_bidder,
                                      correlation_groups=self.correlation_groups,
                                      correlation_devices=self.correlation_devices)

    def _init_new_run(self):
        """Setup everything that is specific to an individual run, including everything nondeterministic"""
        self._setup_bidders()
        self._setup_learning_environment()
        self._setup_learners()

        output_dir = self.run_log_dir

        if self.logging.log_metrics['opt'] and hasattr(self, 'bne_env'):

            if not isinstance(self.bne_env, list):
                # TODO Nils: should perhaps always be a list, even when there is only one BNE
                self.bne_env: List[Environment] = [self.bne_env]
                self._optimal_bid: List[callable] = [self._optimal_bid]
                self.bne_utilities = [self.bne_utilities]

            # set up list for (multiple) BNE valuations and bids
            self.v_opt = [None] * len(self.bne_env)
            self.b_opt = [None] * len(self.bne_env)

            for i, bne_env in enumerate(self.bne_env):
                # dim: [points, bidders, items]
                self.v_opt[i] = torch.stack(
                    [bidder.get_valuation_grid(n_points=self.plot_points)
                     for bidder in [bne_env.agents[j[0]] for j in self._model2bidder]],
                    dim=1
                )
                self.b_opt[i] = torch.stack(
                    [self._optimal_bid[i](self.v_opt[i][:, m, :], player_position=bidder[0])
                     for m, bidder in enumerate(self._model2bidder)],
                    dim=1
                )
                if self.v_opt[i].shape[0] != self.plot_points:
                    print('´plot_points´ changed due to get_valuation_grid')
                    self.plot_points = self.v_opt[i].shape[0]

        is_ipython = 'inline' in plt.get_backend()
        if is_ipython:
            from IPython import display  # pylint: disable=unused-import,import-outside-toplevel
        plt.rcParams['figure.figsize'] = [8, 5]

        if self.logging.enable_logging:
            os.makedirs(output_dir, exist_ok=False)
            if self.logging.save_figure_to_disk_png:
                os.mkdir(os.path.join(output_dir, 'png'))
            if self.logging.save_figure_to_disk_svg:
                os.mkdir(os.path.join(output_dir, 'svg'))
            if self.logging.save_models:
                os.mkdir(os.path.join(output_dir, 'models'))

            print('Started run. Logging to {}.'.format(output_dir))
            self.fig = plt.figure()  # pylint: disable=attribute-defined-outside-init
            self.writer = logging_utils.CustomSummaryWriter(output_dir, flush_secs=30)

            tic = timer()
            self._log_experiment_params()
            self._log_hyperparams()

            self._log_experiment_params()
            logging_utils.save_experiment_config(self.experiment_log_dir, self.config)
            elapsed = timer() - tic
            logging_utils.log_git_commit_hash(self.experiment_log_dir)
        else:
            print('Logging disabled.')
            elapsed = 0
        self.overhead += elapsed

    def _exit_run(self):
        """Cleans up a run after it is completed"""
        if self.logging.enable_logging and self.logging.save_models:
            self._save_models(directory=self.run_log_dir)

        del self.writer  # make this explicit to force cleanup and closing of tb-logfiles
        self.writer = None

        if self.hardware.cuda:
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        # if torch.cuda.memory_allocated() > 0:
        #    warnings.warn("There's a memory leak")

    def _training_loop(self, epoch):
        """Actual training in each iteration."""
        tic = timer()
        # save current params to calculate update norm
        prev_params = [torch.nn.utils.parameters_to_vector(model.parameters())
                       for model in self.models]

        # update model
        utilities = torch.tensor([
            learner.update_strategy_and_evaluate_utility()
            for learner in self.learners
        ])

        if self.logging.enable_logging:
            # pylint: disable=attribute-defined-outside-init
            self._cur_epoch_log_params = {'utilities': utilities, 'prev_params': prev_params}
            elapsed_overhead = self._evaluate_and_log_epoch(epoch=epoch)
            print('epoch {}:\telapsed {:.2f}s, overhead {:.3f}s'.format(epoch, timer() - tic, elapsed_overhead),
                  end="\r")
        else:
            print('epoch {}:\telapsed {:.2f}s'.format(epoch, timer() - tic),
                  end="\r")
        return utilities

    def run(self):
        """Runs the experiment implemented by this class for `epochs` number of iterations."""
        if not self.running.seeds:
            self.running.seeds = list(range(self.running.n_runs))

        assert sum(1 for _ in self.running.seeds) == self.running.n_runs, \
            "Number of seeds doesn't match number of runs."

        for run_id, seed in enumerate(self.running.seeds):
            print(f'\nRunning experiment {run_id} (using seed {seed})')

            try:
                self.run_log_dir = os.path.join(
                    self.experiment_log_dir,
                    f'{run_id:02d} ' + time.strftime('%T ') + str(seed))
                torch.random.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
                np.random.seed(seed)
                if self.logging.stopping_criterion_rel_util_loss_diff:
                    # stopping_list = np.empty((self.n_models,0))
                    stopping_criterion_length = self.logging.stopping_criterion_duration
                    stopping_queue = deque(maxlen=stopping_criterion_length)
                    stopping_criterion_batch_size = min(self.logging.util_loss_batch_size,
                                                        self.logging.stopping_criterion_batch_size)
                    stopping_criterion_grid_size = self.logging.stopping_criterion_grid_size
                    stopping_criterion_frequency = self.logging.stopping_criterion_frequency
                    stop = False

                self._init_new_run()

                for e in range(self.running.n_epochs + 1):
                    utilities = self._training_loop(epoch=e)

                    # Check stopping criterion
                    if self.logging.stopping_criterion_rel_util_loss_diff is not None and \
                            e > 0 and not e % stopping_criterion_frequency:
                        start_time = timer()

                        # Compute relative utility loss
                        loss_ex_ante, _, _ = self._calculate_metrics_util_loss(
                            create_plot_output=False,
                            batch_size=stopping_criterion_batch_size,
                            grid_size=stopping_criterion_grid_size)
                        rel_util_loss = 1 - utilities / (utilities + torch.tensor(loss_ex_ante))
                        stopping_queue.append(rel_util_loss)

                        # Check for convergence when enough data is available
                        if len(stopping_queue) == stopping_queue.maxlen:
                            values = torch.stack(tuple(stopping_queue))
                            if self.logging.enable_logging:
                                stop = self._check_convergence(values, epoch=e)

                        self.overhead = self.overhead + timer() - start_time
                        if stop:
                            print(f'Stopping criterion reached after {e} iterations.')
                            break

                if self.logging.enable_logging and (
                        self.logging.export_step_wise_linear_bid_function_size is not None):
                    bidders = [self.bidders[self._model2bidder[m][0]] for m in range(self.n_models)]
                    logging_utils.export_stepwise_linear_bid(
                        experiment_dir=self.run_log_dir, bidders=bidders,
                        step=self.logging.export_step_wise_linear_bid_function_size)

            finally:
                self._exit_run()

        # Once all runs are done, convert tb event files to csv
        if self.logging.enable_logging and (
                self.logging.save_tb_events_to_csv_detailed or
                self.logging.save_tb_events_to_csv_aggregate or
                self.logging.save_tb_events_to_binary_detailed):
            print('Tabulating tensorboard logs...', end=' ')
            logging_utils.tabulate_tensorboard_logs(
                experiment_dir=self.experiment_log_dir,
                write_detailed=self.logging.save_tb_events_to_csv_detailed,
                write_aggregate=self.logging.save_tb_events_to_csv_aggregate,
                write_binary=self.logging.save_tb_events_to_binary_detailed)

            # logging_utils.print_aggregate_tensorboard_logs(self.experiment_log_dir)
            print('finished.')

    def _check_convergence(self, values: torch.Tensor, stopping_criterion: float = None, epoch: int = None):
        """
        Checks whether difference in stored values is below stopping criterion
        for each model and logs per-player differences to tensorboard.

        args:
            values: Tensor(n_values x n_models)
        returns: bool (True if stopping criterion fulfilled)
        """
        if stopping_criterion is None:
            stopping_criterion = self.logging.stopping_criterion_rel_util_loss_diff

        diffs = values.max(0)[0] - values.min(0)[0]  # size: n_models
        log_params = {'stopping_criterion': diffs}
        self.writer.add_metrics_dict(log_params, self._model_names, epoch, group_prefix='meta')

        return diffs.max().le(stopping_criterion).item()

    ########################################################################################################
    ####################################### Moved logging to here ##########################################
    ########################################################################################################

    # TODO Stefan: method only uses self in eval and for output point
    def _plot(self, plot_data, writer: SummaryWriter or None, epoch=None,
              xlim: list = None, ylim: list = None, labels: list = None,
              x_label="valuation", y_label="bid", fmts: list = None,
              colors: list = None, figure_name: str = 'bid_function',
              plot_points=100):
        """
        This implements plotting simple 2D data.

        Args
            plot_data: tuple of two pytorch tensors first beeing for x axis, second for y.
                Both of dimensions (batch_size, n_models, n_bundles)
            writer: could be replaced by self.writer
            epoch: int, epoch or iteration number
            xlim: list of floats, x axis limits for all n_bundles dimensions
            ylim: list of floats, y axis limits for all n_bundles dimensions
            labels: list of str lables for legend
            fmts: list of str for matplotlib markers and lines
            figure_name: str, for seperate plot saving of e.g. bids and util_loss,
            plot_point: int of number of ploting points for each strategy in each subplot
        """

        if fmts is None:
            fmts = ['o']

        x = plot_data[0].detach().cpu().numpy()
        y = plot_data[1].detach().cpu().numpy()
        n_batch, n_players, n_bundles = y.shape

        n_batch = min(plot_points, n_batch)
        x = x[:n_batch, :, :]
        y = y[:n_batch, :, :]

        # create the plot
        fig, axs = plt.subplots(nrows=1, ncols=n_bundles, sharey=True)
        plt.cla()
        if not isinstance(axs, np.ndarray):
            axs = [axs]  # one plot only

        cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
        n_colors = int(np.ceil(len(fmts) / 2)) if self.config.logging.log_metrics['opt'] else len(fmts)

        # actual plotting
        for plot_idx in range(n_bundles):
            for agent_idx in range(n_players):
                axs[plot_idx].plot(
                    x[:, agent_idx, plot_idx], y[:, agent_idx, plot_idx],
                    fmts[agent_idx % len(fmts)],
                    label=None if labels is None else labels[agent_idx % len(labels)],
                    color=cycle[agent_idx % n_colors] if colors is None else cycle[colors[agent_idx]],
                )

            # formating
            axs[plot_idx].set_xlabel(
                x_label if not isinstance(x_label, list) else x_label[plot_idx]
            )
            if plot_idx == 0:
                axs[plot_idx].set_ylabel(y_label)
                if n_players < 10 and labels is not None:
                    axs[plot_idx].legend(loc='upper left')

            # Set axis limits based on function parameters ´xlim´, ´ylim´ if provided otherwise
            # based on ´self.plot_xmin´ etc. object attributes. In either case, these variables
            # can also be lists for sperate limits of individual plots.
            lims = (xlim, ylim)
            set_lims = (axs[plot_idx].set_xlim, axs[plot_idx].set_ylim)
            str_lims = (['plot_xmin', 'plot_xmax'], ['plot_ymin', 'plot_ymax'])
            # pylint: disable=eval-used
            for lim, set_lim, str_lim in zip(lims, set_lims, str_lims):
                a, b = None, None
                if lim is not None:  # use parameters ´xlim´ etc.
                    if isinstance(lim[0], list):
                        a, b = lim[plot_idx][0], lim[plot_idx][1]
                    else:
                        a, b = lim[0], lim[1]
                elif hasattr(self, str_lim[0]):  # use attributes ´self.plot_xmin´ etc.
                    if isinstance(eval('self.' + str(str_lim[0])), list): # pylint: disable=eval-used
                        a = eval('self.' + str(str_lim[plot_idx]))[0] # pylint: disable=eval-used
                        b = eval('self.' + str(str_lim[plot_idx]))[1] # pylint: disable=eval-used
                    else:
                        a = eval('self.' + str(str_lim[0])) # pylint: disable=eval-used
                        b = eval('self.' + str(str_lim[1])) # pylint: disable=eval-used
                if a is not None:
                    set_lim(a, b)  # call matplotlib function

            axs[plot_idx].locator_params(axis='x', nbins=5)
        title = plt.title if n_bundles == 1 else plt.suptitle
        title('iteration {}'.format(epoch))

        logging_utils.process_figure(fig, epoch=epoch, figure_name=figure_name, tb_group='eval',
                                     tb_writer=writer, display=self.logging.plot_show_inline,
                                     output_dir=self.run_log_dir,
                                     save_png=self.logging.save_figure_to_disk_png,
                                     save_svg=self.logging.save_figure_to_disk_svg)
        return fig

    # TODO: stefan only uses self in output_dir, nowhere else --> can we move this to utils.plotting? etc?
    def _plot_3d(self, plot_data, writer, epoch, figure_name):
        """
        Creating 3d plots. Provide grid if no plot_data is provided
        Args
            plot_data: tuple of two pytorch tensors first beeing the independent, the second the dependent
                Dimensions of first (batch_size, n_models, n_bundles)
                Dimensions of second (batch_size, n_models, 1 or n_bundles), 1 if util_loss
        """
        independent_var = plot_data[0]
        dependent_var = plot_data[1]
        batch_size, n_models, n_bundles = independent_var.shape
        assert n_bundles == 2, "cannot plot != 2 bundles"
        n_plots = dependent_var.shape[2]
        # create the plot
        fig = plt.figure()
        for model in range(n_models):
            for plot in range(n_plots):
                ax = fig.add_subplot(n_models, n_plots, model * n_plots + plot + 1, projection='3d')
                ax.plot_trisurf(
                    independent_var[:, model, 0].detach().cpu().numpy(),
                    independent_var[:, model, 1].detach().cpu().numpy(),
                    dependent_var[:, model, plot].reshape(batch_size).detach().cpu().numpy(),
                    color='yellow',
                    linewidth=0.2,
                    antialiased=True
                )
                ax.zaxis.set_major_locator(LinearLocator(10))
                ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
                ax.set_title('model {}, bundle {}'.format(model, plot))
                ax.view_init(20, -135)
        fig.suptitle('iteration {}'.format(epoch), size=16)
        fig.tight_layout()

        logging_utils.process_figure(fig, epoch=epoch, figure_name=figure_name + '_3d', tb_group='eval',
                                     tb_writer=writer, display=self.logging.plot_show_inline,
                                     output_dir=self.run_log_dir,
                                     save_png=self.logging.save_figure_to_disk_png,
                                     save_svg=self.logging.save_figure_to_disk_svg)
        return fig

    def _evaluate_and_log_epoch(self, epoch: int) -> float:
        """
        Checks which metrics have to be logged and performs logging and plotting.
        Returns:
            - elapsed time in seconds
            - Stefan todos / understanding quesitons
            - TODO: takes log_params. can it be
        """
        start_time = timer()

        # calculate infinity-norm of update step
        new_params = [torch.nn.utils.parameters_to_vector(model.parameters())
                      for model in self.models]
        self._cur_epoch_log_params['update_norm'] = [
            (new_params[i] - self._cur_epoch_log_params['prev_params'][i]).norm(float('inf'))
            for i in range(self.n_models)]
        del self._cur_epoch_log_params['prev_params']

        # logging metrics
        # TODO: should just check if logging is enabled in general... if bne_exists and we log, we always want this
        if self.known_bne and self.logging.log_metrics['opt']:
            utility_vs_bne, epsilon_relative, epsilon_absolute = self._calculate_metrics_known_bne()
            L_2, L_inf = self._calculate_metrics_action_space_norms()
            for i in range(len(self.bne_env)):
                n = '_bne' + str(i + 1) if len(self.bne_env) > 1 else ''
                self._cur_epoch_log_params['utility_vs_bne' + (n if n == '' else n[4:])] \
                    = utility_vs_bne[i]
                self._cur_epoch_log_params['epsilon_relative' + n] = epsilon_relative[i]
                self._cur_epoch_log_params['epsilon_absolute' + n] = epsilon_absolute[i]
                self._cur_epoch_log_params['L_2' + n] = L_2[i]
                self._cur_epoch_log_params['L_inf' + n] = L_inf[i]


        if self.logging.log_metrics['util_loss'] and (epoch % self.logging.util_loss_frequency) == 0:
            create_plot_output = epoch % self.logging.plot_frequency == 0
            self._cur_epoch_log_params['util_loss_ex_ante'], \
            self._cur_epoch_log_params['util_loss_ex_interim'], \
            self._cur_epoch_log_params['estimated_relative_ex_ante_util_loss'] = \
                self._calculate_metrics_util_loss(create_plot_output, epoch)
            print("\tcurrent est. ex-interim loss:" + str(
                [f"{l.item():.4f}" for l in self._cur_epoch_log_params['util_loss_ex_interim']]))

        if self.logging.log_metrics['efficiency'] and (epoch % self.logging.util_loss_frequency) == 0:
            self._cur_epoch_log_params['efficiency'] = \
                self.mechanism.get_efficiency(self.env)

        if self.logging.log_metrics['revenue'] and (epoch % self.logging.util_loss_frequency) == 0:
            self._cur_epoch_log_params['revenue'] = \
                self.mechanism.get_revenue(self.env)

        # plotting
        if epoch % self.logging.plot_frequency == 0:
            print("\tcurrent utilities: " + str(self._cur_epoch_log_params['utilities'].tolist()))

            unique_bidders = [self.env.agents[i[0]] for i in self._model2bidder]
            v = torch.stack(
                [b.valuations[:self.plot_points, ...] for b in unique_bidders],
                dim=1
            )
            b = torch.stack([b.get_action()[:self.plot_points, ...]
                             for b in unique_bidders], dim=1)

            labels = ['NPGA agent {}'.format(i) for i in range(len(self.models))]
            fmts = ['bo'] * len(self.models)
            if self.known_bne and self.logging.log_metrics['opt']:
                for env_idx, _ in enumerate(self.bne_env):
                    v = torch.cat([v, self.v_opt[env_idx]], dim=1)
                    b = torch.cat([b, self.b_opt[env_idx]], dim=1)
                    labels += ['BNE {} agent {}'.format('_' + str(env_idx + 1) if len(self.bne_env) > 1 else '', j)
                               for j in range(len(self.models))]
                    fmts += ['b--'] * len(self.models)

            self._plot(plot_data=(v, b), writer=self.writer, figure_name='bid_function',
                       epoch=epoch, labels=labels, fmts=fmts, plot_points=self.plot_points)

        self.overhead = self.overhead + timer() - start_time
        self._cur_epoch_log_params['overhead_hours'] = self.overhead / 3600
        if self.writer:
            self.writer.add_metrics_dict(self._cur_epoch_log_params, self._model_names, epoch, group_prefix='eval')
        return timer() - start_time

    def _calculate_metrics_known_bne(self):
        """
        Compare performance to BNE and return:
            utility_vs_bne: List[Tensor] of length `len(self.bne_env)`, length of Tensor `n_models`.
            epsilon_relative: List[Tensor] of length `len(self.bne_env)`, length of Tensor `n_models`.
            epsilon_absolute: List[Tensor] of length `len(self.bne_env)`, length of Tensor `n_models`.

        These are all lists of lists. The outer list corrsponds to which BNE is comapred
        (usually there's only one BNE). Each inner list is of length `self.n_models`.
        """
        # shorthand for model to bidder index conversion
        m2b = lambda m: self._model2bidder[m][0]

        utility_vs_bne = [None] * len(self.bne_env)
        epsilon_relative = [None] * len(self.bne_env)
        epsilon_absolute = [None] * len(self.bne_env)

        for bne_idx, bne_env in enumerate(self.bne_env):
            # generally redraw bne_vals, except when this is expensive!
            # length: n_models
            # TODO Stefan: this seems to be false in most settings, even when not desired.
            redraw_bne_vals = not self.logging.cache_eval_actions
            # length: n_models
            utility_vs_bne[bne_idx] = torch.tensor([
                bne_env.get_strategy_reward(
                    model,
                    player_position=m2b(m),
                    draw_valuations=redraw_bne_vals,
                    use_env_valuations=not redraw_bne_vals
                ) for m, model in enumerate(self.models)
            ])
            epsilon_relative[bne_idx] = torch.tensor(
                [1 - utility_vs_bne[bne_idx][i] / self.bne_utilities[bne_idx][m2b(i)]
                 for i, model in enumerate(self.models)]
            )
            epsilon_absolute[bne_idx] = torch.tensor(
                [self.bne_utilities[bne_idx][m2b(i)] - utility_vs_bne[bne_idx][i]
                 for i, model in enumerate(self.models)]
            )

        return utility_vs_bne, epsilon_relative, epsilon_absolute

    def relevant_actions(self):
        r"""Get indecies of actions that matter.

        In some games, and in their BNE, only some of the agents' actions
        matter.

        Returns:
            relevant_actions: (torch.Tensor) of shape (n_agents, n_actions)
                with elements in $\{0, 1\}$ for marking relevancy for BNE.

        """
        return torch.ones(
            size=(len(self.models), self.n_items),
            device=self.config.hardware.device,
            dtype=torch.bool
        )

    def _calculate_metrics_action_space_norms(self):
        """
        Calculate "action space distance" of model and bne-strategy. If
        `self.logging.log_componentwise_norm` is set to true, will only
        return norm of the best action dimension.

        Returns:
            L_2 and L_inf: each a List[Tensor] of length `len(self.bne_env)`, length of Tensor `n_models`.
        """

        L_2 = [None] * len(self.bne_env)
        L_inf = [None] * len(self.bne_env)
        for bne_idx, bne_env in enumerate(self.bne_env):
            # shorthand for model to agent

            # we are only using m2a locally within this loop, so we can safely ignore the following pylint warning:
            # pylint: disable=cell-var-from-loop

            m2a = lambda m: bne_env.agents[self._model2bidder[m][0]]

            L_2[bne_idx] = [
                metrics.norm_strategy_and_actions(
                    model, m2a(i).get_action(), m2a(i).valuations, 2,
                    componentwise=self.logging.log_componentwise_norm,
                    component_selection=self.relevant_actions()[self._model2bidder[i][0], :]
                )
                for i, model in enumerate(self.models)
            ]
            L_inf[bne_idx] = [
                metrics.norm_strategy_and_actions(
                    model, m2a(i).get_action(), m2a(i).valuations, float('inf'),
                    componentwise=self.logging.log_componentwise_norm,
                    component_selection=self.relevant_actions()[self._model2bidder[i][0], :]
                )
                for i, model in enumerate(self.models)
            ]
        return L_2, L_inf

    def _calculate_metrics_util_loss(self, create_plot_output: bool, epoch: int = None,
                                     batch_size=None, grid_size=None):
        """
        Compute mean util_loss of current policy and return
        ex interim util_loss (ex ante util_loss is the average of that tensor)

        Returns:
            ex_ante_util_loss: List[torch.tensor] of length self.n_models
            ex_interim_max_util_loss: List[torch.tensor] of length self.n_models
        """

        env = self.env
        if batch_size is None:
            batch_size = self.logging.util_loss_batch_size
        if grid_size is None:
            grid_size = self.logging.util_loss_grid_size

        assert batch_size <= env.batch_size, \
            "Util_loss for larger than actual batch size not implemented."

        torch.cuda.empty_cache()
        util_loss = [
            metrics.ex_interim_util_loss(env, player_positions[0], batch_size, grid_size,
                                         return_best_response=self.logging.best_response)
            for player_positions in self._model2bidder
        ]
        if self.logging.best_response:
            best_responses = (
                torch.stack([r[1][0] for r in util_loss], dim=1)[:, :, None],
                torch.stack([r[1][1] for r in util_loss], dim=1)[:, :, None]
            )
            util_loss = [t[0] for t in util_loss]
            labels = ['NPGA_{}'.format(i) for i in range(len(self.models))]
            fmts = ['bo'] * len(self.models)
            self._plot(plot_data=best_responses, writer=self.writer,
                       ylim=[0, max(a._grid_ub for a in self.env.agents).cpu()],
                       figure_name='best_responses', y_label='best response',
                       colors=list(range(len(self.models))), epoch=epoch, labels=labels,
                       fmts=fmts, plot_points=self.plot_points)

        ex_ante_util_loss = [util_loss_model.mean() for util_loss_model in util_loss]
        ex_interim_max_util_loss = [util_loss_model.max() for util_loss_model in util_loss]

        # TODO Nils @Stefan: check consisteny with journal version
        estimated_relative_ex_ante_util_loss = [
            (1 - u / (u + l)).item()
            for u, l in zip(self._cur_epoch_log_params['utilities'].tolist(), ex_ante_util_loss)
        ]

        if not hasattr(self, '_max_util_loss'):
            self._max_util_loss = ex_interim_max_util_loss
        if create_plot_output:
            # TODO Nils: differentiate models in plot
            # Transform to output with dim(batch_size, n_models, n_bundle), for util_losses n_bundle=1
            util_losses = torch.stack([util_loss[r] for r in range(len(util_loss))], dim=1)[:, :, None]
            valuations = torch.stack([self.bidders[player_positions[0]].valuations[:batch_size, ...]
                                      for player_positions in self._model2bidder], dim=1)
            plot_output = (valuations, util_losses)
            self._plot(plot_data=plot_output, writer=self.writer,
                       ylim=[0, max(self._max_util_loss).cpu()],
                       figure_name='util_loss_landscape', y_label='ex-interim loss',
                       epoch=epoch, plot_points=self.plot_points)

        return ex_ante_util_loss, ex_interim_max_util_loss, estimated_relative_ex_ante_util_loss

    def _log_experiment_params(self):
        # TODO: write out all experiment params (complete dict) #See issue #113
        # TODO: Stefan: this currently called _per run_. is this desired behavior?

        h_params = {'hyperparameters/batch_size': self.learning.batch_size,
                    'hyperparameters/pretrain_iters': self.learning.pretrain_iters,
                    'hyperparameters/hidden_nodes': str(self.learning.hidden_nodes),
                    'hyperparameters/hidden_activations': str(self.learning.hidden_activations),
                    'hyperparameters/optimizer_hyperparams': str(self.learning.optimizer_hyperparams),
                    'hyperparameters/optimizer_type': self.learning.optimizer_type}

        try:
            self._hparams_metrics['epsilon_relative'] = self._cur_epoch_log_params['epsilon_relative']
        except AttributeError:
            try:
                self._hparams_metrics['util_loss_ex_interim'] = \
                    self._cur_epoch_log_params['util_loss_ex_interim']
            except AttributeError:
                pass
        self.writer.add_hparams(hparam_dict=h_params, metric_dict=self._hparams_metrics)

    def _log_hyperparams(self, epoch=0):
        """Everything that should be logged on every learning_rate update"""

        for i, model in enumerate(self.models):
            self.writer.add_text('hyperparameters/neural_net_spec', str(model), epoch)  # ToDO To hyperparams
            self.writer.add_graph(model, self.env.agents[i].valuations)

    def _save_models(self, directory):
        # TODO: maybe we should also log out all pointwise util_losses in the ending-epoch to disk to
        # use it to make nicer plots for a publication? --> will be done elsewhere. Logging. Assigned to @Hlib/@Stefan
        for model, player_position in zip(self.models, self._model2bidder):
            name = 'model_' + str(player_position[0]) + '.pt'
            torch.save(model.state_dict(), os.path.join(directory, 'models', name))
