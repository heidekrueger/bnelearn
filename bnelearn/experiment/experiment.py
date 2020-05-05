"""
This module defines an experiment. It includes logging and plotting since they 
can often be shared by specific experiments.

TODO:
    -...
"""
import sys
import os
from abc import ABC, abstractmethod
from typing import Iterable, List

import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import pandas as pd

from time import perf_counter as timer
import matplotlib.pyplot as plt
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from mpl_toolkits.mplot3d import Axes3D

# pylint: disable=unnecessary-pass,unused-argument

from bnelearn.bidder import Bidder
from bnelearn.environment import Environment, AuctionEnvironment
from bnelearn.mechanism import Mechanism
from bnelearn.learner import Learner, ESPGLearner
from bnelearn.strategy import NeuralNetStrategy
import bnelearn.util.metrics as metrics

from bnelearn.experiment.gpu_controller import GPUController
from bnelearn.experiment.configurations import ExperimentConfiguration, LearningConfiguration, LoggingConfiguration


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
    mechanism: Mechanism
    positive_output_point: torch.Tensor  # TODO: check type and dimensions

    ## Fields required for plotting
    plot_xmin: float
    plot_xmax: float
    plot_ymin: float
    plot_ymax: float
    ## Optional - set only in some settings

    ## Equilibrium environment
    bne_utilities: torch.Tensor or List[float]  # TODO: check if type hint correct
    bne_env: AuctionEnvironment
    _optimal_bid: callable

    def __init__(self, experiment_config: ExperimentConfiguration, learning_config: LearningConfiguration,
                 logging_config: LoggingConfiguration, gpu_config: GPUController, known_bne=False):

        # Stuff that shuld go somewhere else completely :-D
        # TODO: Stefan: This should not be here, maybe set in frontend and pass?
        root_path = os.path.join(os.path.expanduser('~'), 'bnelearn')
        if root_path not in sys.path:
            sys.path.append(root_path)
        self.log_root = os.path.join(root_path, 'experiments')

        # Configs
        self.experiment_config = experiment_config
        self.learning_config = learning_config
        self.logging_config = logging_config
        self.gpu_config = gpu_config

        # Global Stuff that should be initiated here
        self.plot_frequency = LoggingConfiguration.plot_frequency
        self.max_epochs = LoggingConfiguration.max_epochs
        self.plot_points = LoggingConfiguration.plot_points

        # Everything that will be set up per run initioated with none
        self.log_dir = None  # TODO: is this redundant? someone said it smells
        self.fig = None
        self.writer = None
        self.overhead = 0.0

        self.models: Iterable[torch.nn.Module] = None
        self.bidders: Iterable[Bidder] = None
        self.env: Environment = None
        self.learners: Iterable[Learner] = None

        # These are set on first _log_experiment
        self.v_opt: torch.Tensor = None
        self.b_opt: torch.Tensor = None

        ### Save locally - can haves
        # Logging
        if logging_config.regret_batch_size is not None:
            self.regret_batch_size = logging_config.regret_batch_size
        if logging_config.regret_grid_size is not None:
            self.regret_grid_size = logging_config.regret_grid_size

        # The following required attrs have already been set in many subclasses in earlier logic.
        # Only set here if they haven't. Don't overwrite.
        if not hasattr(self, 'n_players'):
            self.n_players = experiment_config.n_players
        if not hasattr(self, 'payment_rule'):
            self.payment_rule = experiment_config.payment_rule

        ### actual logic
        # Inverse of bidder --> model lookup table
        self._model2bidder: List[List[int]] = [[] for m in range(self.n_models)]
        for b_id, m_id in enumerate(self._bidder2model):
            self._model2bidder[m_id].append(b_id)

        self._setup_mechanism()

        self.known_bne = known_bne  # needs to be set in subclass and either specified as input or set there
        # Cannot lot 'opt' without known bne
        # TODO: Stefan: currently it's not always possible to infer if known bne exists before calling this (super) init
        if logging_config.log_metrics['opt'] or logging_config.log_metrics['l2']:
            assert self.known_bne, "Cannot log 'opt'/'l2'/'rmse' without known_bne"

        if self.known_bne:
            self._setup_eval_environment()

    # TODO: rename this
    def _setup_run(self):
        """Setup everything that is specific to an individual run, including everything nondeterministic"""
        # setup the experiment, don't mess with the order
        self._setup_bidders()
        self._setup_learning_environment()
        self._setup_learners()

    @abstractmethod
    def _setup_mechanism(self):
        pass

    # TODO: move entire name/dir logic out of logger into run
    @abstractmethod
    def _get_logdir(self):
        """"""
        pass

    @abstractmethod
    def _strat_to_bidder(self, strategy, batch_size, player_position=None, cache_actions=False):
        pass

    @abstractmethod
    def _setup_bidders(self):
        """
        """
        pass

    @abstractmethod
    def _setup_learning_environment(self):
        """This method should set up the environment that is used for learning. """
        pass

    def _setup_learners(self):
        # TODO: the current strat_to_player kwargs is weird. Cross-check this with 
        # how values are evaluated in learner.
        # ideally, we can abstract this function away and move the functionality to the base Experiment class.
        # Implementation in SingleItem case is identical except for player position argument below.
        self.learners = []
        for m_id, model in enumerate(self.models):
            self.learners.append(
                ESPGLearner(
                    model=model,
                    environment=self.env,
                    hyperparams=self.learning_config.learner_hyperparams,
                    optimizer_type=self.learning_config.optimizer,
                    optimizer_hyperparams=self.learning_config.optimizer_hyperparams,
                    strat_to_player_kwargs={"player_position": self._model2bidder[m_id][0]}
                )
            )

    def _setup_bidders(self):
        """
        1. Create and save the models and bidders
        2. Save the model parameters (#TODO, Paul: For everyone)
        """
        print('Setting up bidders...')
        self.models = [None] * self.n_models

        for i in range(len(self.models)):
            self.models[i] = NeuralNetStrategy(
                self.learning_config.input_length,
                hidden_nodes=self.learning_config.hidden_nodes,
                hidden_activations=self.learning_config.hidden_activations,
                ensure_positive_output=self.positive_output_point,
                output_length=self.n_items
            ).to(self.gpu_config.device)

        self.bidders = [
            self._strat_to_bidder(self.models[m_id], batch_size=self.learning_config.batch_size, player_position=i)
            for i, m_id in enumerate(self._bidder2model)]

        self.n_parameters = [sum([p.numel() for p in model.parameters()]) for model in
                             self.models]

        if self.learning_config.pretrain_iters > 0:
            print('\tpretraining...')

            if hasattr(self, 'pretrain_transform'):
                pretrain_transform = self.pretrain_transform
            else:
                pretrain_transform = None

            for i, model in enumerate(self.models):
                model.pretrain(self.bidders[self._model2bidder[i][0]].valuations,
                               self.learning_config.pretrain_iters, pretrain_transform)

    def _setup_eval_environment(self):
        """Overwritten by subclasses with known BNE.
        Sets up an environment used for evaluation of learning agents (e.g.) vs known BNE"""
        raise NotImplementedError("This Experiment has no implemented BNE!")

    def _setup_learning_environment(self):
        self.env = AuctionEnvironment(self.mechanism,
                                      agents=self.bidders,
                                      batch_size=self.learning_config.batch_size,
                                      n_players=self.n_players,
                                      strategy_to_player_closure=self._strat_to_bidder)

    # TODO: why?
    @staticmethod
    def get_risk_profile(risk) -> str:
        if risk == 1.0:
            return 'risk_neutral'
        elif risk == 0.5:
            return 'risk_averse'
        else:
            return 'other'

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

        # TODO: everything after this is logging --> measure overhead
        if self.logging_config.logging:
            log_params = {'utilities': utilities, 'prev_params': prev_params}
            elapsed_overhead = self.log_training_iteration(log_params=log_params, epoch=epoch)
            print('epoch {}:\t elapsed {:.2f}s, overhead {:.3f}s'.format(epoch, timer() - tic, elapsed_overhead))
        else:
            print('epoch {}:\t elapsed {:.2f}s'.format(epoch, timer() - tic))

    def run(self, epochs, n_runs: int = 1, run_comment: str = None, seeds: Iterable[int] = None):
        """Runs the experiment implemented by this class for `epochs` number of iterations."""

        if not seeds:
            seeds = list(range(n_runs))

        for run in range(n_runs):
            seed = seeds[run]
            print('Running experiment {} (using seed {})'.format(run, seed))
            torch.random.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            np.random.seed(seed)

            self._setup_run()

            self.log_dir = self._get_logdir()

            # TODO: setup Writer here, or make logger an object that takes
            # with Logger ... : (especially, needs to be destroyed on end of run!)
            if self.logging_config.logging:
                self.log_experiment(run_comment=run_comment, max_epochs=epochs, run=run)
            # disable this to continue training?
            epoch = 0
            for epoch in range(epoch, epoch + epochs + 1):
                self._training_loop(epoch=epoch)

            if self.logging_config.logging and self.logging_config.save_tb_events_to_csv:
                self.log_tb_events()

            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            # if torch.cuda.memory_allocated() > 0:
            #    warnings.warn('Theres a memory leak')

    ########################################################################################################
    ####################################### Moved logging to here ##########################################
    ########################################################################################################
    # based on https://stackoverflow.com/a/57411105/4755970
    # output_dir must be the directory immediately above the runs and each run must have the same shape.
    # No aggregation of multiple subdirectories for now.
    def log_tb_events(self):
        output_dir = self.log_dir

        # runs are all subdirectories that don't start with '.' (exclude '.ipython_checkpoints')
        # add more filters as needed
        runs = [x.name for x in os.scandir(output_dir) if
                x.is_dir() and not x.name.startswith('.') and not x.name == 'alternative']

        d = {'run': [], 'tag': [], 'epoch': [], 'value': [], 'wall_time': []}
        for run in runs:
            ea = EventAccumulator(os.path.join(output_dir, run)).Reload()
            tags = ea.Tags()['scalars']

            for tag in tags:
                for event in ea.Scalars(tag):
                    d['run'].append(run)
                    d['tag'].append(tag)
                    d['value'].append(event.value)
                    d['wall_time'].append(event.wall_time)
                    d['epoch'].append(event.step)

        df = pd.DataFrame(d)
        df.to_csv(os.path.join(output_dir, f'all_results.csv'))

    # Generalize as much as possible to avoid code overload and too much individualism
    def _plot(self, fig, plot_data, writer: SummaryWriter or None, epoch=None,
              xlim: list = None, ylim: list = None, labels: list = None,
              x_label="valuation", y_label="bid", fmts=['o'],
              figure_name: str = 'bid_function', plot_points=100):
        """
        This implements plotting simple 2D data.

        Args
            fig: matplotlib.figure, TODO might not be needed
            plot_data: tuple of two pytorch tensors first beeing for x axis, second for y.
                Both of dimensions (batch_size, n_models, n_bundles)
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
        x = x[:n_batch, :, :]
        y = y[:n_batch, :, :]

        # create the plot
        fig, axs = plt.subplots(nrows=1, ncols=n_bundles, sharey=True)
        plt.cla()
        if not isinstance(axs, np.ndarray):
            axs = [axs]  # one plot only

        cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']

        # actual plotting
        for plot_idx in range(n_bundles):
            for agent_idx in range(n_players):
                axs[plot_idx].plot(
                    x[:, agent_idx, plot_idx], y[:, agent_idx, plot_idx],
                    fmts[agent_idx % len(fmts)],
                    label=None if labels is None else labels[agent_idx % len(labels)],
                    color=cycle[agent_idx % len(set(fmts))],
                )

            # formating
            axs[plot_idx].set_xlabel(x_label)
            if plot_idx == 0:
                axs[plot_idx].set_ylabel(y_label)
                if n_players < 10 and labels is not None:
                    axs[plot_idx].legend(loc='upper left')

            lims = (xlim, ylim)
            set_lims = (axs[plot_idx].set_xlim, axs[plot_idx].set_ylim)
            str_lims = (['plot_xmin', 'plot_xmax'], ['plot_ymin', 'plot_ymax'])
            # TODO: Stefan: können wir hier ein paar comments dazu haben, was da passiert?
            for lim, set_lim, str_lim in zip(lims, set_lims, str_lims):
                a, b = None, None
                if lim is not None:
                    if isinstance(lim[0], list):
                        a, b = lim[plot_idx][0], lim[plot_idx][1]
                    else:
                        a, b = lim[0], lim[1]
                elif hasattr(self, str_lim[0]):
                    if isinstance(eval('self.' + str(str_lim[0])), list):
                        a = eval('self.' + str(str_lim[plot_idx]))[0]
                        b = eval('self.' + str(str_lim[plot_idx]))[1]
                    else:
                        a, b = eval('self.' + str(str_lim[0])), eval('self.' + str(str_lim[1]))
                if a is not None:
                    set_lim(a, b)

            axs[plot_idx].locator_params(axis='x', nbins=5)
        title = plt.title if n_bundles == 1 else plt.suptitle
        title('iteration {}'.format(epoch))

        self._process_figure(fig, writer=writer, epoch=epoch, figure_name=figure_name)

        return fig

    def _plot_3d(self, plot_data, writer, epoch, figure_name):
        """
        Creating 3d plots. Provide grid if no plot_data is provided
        Args
            plot_data: tuple of two pytorch tensors first beeing the independent, the second the dependent
                Dimensions of first (batch_size, n_models, n_bundles)
                Dimensions of second (batch_size, n_models, 1 or n_bundles), 1 if regret
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

        self._process_figure(fig, writer=writer, epoch=epoch, figure_name=figure_name + "_3d")
        return fig

    # TODO: when adding log_dir and logging_config as arguments this could be static as well
    def _process_figure(self, fig, writer=None, epoch=None, figure_name='plot', group='eval', filename=None):
        """displays, logs and/or saves figure built in plot method"""

        if not filename:
            filename = figure_name

        if self.logging_config.save_figure_to_disk_png:
            plt.savefig(os.path.join(self.log_dir, 'png', f'{filename}_{epoch:05}.png'))

        if self.logging_config.save_figure_to_disk_svg:
            plt.savefig(os.path.join(self.log_dir, 'svg', f'{filename}_{epoch:05}.svg'),
                        format='svg', dpi=1200)
        if writer:
            writer.add_figure(f'{group}/{figure_name}', fig, epoch)
        if self.logging_config.plot_show_inline:
            # display.display(plt.gcf())
            plt.show()

    def _log_hyperparams(self, epoch=0):
        """Everything that should be logged on every learning_rate update"""

        for i, model in enumerate(self.models):
            self.writer.add_text('hyperparameters/neural_net_spec', str(model), epoch)
            self.writer.add_graph(model, self.env.agents[i].valuations)

        self.writer.add_scalar('hyperparameters/batch_size', self.learning_config.batch_size, epoch)
        self.writer.add_scalar('hyperparameters/epochs', self.logging_config.max_epochs, epoch)
        self.writer.add_scalar(
            'hyperparameters/pretrain_iters',
            self.learning_config.pretrain_iters,
            epoch
        )

    # TODO: plot_xmin and xmax makes not sense since this might be outside the value function
    def log_experiment(self, run_comment, max_epochs, run=""):
        self.max_epochs = max_epochs

        # setting up plotting
        self.plot_points = min(self.logging_config.plot_points, self.learning_config.batch_size)

        if self.logging_config.log_metrics['opt'] and hasattr(self, 'bne_env'):
            # dim: [points, bidders, items]
            self.v_opt = torch.stack(
                [b.draw_valuations_grid_(self.plot_points)
                 for b in [self.bne_env.agents[i[0]] for i in self._model2bidder]],
                dim=1
            )
            self.b_opt = torch.stack(
                [self._optimal_bid(self.v_opt[:, m, :], player_position=b[0])
                 for m, b in enumerate(self._model2bidder)],
                dim=1
            )
            if self.v_opt.shape[0] != self.plot_points:
                print('´plot_points´ changed due to ´draw_valuations_grid_´')
                self.plot_points = self.v_opt.shape[0]

        is_ipython = 'inline' in plt.get_backend()
        if is_ipython:
            from IPython import display
        plt.rcParams['figure.figsize'] = [8, 5]

        if os.name == 'nt':
            raise ValueError('The run_name may not contain : on Windows!')
        run_name = self.logging_config.file_name + '_' + str(run)
        if run_comment:
            run_name = run_name + ' - ' + str(run_comment)

        self.log_dir = os.path.join(self.log_root, self.log_dir, run_name)
        os.makedirs(self.log_dir, exist_ok=False)
        if self.logging_config.save_figure_to_disk_png:
            os.mkdir(os.path.join(self.log_dir, 'png'))
        if self.logging_config.save_figure_to_disk_svg:
            os.mkdir(os.path.join(self.log_dir, 'svg'))

        print('Started run. Logging to {}'.format(self.log_dir))
        self.fig = plt.figure()

        self.writer = SummaryWriter(self.log_dir, flush_secs=30)
        start_time = timer()
        self._log_experimentparams()  # TODO: what to use
        self._log_hyperparams()
        elapsed = timer() - start_time
        self.overhead += elapsed

    def log_training_iteration(self, log_params: dict, epoch: int) -> float:
        """
        Checks which metrics have to be logged and performs logging and plotting.
        Returns:
            - elapsed time in seconds
        TODO:
            - Currently we get bne_utility of only one model. Change to avg of all?
        """
        start_time = timer()

        # TODO, Paul: Can delete?: model_is_global = len(self.models) == 1

        # calculate infinity-norm of update step
        new_params = [torch.nn.utils.parameters_to_vector(model.parameters())
                      for model in self.models]
        log_params['update_norm'] = [(new_params[i] - log_params['prev_params'][i]).norm(float('inf'))
                                     for i in range(self.n_models)]
        del log_params['prev_params']

        # logging metrics
        if self.logging_config.log_metrics['opt']:
            log_params['utility_vs_bne'], log_params['epsilon_relative'], log_params['epsilon_absolute'] = \
                self._log_metric_opt()

        if self.logging_config.log_metrics['l2']:
            log_params['L_2'], log_params['L_inf'] = self._log_metric_l()

        if self.logging_config.log_metrics['regret'] and (epoch % self.logging_config.regret_frequency) == 0:
            create_plot_output = epoch % self.logging_config.plot_frequency == 0
            log_params['regret_ex_ante'], log_params['regret_ex_interim'] = \
                self._log_metric_regret(create_plot_output, epoch)

        # plotting
        if epoch % self.logging_config.plot_frequency == 0:
            print("\tcurrent utilities: " + str(log_params['utilities'].tolist()))

            unique_bidders = [self.env.agents[i[0]] for i in self._model2bidder]
            v = torch.stack(
                [b.valuations[:self.plot_points, ...] for b in unique_bidders],
                dim=1
            )
            b = torch.stack([b.get_action()[:self.plot_points, ...]
                             for b in unique_bidders], dim=1)

            labels = ['NPGA_{}'.format(i) for i in range(len(self.models))]
            fmts = ['bo'] * len(self.models)
            if self.logging_config.log_metrics['opt']:
                print(
                    "\tutilities vs BNE: {}\n\tepsilon (abs/rel): ({}, {})" \
                        .format(
                        log_params['utility_vs_bne'].tolist(),
                        log_params['epsilon_relative'].tolist(),
                        log_params['epsilon_absolute'].tolist()
                    )
                )
                v = torch.cat([v, self.v_opt], dim=1)
                b = torch.cat([b, self.b_opt], dim=1)
                labels += ['BNE_{}'.format(i) for i in range(len(self.models))]
                fmts += ['b--'] * len(self.models)

            self._plot(fig=self.fig, plot_data=(v, b), writer=self.writer, figure_name='bid_function',
                       epoch=epoch, labels=labels, fmts=fmts, plot_points=self.plot_points)

        self.overhead = self.overhead + timer() - start_time
        log_params['overhead_hours'] = self.overhead / 3600
        self._log_metrics(log_params, epoch=epoch)
        return timer() - start_time

    def _log_metrics(self, metrics_dict: dict, epoch: int, prefix: str = 'eval',
                     param_group_postfix: str = '', metric_prefix: str = ''):
        """ Writes everthing from ´metrics_dict´ to disk via the ´self.writer´.
            keys in ´metrics_dict´ represent the metric name, values should be of type
            float, list, dict, or torch.Tensor.
        """
        name_list = ['agent_{}'.format(i) for i in range(self.experiment_config.n_players)]

        for metric_key, metric_val in metrics_dict.items():
            tag = prefix + param_group_postfix + '/' + metric_prefix + str(metric_key)

            if isinstance(metric_val, float):
                self.writer.add_scalar(tag, metric_val, epoch)

            elif isinstance(metric_val, list):
                self.writer.add_scalars(tag, dict(zip(name_list, metric_val)), epoch)

            elif isinstance(metric_val, dict):
                for key, val in metric_val.items():
                    self.writer.add_scalars(
                        tag, dict(zip([name + '/' + str(key) for name in name_list], val)), epoch
                    )

            elif torch.is_tensor(metric_val):
                self.writer.add_scalars(tag, dict(zip(name_list, metric_val.tolist())), epoch)

            else:
                raise TypeError('metric type {} cannot be saved'.format(type(metric_val)))

    def _log_metric_opt(self):
        """
        Compare performance to BNE and log:
        utility_vs_bne
        epsilon_relative
        epsilon_absolute
        """

        utility_vs_bne = torch.tensor([
            self.bne_env.get_reward(
                self._strat_to_bidder(
                    model, player_position=i,  # TODO: shouldn't this be model_2_bidder i[0]
                    batch_size=self.logging_config.eval_batch_size
                ),
                draw_valuations=False
            ) for i, model in enumerate(self.models)
        ])  # TODO: False because expensive for normal priors
        epsilon_relative = torch.tensor([1 - utility_vs_bne[i] / self.bne_utilities[i]
                                         for i, model in enumerate(self.models)])
        epsilon_absolute = torch.tensor([self.bne_utilities[i] - utility_vs_bne[i]
                                         for i, model in enumerate(self.models)])

        return utility_vs_bne, epsilon_relative, epsilon_absolute

    def _log_metric_l(self):
        """
        Compare action to BNE and log:
        l2 (TODO: add formular)
        l_inf (TODO: add formular)
        """
        L_2 = [metrics.norm_strategy_and_actions(model, self.bne_env.agents[i].get_action(),
                                                 self.bne_env.agents[i].valuations, 2)
               for i, model in enumerate(self.models)]
        L_inf = [metrics.norm_strategy_and_actions(model, self.bne_env.agents[i].get_action(),
                                                   self.bne_env.agents[i].valuations, float('inf'))
                 for i, model in enumerate(self.models)]
        return L_2, L_inf

    def _log_metric_regret(self, create_plot_output: bool, epoch: int = None):
        """
        Compute mean regret of current policy and return
        ex interim regret (ex ante regret is the average of that tensor)
        """

        env = self.env
        regret_batch_size = self.logging_config.regret_batch_size
        regret_grid_size = self.logging_config.regret_grid_size

        bid_profile = torch.zeros(regret_batch_size, env.n_players, env.agents[0].n_items,
                                  dtype=env.agents[0].valuations.dtype, device=env.mechanism.device)
        regret_grid = torch.zeros(regret_grid_size, env.n_players, dtype=env.agents[0].valuations.dtype,
                                  device=env.mechanism.device)

        for agent in env.agents:
            i = agent.player_position

            # TODO Nils: ugly work-around for split-award: needs seperate plot and regret bounds
            if hasattr(agent, 'grid_lb_regret'):
                v_lb = agent.grid_lb_regret
                v_ub = agent.grid_ub_regret
            else:
                v_lb = agent.grid_lb
                v_ub = agent.grid_ub

            # Only supports regret_batch_size <= batch_size
            bid_profile[:, i, :] = agent.get_action()[:regret_batch_size, ...]
            regret_grid[:, i] = torch.linspace(v_lb, v_ub, regret_grid_size,
                                               device=env.mechanism.device)

        torch.cuda.empty_cache()
        regret = [
            metrics.ex_interim_regret(
                env.mechanism, bid_profile,
                learner.strat_to_player_kwargs['player_position'],
                env.agents[learner.strat_to_player_kwargs['player_position']].valuations[:regret_batch_size, ...],
                regret_grid[:, learner.strat_to_player_kwargs['player_position']]
            )
            for learner in self.learners
        ]
        ex_ante_regret = [model_tuple[0].mean() for model_tuple in regret]
        ex_interim_max_regret = [model_tuple[0].max() for model_tuple in regret]
        if create_plot_output:
            # TODO, Paul: Transform to output with dim(batch_size, n_models, n_bundle)
            regrets = torch.stack([regret[r][0] for r in range(len(regret))], dim=1)[:, :, None]
            valuations = torch.stack([regret[r][1] for r in range(len(regret))], dim=1)
            plot_output = (valuations, regrets)
            self._plot(fig=self.fig, plot_data=plot_output, writer=self.writer,
                       ylim=[0, max(ex_interim_max_regret).cpu()],
                       figure_name='regret_function', epoch=epoch, plot_points=self.plot_points)
            # TODO, Paul: Check in detail if correct!?
        return ex_ante_regret, ex_interim_max_regret

    def _log_experimentparams(self):
        # TODO: write out all experiment params (complete dict)
        pass

    def _log_trained_model(self):
        # TODO: write out the trained model at the end of training @Stefan
        # Stefan: Looks good to me. 
        # TODO: maybe we should also log out all pointwise regrets in the ending-epoch to disk to use it to make nicer plots for a publication?
        # Proposal Nils:
        for i, model in enumerate(self.models):
            name = 'saved_model_' + str(i) + '.pt'
            torch.save(model.state_dict(), os.path.join(self.log_dir, name))
