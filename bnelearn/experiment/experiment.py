"""Defines experiment class"""


from abc import ABC, abstractmethod
from typing import Iterable, List

import torch
from torch.utils.tensorboard import SummaryWriter
import numpy as np

# pylint: disable=unnecessary-pass,unused-argument

from bnelearn.bidder import Bidder
from bnelearn.environment import Environment
from bnelearn.mechanism import Mechanism
from bnelearn.learner import Learner

from bnelearn.experiment.gpu_controller import GPUController
from bnelearn.experiment.configurations import ExperimentConfiguration, LearningConfiguration, LoggingConfiguration
from bnelearn.experiment.logger import Logger



class Experiment(ABC):
    """Abstract Class representing an experiment"""

    # abstract fields that must be set in subclass init
    _bidder2model: List[int] = NotImplemented
    n_models: int = NotImplemented
    # TODO: make all fields that MUST be set in subclass abstract members

    def __init__(self, experiment_config: dict, learning_config: LearningConfiguration,
                 logging_config: LoggingConfiguration, gpu_config: GPUController, known_bne=False):

        # Configs
        self.experiment_config = experiment_config
        self.learning_config = learning_config
        self.logging_config = logging_config
        self.gpu_config = gpu_config

        # Save locally
        self.n_players = experiment_config.n_players

        # TODO: decouple --> logic should be in subclasses ?
        #if 'valuation_prior' in experiment_config.keys():
        #    self.valuation_prior = experiment_config['valuation_prior']
        #if 'payment_rule' in experiment_config.keys():
        #    self.mechanism_type = experiment_config.payment_rule

        # TODO: these may possibly stay here, uncommented for now because of added complexity (due to separate regret logging implementation)
        if logging_config.regret_batch_size is not None:
            self.regret_batch_size = logging_config.regret_batch_size
        if logging_config.regret_grid_size is not None:
            self.regret_grid_size = logging_config.regret_grid_size

        # Misc
        self.base_dir = None
        self.models: Iterable[torch.nn.Module] = None

        # Inverse of bidder --> model lookup table
        self._model2bidder: List[List[int]] = [[] for m in range(self.n_models)]
        for b_id, m_id in enumerate(self._bidder2model):
            self._model2bidder[m_id].append(b_id)


        self.mechanism: Mechanism = None
        self.bidders: Iterable[Bidder] = None
        self.env: Environment = None
        self.learners: Iterable[Learner] = None

        # TODO: remove this? move all logging logic into experiment itself?
        self.logger: Logger = None
        root_path = os.path.join(os.path.expanduser('~'), 'bnelearn')
        if root_path not in sys.path:
            sys.path.append(root_path)
        
        self.log_dir = None
        self.fig = None
        self.writer = None
        self.overhead = 0.0
        self.known_bne = known_bne


        # setup everything deterministic that is shared among runs
        self._setup_mechanism()

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
    def _setup_logger(self, base_dir):
        """Creates logger for run.
        THIS IS A TEMPORARY WORKAROUND TODO
        """
        pass

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

    @abstractmethod
    def _setup_learners(self):
        """This method should set up learners for each of the models that are learnable."""
        pass

    def _setup_eval_environment(self):
        """Overwritten by subclasses with known BNE.
        Sets up an environment used for evaluation of learning agents (e.g.) vs known BNE"""

        # this base class method should never be called, otherwise something is wrong in subclass logic.
        # i.e. erroneously assuming a known BNE exists when it doesn't.
        raise NotImplementedError("This Experiment has no implemented BNE!")


    # TODO: why?
    @staticmethod
    def get_risk_profile(risk) -> str:
        if risk == 1.0:
            return 'risk_neutral'
        elif risk == 0.5:
            return 'risk_averse'
        else:
            return 'other'

    @abstractmethod
    def _training_loop(self, epoch, logger):
        """Main training loop to be executed in each iteration."""
        pass

    def run(self, epochs, n_runs: int = 1, run_comment: str=None, seeds: Iterable[int] = None):
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

            log_dir = self._get_logdir()
            logger = self._setup_logger(log_dir)

            # TODO: setup Writer here, or make logger an object that takes
            # with Logger ... : (especially, needs to be destroyed on end of run!)

            logger.log_experiment(run_comment=run_comment, max_epochs=epochs)
            # disable this to continue training?
            epoch = 0
            for epoch in range(epoch, epoch + epochs + 1):
                self._training_loop(epoch=epoch, logger=logger)

            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            # if torch.cuda.memory_allocated() > 0:
            #    warnings.warn('Theres a memory leak')

########################################################################################################
####################################### Moved logging to here ##########################################
########################################################################################################

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

#TODO: when adding log_dir and logging_config as arguments this could be static as well
    def _process_figure(self, fig, writer=None, epoch=None, figure_name='plot', group ='eval', filename=None):
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
        if self.logging_config.show_plot_inline:
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
    #TODO: This comes from single_item, but is VERY similar to MultiUnit. Only self.n_units dimension missing. Adapt!
    #TODO: plot_xmin and xmax makes not sense since this might be outside the value function
    def log_experiment(self, run_comment, max_epochs):
        self.max_epochs = max_epochs
        # setting up plotting
        self.plot_points = min(self.logging_config.plot_points, self.learning_config.batch_size)

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