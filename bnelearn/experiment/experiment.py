"""
This module defines an experiment. It includes logging and plotting since they
can often be shared by specific experiments.
"""

from abc import ABC, abstractmethod
from time import perf_counter as timer
from typing import Iterable, List, Callable
import warnings
import traceback


import numpy as np
import torch
from matplotlib.ticker import FormatStrFormatter, LinearLocator
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D  # pylint: disable=unused-import

from torch.utils.tensorboard import SummaryWriter

from bnelearn.bidder import Bidder
from bnelearn.environment import AuctionEnvironment, Environment
from bnelearn.experiment.configurations import (ExperimentConfig)
from bnelearn.learner import ESPGLearner, Learner
from bnelearn.mechanism import Mechanism
from bnelearn.strategy import NeuralNetStrategy
from bnelearn.sampler import ValuationObservationSampler

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
    valuation_size: int
    observation_size: int
    action_size: int
    mechanism: Mechanism
    positive_output_point: torch.Tensor  # shape must be valid model input
    input_length: int

    ## Equilibrium environment
    bne_utilities: torch.Tensor or List[float]  # dimension: n_players
    bne_env: AuctionEnvironment or List[AuctionEnvironment]
    _optimal_bid: Callable or List[Callable]

    def __init__(self, config: ExperimentConfig):
        # Configs, params are duplicated for the ease of usage and brevity
        self.config = config
        self.running = config.running
        self.setting = config.setting
        self.logging = config.logging
        self.learning = config.learning
        self.hardware = config.hardware

        self.sampler: ValuationObservationSampler = None
        self.models: Iterable[torch.nn.Module] = None
        self.bidders: Iterable[Bidder] = None
        self.env: Environment = None
        self.learners: Iterable[Learner] = None

        # TODO: Get rid of these. payment rule should not be part of the
        # experiment interface.
        # The following required attrs have already been set in many subclasses in earlier logic.
        # Only set here if they haven't. Don't overwrite.
        if not hasattr(self, 'n_players'):
            self.n_players = self.setting.n_players
        if not hasattr(self, 'payment_rule'):
            self.payment_rule = self.setting.payment_rule        

        ### actual logic
        # Inverse of bidder --> model lookup table
        self._model2bidder: List[List[int]] = [[] for _ in range(self.n_models)]
        for b_id, m_id in enumerate(self._bidder2model):
            self._model2bidder[m_id].append(b_id)
        self._model_names = self._get_model_names()

        self._setup_mechanism()
        self._setup_sampler()

        self.known_bne = self._check_and_set_known_bne()
        if self.known_bne:
            self._setup_eval_environment()
        
        # Logger is instantiated in the subclasses
        self.logger = None

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
    def _setup_sampler(self):
        """Defines and initializes a sampler to retrieve observations and
           valuations.
        """

    @abstractmethod
    def _strat_to_bidder(self, strategy, batch_size, player_position=None, enable_action_caching=False) -> Bidder:
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
        print('\tSetting up bidders...')
        # this method is part of the init workflow, so we #pylint: disable=attribute-defined-outside-init
        self.models = [None] * self.n_models

        for i in range(len(self.models)):
            self.models[i] = NeuralNetStrategy(
                self.observation_size,
                hidden_nodes=self.learning.hidden_nodes,
                hidden_activations=self.learning.hidden_activations,
                ensure_positive_output=self.positive_output_point.to('cpu'),  # models init. on cpu
                output_length=self.action_size
            ).to(self.hardware.device)

        self.bidders = [
            self._strat_to_bidder(strategy=self.models[m_id], batch_size=self.learning.batch_size, player_position=i)
            for i, m_id in enumerate(self._bidder2model)]

        self.n_parameters = [sum([p.numel() for p in model.parameters()]) for model in
                             self.models]

        if self.learning.pretrain_iters > 0:
            print('Pretraining...')

            if hasattr(self, 'pretrain_transform'):
                pretrain_transform = self.pretrain_transform  # pylint: disable=no-member
            else:
                pretrain_transform = None

            _, obs = self.sampler.draw_profiles()

            for i, model in enumerate(self.models):
                pos = self._model2bidder[i][0]
                model.pretrain(obs[:, pos, :],
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
        self.env = AuctionEnvironment(self.mechanism,
                                      agents=self.bidders,
                                      valuation_observation_sampler=self.sampler,
                                      batch_size=self.learning.batch_size,
                                      n_players=self.n_players,
                                      strategy_to_player_closure=self._strat_to_bidder)

    def _init_new_run(self):
        """Setup everything that is specific to an individual run, including everything nondeterministic"""
        self._setup_bidders()
        self._setup_learning_environment()
        self._setup_learners()

        if self.logger is not None:
            self.logger.init_new_run(models = self.models, env = self.env, bne_utilities = self.bne_utilities)


    def _exit_run(self, global_step=None):
        """Cleans up a run after it is completed"""
        if self.logger is not None:
            self.logger.exit_run(global_step=global_step)

        if self.hardware.cuda:
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

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

        if self.logger is not None:
            self.logger.log_epoch(epoch, utilities, prev_params)
        
        print('epoch {}:\telapsed {:.2f}s'.format(epoch, timer() - tic), end="\r")
        return utilities

    def run(self) -> bool:
        """Runs the experiment implemented by this class, i.e. all defined runs.

        If a run fails for whatever reason, a warning will be raised and the
        next run will be triggered until all runs have completed/failed.

        Returns:
            success (bool): True if all runs ran successfully, false otherwise.
        """
        encountered_errors: bool = False

        if not self.running.seeds:
            self.running.seeds = list(range(self.running.n_runs))

        assert sum(1 for _ in self.running.seeds) == self.running.n_runs, \
            "Number of seeds doesn't match number of runs."

        for run_id, seed in enumerate(self.running.seeds):
            print(f'\nRunning experiment {run_id} (using seed {seed})')
            epoch = None
            try:
                if self.logger is not None:
                    self.logger.set_current_run(run_id=run_id, seed=seed)              

                torch.random.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
                np.random.seed(seed)

                self._init_new_run()

                for epoch in range(self.running.n_epochs + 1):
                    utilities = self._training_loop(epoch=epoch)
                
                if self.logger is not None and self.logging.export_step_wise_linear_bid_function_size is not None:
                    bidders = [self.bidders[self._model2bidder[m][0]] for m in range(self.n_models)]
                    self.logger.export_stepwise_linear_bid(bidders=bidders)                    

            except Exception as e:
                encountered_errors = True
                tb = traceback.format_exc()
                print("\t Error... aborting run.")
                warnings.warn(f"WARNING: Run {run_id} failed with {type(e)}! Traceback:\n{tb}")

            finally:
                self._exit_run(global_step=epoch)

        if self.logger is not None:
            self.logger.tabulate_tensorboard_logs()   

        return not encountered_errors


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

            for lim, set_lim, str_lim in zip(lims, set_lims, str_lims):
                a, b = None, None
                if lim is not None:  # use parameters ´xlim´ etc.
                    if isinstance(lim[0], list):
                        a, b = lim[plot_idx][0], lim[plot_idx][1]
                    else:
                        a, b = lim[0], lim[1]
                elif hasattr(self, str_lim[0]):  # use attributes ´self.plot_xmin´ etc.
                    if isinstance(eval('self.' + str(str_lim[0])), list):
                        a = eval('self.' + str(str_lim[plot_idx]))[0]
                        b = eval('self.' + str(str_lim[plot_idx]))[1]
                    else:
                        a = eval('self.' + str(str_lim[0]))
                        b = eval('self.' + str(str_lim[1]))
                if a is not None:
                    set_lim(a, b)  # call matplotlib function

            axs[plot_idx].locator_params(axis='x', nbins=5)
        title = plt.title if n_bundles == 1 else plt.suptitle
        title('iteration {}'.format(epoch))

        self.logger.process_figure(fig, epoch=epoch, figure_name=figure_name, tb_group='eval', tb_writer=writer)
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

        self.logger.process_figure(fig, epoch=epoch, figure_name=figure_name + '_3d', tb_group='eval', tb_writer=writer)
        return fig
