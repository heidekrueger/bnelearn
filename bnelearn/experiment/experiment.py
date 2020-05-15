"""
This module defines an experiment. It includes logging and plotting since they
can often be shared by specific experiments.
"""

import os
import time
from abc import ABC, abstractmethod
from time import perf_counter as timer
from typing import Iterable, List

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.ticker import FormatStrFormatter, LinearLocator
from mpl_toolkits.mplot3d import Axes3D # pylint: disable=(whatever-message-pylint has. I assume unused-import)
from torch.utils.tensorboard import SummaryWriter
import bnelearn.util.logging as logging_utils
import bnelearn.util.metrics as metrics
from bnelearn.bidder import Bidder
from bnelearn.environment import AuctionEnvironment, Environment
from bnelearn.experiment.configurations import (ExperimentConfiguration,
                                                LearningConfiguration,
                                                LoggingConfiguration)
from bnelearn.experiment.gpu_controller import GPUController
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
    positive_output_point: torch.Tensor # shape must be valid model input
    input_length: int

    ## Fields required for plotting
    plot_xmin: float
    plot_xmax: float
    plot_ymin: float
    plot_ymax: float
    ## Optional - set only in some settings

    ## Equilibrium environment
    bne_utilities: torch.Tensor or List[float]  # dimension: n_players
    bne_env: AuctionEnvironment
    _optimal_bid: callable

    def __init__(self, experiment_config: ExperimentConfiguration, learning_config: LearningConfiguration,
                 logging_config: LoggingConfiguration, gpu_config: GPUController, known_bne=False):
        # Configs
        self.experiment_config = experiment_config
        self.learning_config = learning_config
        self.logging_config = logging_config
        self.gpu_config = gpu_config

        # Global Stuff that should be initiated here
        self.plot_frequency = logging_config.plot_frequency
        self.plot_points = min(logging_config.plot_points, learning_config.batch_size)

        # Everything that will be set up per run initioated with none
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

        ### Save locally - can haves
        # Logging
        if logging_config.util_loss_batch_size is not None:
            self.util_loss_batch_size = logging_config.util_loss_batch_size
        if logging_config.util_loss_grid_size is not None:
            self.util_loss_grid_size = logging_config.util_loss_grid_size



        # The following required attrs have already been set in many subclasses in earlier logic.
        # Only set here if they haven't. Don't overwrite.
        if not hasattr(self, 'n_players'):
            self.n_players = experiment_config.n_players
        if not hasattr(self, 'payment_rule'):
            self.payment_rule = experiment_config.payment_rule

        # sets log dir for experiment. Individual runs will log to subdirectories of this.
        self.experiment_log_dir = os.path.join(logging_config.log_root_dir,
                                               self._get_logdir_hierarchy(),
                                               logging_config.experiment_dir)

        ### actual logic
        # Inverse of bidder --> model lookup table
        self._model2bidder: List[List[int]] = [[] for m in range(self.n_models)]
        for b_id, m_id in enumerate(self._bidder2model):
            self._model2bidder[m_id].append(b_id)
        self._model_names = self._get_model_names()

        self._setup_mechanism()

        self.known_bne = known_bne  # needs to be set in subclass and either specified as input or set there
        # Cannot log 'opt' without known bne
        if logging_config.log_metrics['opt'] or logging_config.log_metrics['l2']:
            assert self.known_bne, "Cannot log 'opt'/'l2'/'rmse' without known_bne"

        if self.known_bne:
            self._setup_eval_environment()



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
        if self.n_models ==1:
            return []
        return ['bidder' + str(bidders[0]) if len(bidders)==1 else
                'bidders'+ ''.join([str(b) for b in bidders])
                for bidders in self._model2bidder]

    @abstractmethod
    def _strat_to_bidder(self, strategy, batch_size, player_position=None, cache_actions=False):
        pass

    def _setup_learners(self):

        self.learners = [
            ESPGLearner(model=model,
                        environment=self.env,
                        hyperparams=self.learning_config.learner_hyperparams,
                        optimizer_type=self.learning_config.optimizer,
                        optimizer_hyperparams=self.learning_config.optimizer_hyperparams,
                        strat_to_player_kwargs={"player_position": self._model2bidder[m_id][0]})
            for m_id, model in enumerate(self.models)]

    def _setup_bidders(self):
        """
        1. Create and save the models and bidders
        2. Save the model parameters
        """
        print('Setting up bidders...')
        self.models = [None] * self.n_models

        for i in range(len(self.models)):
            self.models[i] = NeuralNetStrategy(
                self.input_length,
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
                pretrain_transform = self.pretrain_transform # pylint: disable=no-member
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

    def _init_new_run(self):
        """Setup everything that is specific to an individual run, including everything nondeterministic"""
        self._setup_bidders()
        self._setup_learning_environment()
        self._setup_learners()

        output_dir = self.run_log_dir

        if self.logging_config.log_metrics['opt'] and hasattr(self, 'bne_env'):
            # dim: [points, bidders, items]
            self.v_opt = torch.stack(
                [b.draw_values_grid(self.plot_points)
                 for b in [self.bne_env.agents[i[0]] for i in self._model2bidder]],
                dim=1
            )
            self.b_opt = torch.stack(
                [self._optimal_bid(self.v_opt[:, m, :], player_position=b[0])
                 for m, b in enumerate(self._model2bidder)],
                dim=1
            )
            if self.v_opt.shape[0] != self.plot_points:
                print('´plot_points´ changed due to ´draw_values_grid´')
                self.plot_points = self.v_opt.shape[0]

        is_ipython = 'inline' in plt.get_backend()
        if is_ipython:
            from IPython import display
        plt.rcParams['figure.figsize'] = [8, 5]

        if self.logging_config.enable_logging:
            os.makedirs(output_dir, exist_ok=False)
            if self.logging_config.save_figure_to_disk_png :
                os.mkdir(os.path.join(output_dir, 'png'))
            if self.logging_config.save_figure_to_disk_svg:
                os.mkdir(os.path.join(output_dir, 'svg'))
            if self.logging_config.save_models:
                os.mkdir(os.path.join(output_dir, 'models'))

            print('Started run. Logging to {}'.format(output_dir))
            self.fig = plt.figure()
            self.writer = logging_utils.CustomSummaryWriter(output_dir, flush_secs=30)

            tic = timer()
            if self.logging_config.enable_logging:
                self._log_experiment_params() #TODO: should probably be called only once, not every run
                self._log_hyperparams()
            elapsed = timer() - tic
        else:
            print('Logging disabled.')
            elapsed = 0
        self.overhead += elapsed

    def _exit_run(self):
        """Cleans up a run after it is completed"""
        if self.logging_config.enable_logging and self.logging_config.save_models:
            self._save_models(directory = self.run_log_dir)

        del self.writer #make this explicit to force cleanup and closing of tb-logfiles
        self.writer = None

        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        # if torch.cuda.memory_allocated() > 0:
        #    warnings.warn('Theres a memory leak')

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

        if self.logging_config.enable_logging:
            log_params = {'utilities': utilities, 'prev_params': prev_params}
            elapsed_overhead = self._evaluate_and_log_epoch(log_params=log_params, epoch=epoch)
            print('epoch {}:\t elapsed {:.2f}s, overhead {:.3f}s'.format(epoch, timer() - tic, elapsed_overhead))
        else:
            print('epoch {}:\t elapsed {:.2f}s'.format(epoch, timer() - tic))

        return utilities


    def run(self, epochs, n_runs: int = 1, seeds: Iterable[int] = None):
        """Runs the experiment implemented by this class for `epochs` number of iterations."""
        if not seeds:
            seeds = list(range(n_runs))

        assert len(seeds) == n_runs, "Number of seeds doesn't match number of runs."

        for run_id, seed in enumerate(seeds):
            print(f'Running experiment {run_id} (using seed {seed})')
            self.run_log_dir = os.path.join(
                self.experiment_log_dir,
                f'{run_id:02d} ' + time.strftime('%H.%M.%S ') + str(seed))
            torch.random.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            np.random.seed(seed)
            if self.logging_config.stopping_criterion_rel_util_loss_dif:
                stopping_list = []

            self._init_new_run()

            for e in range(epochs+1):
                    
                utilities = self._training_loop(epoch=e)
                # If the stopping criterion is set, ...
                if self.logging_config.stopping_criterion_rel_util_loss_dif is not None and e%100 == 0:
                    start_time = timer()
                    util_loss_batch_size_tmp = self.logging_config.util_loss_batch_size
                    util_loss_grid_size_tmp = self.logging_config.util_loss_grid_size
                    self.logging_config.util_loss_batch_size = min(self.logging_config.util_loss_batch_size, 2**12)
                    self.logging_config.util_loss_grid_size = 2**7
                    # ...get the util_loss
                    loss_ex_ante, _ = self._calculate_metrics_util_loss(False)
                    self.logging_config.util_loss_batch_size = util_loss_batch_size_tmp
                    self.logging_config.util_loss_grid_size = util_loss_grid_size_tmp
                    # ...and compute the rel_util_loss
                    stopping_list.append(1 - utilities/(utilities + torch.tensor(loss_ex_ante)))
                    if len(stopping_list) >= 3:
                        # ...finally check for convergence
                        if self._check_convergence(stopping_list, self.logging_config.stopping_criterion_rel_util_loss_dif):
                            break
                    self.overhead = self.overhead + timer() - start_time
            self._exit_run()

        # Once all runs are done, convert tb event files to csv
        if self.logging_config.enable_logging and (
                self.logging_config.save_tb_events_to_csv_detailed or
                self.logging_config.save_tb_events_to_csv_aggregate or
                self.logging_config.save_tb_events_to_binary_detailed):
            logging_utils.tabulate_tensorboard_logs(
                experiment_dir=self.experiment_log_dir,
                write_detailed=self.logging_config.save_tb_events_to_csv_detailed,
                write_aggregate=self.logging_config.save_tb_events_to_csv_aggregate,
                write_binary=self.logging_config.save_tb_events_to_binary_detailed)

            logging_utils.print_aggregate_tensorboard_logs(self.experiment_log_dir)
            logging_utils.print_full_tensorboard_logs(self.experiment_log_dir)

    def _check_convergence(self, stopping_list: list, stopping_criterion: float):
        """
        Checks whether the stored values in stopping_list all fullfil the stopping criterion
        args:
            stopping_list: list[3]
        """
        print(stopping_list)
        for k in range(len(stopping_list)):
            for k2 in range(k+1, len(stopping_list)):
                for bidder in range(len(stopping_list[0])):
                    print(abs(stopping_list[k][bidder]-stopping_list[k2][bidder]))
                    if(abs(stopping_list[k][bidder]-stopping_list[k2][bidder]) > stopping_criterion):
                        stopping_list.pop(0)
                        return False
        return True

    ########################################################################################################
    ####################################### Moved logging to here ##########################################
    ########################################################################################################


    # TODO Stefan: method only uses self in eval and for output point
    def _plot(self, plot_data, writer: SummaryWriter or None, epoch=None,
              xlim: list = None, ylim: list = None, labels: list = None,
              x_label="valuation", y_label="bid", fmts: list = None,
              figure_name: str = 'bid_function', plot_points=100):
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

            """
            set axis limits based on function parameters ´xlim´, ´ylim´ if provided otherwise
            based on ´self.plot_xmin´ etc. object attributes. In either case, these variables
            can also be lists for sperate limits of individual plots.
            """
            lims = (xlim, ylim)
            set_lims = (axs[plot_idx].set_xlim, axs[plot_idx].set_ylim)
            str_lims = (['plot_xmin', 'plot_xmax'], ['plot_ymin', 'plot_ymax'])
            for lim, set_lim, str_lim in zip(lims, set_lims, str_lims):
                a, b = None, None
                if lim is not None: # use parameters ´xlim´ etc.
                    if isinstance(lim[0], list):
                        a, b = lim[plot_idx][0], lim[plot_idx][1]
                    else:
                        a, b = lim[0], lim[1]
                elif hasattr(self, str_lim[0]): # use attributes ´self.plot_xmin´ etc.
                    if isinstance(eval('self.' + str(str_lim[0])), list):
                        a = eval('self.' + str(str_lim[plot_idx]))[0]
                        b = eval('self.' + str(str_lim[plot_idx]))[1]
                    else:
                        a, b = eval('self.' + str(str_lim[0])), eval('self.' + str(str_lim[1]))
                if a is not None:
                    set_lim(a, b) # call matplotlib function

            axs[plot_idx].locator_params(axis='x', nbins=5)
        title = plt.title if n_bundles == 1 else plt.suptitle
        title('iteration {}'.format(epoch))

        logging_utils.process_figure(fig, epoch=epoch, figure_name=figure_name, tb_group='eval',
                                     tb_writer=writer, display=self.logging_config.plot_show_inline,
                                     output_dir=self.run_log_dir,
                                     save_png=self.logging_config.save_figure_to_disk_png,
                                     save_svg = self.logging_config.save_figure_to_disk_svg)
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

        logging_utils.process_figure(fig, epoch=epoch, figure_name=figure_name+'_3d', tb_group='eval',
                                     tb_writer=writer, display=self.logging_config.plot_show_inline,
                                     output_dir=self.run_log_dir,
                                     save_png=self.logging_config.save_figure_to_disk_png,
                                     save_svg = self.logging_config.save_figure_to_disk_svg)
        return fig

    def _evaluate_and_log_epoch(self, log_params: dict, epoch: int) -> float:
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
        log_params['update_norm'] = [(new_params[i] - log_params['prev_params'][i]).norm(float('inf'))
                                     for i in range(self.n_models)]
        del log_params['prev_params']

        # logging metrics
        if self.logging_config.log_metrics['opt']:
            log_params['utility_vs_bne'], log_params['epsilon_relative'], log_params['epsilon_absolute'] = \
                self._calculate_metrics_known_bne()

        if self.logging_config.log_metrics['l2']:
            log_params['L_2'], log_params['L_inf'] = self._calculate_metrics_action_space_norms()

        if self.logging_config.log_metrics['util_loss'] and (epoch % self.logging_config.util_loss_frequency) == 0:
            create_plot_output = epoch % self.logging_config.plot_frequency == 0
            log_params['util_loss_ex_ante'], log_params['util_loss_ex_interim'] = \
                self._calculate_metrics_util_loss(create_plot_output, epoch)

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

            self._plot(plot_data=(v, b), writer=self.writer, figure_name='bid_function',
                       epoch=epoch, labels=labels, fmts=fmts, plot_points=self.plot_points)

        self.overhead = self.overhead + timer() - start_time
        log_params['overhead_hours'] = self.overhead / 3600
        if self.writer:
            self.writer.add_metrics_dict(log_params, self._model_names, epoch, group_prefix = 'eval')
        return timer() - start_time


    def _calculate_metrics_known_bne(self):
        """
        Compare performance to BNE and return:
        utility_vs_bne
        epsilon_relative
        epsilon_absolute

        Each is a list of length self.n_models
        """
        # shorthand for model to bidder index conversion
        m2b = lambda m: self._model2bidder[m][0]

        # length: n_models
        utility_vs_bne = torch.tensor([
            self.bne_env.get_reward(
                self._strat_to_bidder(
                    model, player_position=m2b(i),
                    batch_size=self.logging_config.eval_batch_size
                ),
                draw_valuations=False # False because we want to use cached actions when set, reevaluation is expensive
            ) for i, model in enumerate(self.models)
        ])
        epsilon_relative = torch.tensor(
            [1 - utility_vs_bne[i] / self.bne_utilities[m2b(i)] for i, model in enumerate(self.models)])
        epsilon_absolute = torch.tensor(
            [self.bne_utilities[m2b(i)] - utility_vs_bne[i] for i, model in enumerate(self.models)])

        return utility_vs_bne, epsilon_relative, epsilon_absolute

    def _calculate_metrics_action_space_norms(self):
        """
        Calculate "action space distance" of model and bne-strategy

        Returns:
            L_2 and L_inf, each a list of length self.models
        """
        # shorthand for model to agent
        m2a = lambda m: self.bne_env.agents[self._model2bidder[m][0]]

        L_2 = [metrics.norm_strategy_and_actions(model, m2a(i).get_action(), m2a(i).valuations, 2)
               for i, model in enumerate(self.models)]
        L_inf = [metrics.norm_strategy_and_actions(model, m2a(i).get_action(), m2a(i).valuations, float('inf'))
                 for i, model in enumerate(self.models)]
        return L_2, L_inf

    def _calculate_metrics_util_loss(self, create_plot_output: bool, epoch: int = None):
        """
        Compute mean util_loss of current policy and return
        ex interim util_loss (ex ante util_loss is the average of that tensor)
        """

        env = self.env
        util_loss_batch_size = self.logging_config.util_loss_batch_size
        util_loss_grid_size = self.logging_config.util_loss_grid_size

        assert util_loss_batch_size <= env.batch_size, "Util_loss for larger than actual batch size not implemented."
        bid_profile = torch.zeros(util_loss_batch_size, env.n_players, env.agents[0].n_items,
                                  dtype=env.agents[0].valuations.dtype, device=env.mechanism.device)

        for agent in env.agents:
            # Only supports util_loss_batch_size <= batch_size
            bid_profile[:, agent.player_position, :] = agent.get_action()[:util_loss_batch_size, ...]

        torch.cuda.empty_cache()
        util_loss = [
            metrics.ex_interim_util_loss(
                env.mechanism, bid_profile,
                learner.strat_to_player_kwargs['player_position'],
                env.agents[learner.strat_to_player_kwargs['player_position']].valuations[:util_loss_batch_size, ...],
                env.agents[learner.strat_to_player_kwargs['player_position']].draw_values_grid(util_loss_grid_size)
            )
            for learner in self.learners
        ]
        ex_ante_util_loss = [model_tuple[0].mean() for model_tuple in util_loss]
        ex_interim_max_util_loss = [model_tuple[0].max() for model_tuple in util_loss]
        if create_plot_output:
            # Transform to output with dim(batch_size, n_models, n_bundle), for util_losss n_bundle=1
            util_losss = torch.stack([util_loss[r][0] for r in range(len(util_loss))], dim=1)[:, :, None]
            valuations = torch.stack([util_loss[r][1] for r in range(len(util_loss))], dim=1)
            plot_output = (valuations, util_losss)
            self._plot(plot_data=plot_output, writer=self.writer,
                       ylim=[0, max(ex_interim_max_util_loss).cpu()],
                       figure_name='util_loss_function', epoch=epoch, plot_points=self.plot_points)
        return ex_ante_util_loss, ex_interim_max_util_loss

    def _log_experiment_params(self):
        # TODO: write out all experiment params (complete dict) #See issue #113
        # TODO: Stefan: this currently called _per run_. is this desired behavior?
        pass

    def _log_hyperparams(self, epoch=0):
        """Everything that should be logged on every learning_rate update"""

        for i, model in enumerate(self.models):
            self.writer.add_text('hyperparameters/neural_net_spec', str(model), epoch)
            self.writer.add_graph(model, self.env.agents[i].valuations)

        self.writer.add_scalar('hyperparameters/batch_size', self.learning_config.batch_size, epoch)
        self.writer.add_scalar(
            'hyperparameters/pretrain_iters',
            self.learning_config.pretrain_iters,
            epoch
        )

    def _save_models(self, directory):
        # TODO: maybe we should also log out all pointwise util_losss in the ending-epoch to disk to
        # use it to make nicer plots for a publication? --> will be done elsewhere. Logging. Assigned to @Hlib/@Stefan
        for model, player_position in zip(self.models, self._model2bidder):
            name = 'model_' + str(player_position[0]) + '.pt'
            torch.save(model.state_dict(), os.path.join(directory, 'models', name))
