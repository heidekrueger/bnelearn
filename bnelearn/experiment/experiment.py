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
#from bnelearn.experiment.logger import Logger

import matplotlib.pyplot as plt
import sys
import os
import time
from timeit import default_timer as timer
from matplotlib import colors as mcolors
from mpl_toolkits.mplot3d import Axes3D
import bnelearn.util.metrics as metrics

import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import LinearLocator, FormatStrFormatter

from bnelearn.learner import ESPGLearner
from bnelearn.strategy import Strategy, NeuralNetStrategy
from bnelearn.environment import AuctionEnvironment

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

        # Save locally - must haves
        self.n_players = experiment_config.n_players
        self.payment_rule = experiment_config.payment_rule
        self.n_items = None
        self.models: Iterable[torch.nn.Module] = None
        self.mechanism: Mechanism = None
        self.bidders: Iterable[Bidder] = None
        self.env: Environment = None
        self.learners: Iterable[Learner] = None
        self.positive_output_point = None

        self.plot_frequency = LoggingConfiguration.plot_frequency
        self.max_epochs = LoggingConfiguration.max_epochs
        self.plot_points = LoggingConfiguration.plot_points
        #TODO: Smells like redundancy
        self.log_dir = None
        # TODO: remove this? move all logging logic into experiment itself?
        root_path = os.path.join(os.path.expanduser('~'), 'bnelearn')
        if root_path not in sys.path:
            sys.path.append(root_path)
        self.log_root=os.path.join(root_path, 'experiments')
        self.fig = None
        self.writer = None
        self.overhead = 0.0
        self.known_bne = known_bne

        # Cannot lot 'opt' without known bne
        if logging_config.log_metrics['opt'] or logging_config.log_metrics['l2']:
            assert self.known_bne, "Cannot log 'opt'/'l2'/'rmse' without known_bne"

        ### Save locally - can haves
        # Logging
        if logging_config.regret_batch_size is not None:
            self.regret_batch_size = logging_config.regret_batch_size
        if logging_config.regret_grid_size is not None:
            self.regret_grid_size = logging_config.regret_grid_size

        #TODO: Add if logging bne once implemented
        self.v_opt = None
        self.b_opt = None
        self.bne_utilities = None
        self.bne_env = None

        # Plotting
        self.plot_xmin = None
        self.plot_xmax = None
        self.plot_ymin = None
        self.plot_ymax = None

    def _model_2_bidders(self):
        # Inverse of bidder --> model lookup table
        self._model2bidder: List[List[int]] = [[] for m in range(self.n_models)]
        for b_id, m_id in enumerate(self._bidder2model):
            self._model2bidder[m_id].append(b_id)

    def _setup_mechanism_and_eval_environment(self):
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
            self.learners.append(ESPGLearner(model=model,
                                 environment=self.env,
                                 hyperparams=self.learning_config.learner_hyperparams,
                                 optimizer_type=self.learning_config.optimizer,
                                 optimizer_hyperparams=self.learning_config.optimizer_hyperparams,
                                 strat_to_player_kwargs={"player_position": self._model2bidder[m_id][0]}
                                 ))

    def _setup_bidders(self):
            #TODO, Paul: Consolidate with the others. If not possible for all
            # split in setup_bidders, setup
            """
            1. Create and save the models and bidders
            2. Save the model parameters (#TODO, Paul: For everyone)
            """
            print('Setting up bidders...')
            # TODO: this seems tightly coupled with setup learners... can we change this?
            self.models = [None] * self.n_models

            for i in range(len(self.models)):
                self.models[i] = NeuralNetStrategy(
                    self.learning_config.input_length, hidden_nodes=self.learning_config.hidden_nodes,
                    hidden_activations=self.learning_config.hidden_activations,
                    ensure_positive_output=self.positive_output_point,
                    output_length = self.n_items
                ).to(self.gpu_config.device)

            self.bidders = [
                self._strat_to_bidder(self.models[m_id], batch_size=self.learning_config.batch_size, player_position=i)
                for i, m_id in enumerate(self._bidder2model)]


            self.n_parameters = [sum([p.numel() for p in model.parameters()]) for model in
                                self.models]

            if self.learning_config.pretrain_iters > 0:
                print('\tpretraining...')
                #TODO: why is this on per bidder basis when everything else is on per model basis?
                for i, model in enumerate(self.models):
                    model.pretrain(self.bidders[self._model2bidder[i][0]].valuations, 
                    self.learning_config.pretrain_iters)

    def _setup_eval_environment(self):
        """Overwritten by subclasses with known BNE.
        Sets up an environment used for evaluation of learning agents (e.g.) vs known BNE"""

        # this base class method should never be called, otherwise something is wrong in subclass logic.
        # i.e. erroneously assuming a known BNE exists when it doesn't.
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

        # save current params to calculate update norm
        prev_params = [torch.nn.utils.parameters_to_vector(model.parameters())
                       for model in self.models]

        # update model
        utilities = torch.tensor([
            learner.update_strategy_and_evaluate_utility()
            for learner in self.learners
        ])

        #TODO: everything after this is logging --> measure overhead
        log_params = {'utilities': utilities, 'prev_params': prev_params}
        self.log_training_iteration(log_params=log_params, epoch=epoch)

        print('epoch {}:\t{}s'.format(epoch, round(self.overhead, 2)))

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

            self.log_dir = self._get_logdir()

            # TODO: setup Writer here, or make logger an object that takes
            # with Logger ... : (especially, needs to be destroyed on end of run!)

            self.log_experiment(run_comment=run_comment, max_epochs=epochs, run=run)
            # disable this to continue training?
            epoch = 0
            for epoch in range(epoch, epoch + epochs + 1):
                self._training_loop(epoch=epoch)

            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            # if torch.cuda.memory_allocated() > 0:
            #    warnings.warn('Theres a memory leak')

########################################################################################################
####################################### Moved logging to here ##########################################
########################################################################################################
# Generalize as much as possible to avoid code overload and too much individualism
    def _plot(self, fig, plot_data, writer: SummaryWriter or None, epoch=None,
              xlim: list=None, ylim: list=None, labels: list=None,
              x_label="valuation", y_label="bid", fmts=['o'],
              figure_name: str='bid_function', plot_points=100):
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

            lims = (xlim, ylim)
            set_lims = (axs[plot_idx].set_xlim, axs[plot_idx].set_ylim)
            str_lims = (['plot_xmin', 'plot_xmax'], ['plot_ymin', 'plot_ymax'])
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
        assert n_bundles==2, "cannot plot != 2 bundles"
        n_plots = dependent_var.shape[2]
        # create the plot
        fig = plt.figure()
        for model in range(n_models):
            for plot in range(n_plots):
                ax = fig.add_subplot(n_models, n_plots, model*n_plots+plot+1, projection='3d')
                ax.plot_trisurf(
                    independent_var[:,model,0].detach().cpu().numpy(),
                    independent_var[:,model,1].detach().cpu().numpy(),
                    dependent_var[:,model,plot].reshape(batch_size).detach().cpu().numpy(),
                    color = 'yellow',
                    linewidth = 0.2,
                    antialiased = True
                )
                ax.zaxis.set_major_locator(LinearLocator(10))
                ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
                ax.set_title('model {}, bundle {}'.format(model, plot))
                ax.view_init(20, -135)
        fig.suptitle('iteration {}'.format(epoch), size=16)
        fig.tight_layout()

        self._process_figure(fig, writer=writer, epoch=epoch, figure_name=figure_name+"_3d")
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
        if self.logging_config.plot_show_inline:
            # display.display(plt.gcf())
            plt.show()

    @abstractmethod
    def _log_experimentparams(self):
        pass

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

    #TODO: plot_xmin and xmax makes not sense since this might be outside the value function
    def log_experiment(self, run_comment, max_epochs, run=""):
        self.max_epochs = max_epochs

        # setting up plotting
        self.plot_points = min(self.logging_config.plot_points, self.learning_config.batch_size)

        if self.logging_config.log_metrics['opt']:
            self.v_opt = torch.stack(
                [b.draw_valuations_grid_(self.plot_points)
                 for b in [self.bne_env.agents[i[0]] for i in self._model2bidder]],
                dim=1
            )
            self.b_opt = torch.stack(
                [self._optimal_bid(self.v_opt[:,m,:], player_position=b[0])
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
        self._log_experimentparams() # TODO: what to use
        self._log_hyperparams()
        elapsed = timer() - start_time
        self.overhead += elapsed

    #TODO: Have to get bne_utilities for all models instead of bne_utoility of only one!?
    #TODO: Create one method per metric and check which ones to compute
    def log_training_iteration(self, log_params: dict, epoch: int):
        start_time = timer()

        #TODO, Paul: Can delete?: model_is_global = len(self.models) == 1

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
            create_plot_output = False
            if epoch % self.logging_config.plot_frequency == 0:
                create_plot_output = True
            log_params['regret_ex_ante'], log_params['regret_ex_interim'] = \
                self._log_metric_regret(create_plot_output, epoch)

        # plotting
        if epoch % self.logging_config.plot_frequency == 0:
            print("\tcurrent utilities: " + str(log_params['utilities'].tolist()))

            unique_bidders = [self.env.agents[i[0]] for i in self._model2bidder]
            v = torch.stack(
                [b.valuations[:self.plot_points,...]
                 for b in unique_bidders],
                dim=1
            )
            b = torch.stack([b.get_action()[:self.plot_points,...]
                             for b in unique_bidders], dim=1)

            labels = ['NPGA_{}'.format(i) for i in range(len(self.models))]
            fmts = ['bo'] * len(self.models)
            if self.logging_config.log_metrics['opt']:
                print("\tutilities vs BNE: {}\n\tepsilon (abs/rel): ({}, {})" \
                    .format(
                        log_params['utility_vs_bne'].tolist(),
                        log_params['epsilon_relative'].tolist(),
                        log_params['epsilon_absolute'].tolist()
                    )
                )
                # TODO: handle case of no opt strategy
                v = torch.cat([v, self.v_opt], dim=1)
                b = torch.cat([b, self.b_opt], dim=1)
                labels += ['BNE_{}'.format(i) for i in range(len(self.models))]
                fmts += ['b--'] * len(self.models)

            #TODO, P: What is plotted here? opt or normal?
            # Nils: both in one. If there is a BNE, it's cat to v, b resp.
            self._plot(fig=self.fig, plot_data=(v, b), writer=self.writer, figure_name='bid_function',
                       epoch=epoch, labels=labels, fmts=fmts, plot_points=self.plot_points)

        self.overhead = self.overhead + timer() - start_time
        log_params['overhead_hours'] = self.overhead / 3600

        self._log_metrics(log_params, epoch=epoch)

    def _log_metrics(self, metrics_dict: dict, epoch: int, prefix: str='eval',
                     param_group_postfix: str='', metric_prefix: str=''):
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
                    model, player_position=i,
                    batch_size=self.logging_config.eval_batch_size
                ),
                draw_valuations=False
            ) for i, model in enumerate(self.models)
        ]) #TODO: False because expensive for normal priors
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
        # TODO Nils: @Paul please check logic here. Major changes: (1) use original draw_valuations,
        # as its logic is needed, (2) trying to calculate for all models at once.

        env = self.env
        regret_batch_size = self.logging_config.regret_batch_size
        regret_grid_size = self.logging_config.regret_grid_size

        bid_profile = torch.zeros(regret_batch_size, env.n_players, env.agents[0].n_items,
                                  dtype=env.agents[0].valuations.dtype, device=env.mechanism.device)
        regret_grid = torch.zeros(regret_grid_size, env.n_players, dtype=env.agents[0].valuations.dtype,
                                  device=env.mechanism.device)

        for agent in env.agents:
            i = agent.player_position
            
            # TODO Nils: downward compatible: u_lo/hi should always be of same type: list
            u_lo = self.u_lo[i] if isinstance(self.u_lo, list) else self.u_lo
            u_hi = self.u_hi[i] if isinstance(self.u_hi, list) else self.u_hi

            # TODO Nils: only supports regret_batch_size <= batch_size
            bid_profile[:,i,:] = agent.get_action()[:regret_batch_size,...]
            regret_grid[:,i] = torch.linspace(u_lo, u_hi, regret_grid_size,
                                              device=env.mechanism.device)

        torch.cuda.empty_cache()
        regret = [metrics.ex_interim_regret(env.mechanism, bid_profile, 
                                            learner.strat_to_player_kwargs['player_position'],
                                            env.agents[learner.strat_to_player_kwargs['player_position']].valuations[:regret_batch_size,...],
                                            regret_grid[:,learner.strat_to_player_kwargs['player_position']])
                  for learner in self.learners]
        ex_ante_regret = [model_tuple[0].mean() for model_tuple in regret]
        ex_interim_max_regret = [model_tuple[0].max() for model_tuple in regret]
        if create_plot_output:
            #TODO, Paul: Transform to output with dim(batch_size, n_models, n_bundle)
            regrets = torch.stack([regret[r][0] for r in range(len(regret))], dim=1)[:,:,None]
            valuations = torch.stack([regret[r][1] for r in range(len(regret))], dim=1)
            plot_output = (valuations,regrets)
            self._plot(fig=self.fig, plot_data=plot_output, writer=self.writer, ylim=[0,max(ex_interim_max_regret).cpu()],
                       figure_name='regret_function', epoch=epoch, plot_points=self.plot_points)
            #TODO, Paul: Check in detail if correct!?
        return ex_ante_regret, ex_interim_max_regret

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

        self._plot(
            fig=self.fig, plot_data=(valuations, regrets), writer=self.writer,
            epoch=epoch, xlim=[self.plot_xmin, self.plot_xmax],
            ylim=[0, max_regret.detach().cpu().numpy()],
            x_label="valuation", y_label="regret", figure_name='regret'
        )

        # TODO: why? don't we redraw valuations at beginning of loop anyway.
        for agent in env.agents:
            agent.batch_size = original_batch_size
            agent.draw_valuations_new_batch_(original_batch_size)

        elapsed = timer() - start_time
        self.overhead += elapsed

    def _log_experimentparams(self):
        #TODO: write out all experiment params (complete dict)
        pass

    def _log_trained_model(self):
        #TODO: write out the trained model at the end of training @Stefan
        # Proposal Nils:
        for i, model in enumerate(self.models):
            name = 'saved_model_' + str(i) + '.pt'
            torch.save(model.state_dict(), os.path.join(self.log_dir, name))
