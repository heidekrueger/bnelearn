"""Defines experiment class"""
import os.path
import time
import warnings
from abc import ABC, abstractmethod

import matplotlib.pyplot as plt
import torch
from pandas import np
from torch.utils.tensorboard import SummaryWriter

# pylint: disable=unnecessary-pass,unused-argument
from bnelearn.environment import Environment
from bnelearn.experiment.gpu_controller import GPUController
from bnelearn.experiment.logger import Logger
from bnelearn.experiment.plotter import Plotter
from bnelearn.learner import Learner
from bnelearn.strategy import Strategy


class Experiment(ABC):
    """Abstract Class representing an experiment"""

    def __init__(self, mechanism_type, n_players, gpu_config: GPUController, logger: Logger, plotter: Plotter,
                 learner_hyperparams: dict, optimizer_type: torch.optim, optimizer_hyperparams: dict,
                 input_length: int, hidden_nodes: list, hidden_activations: list, pretrain_iters: int = 500,
                 batch_size: int = 2**13, eval_batch_size: int = 2**12, cache_eval_actions: bool = True,
                 risk: float = 1.0, n_runs: int = 1):

        self.n_players = n_players
        self.mechanism_type = mechanism_type
        self.gpu_config = gpu_config
        self.n_runs = n_runs
        self.risk = risk
        self.risk_profile = Experiment.get_risk_profile(risk)
        self.logger = logger
        self.plotter = plotter
        self.learner_hyperparams = learner_hyperparams
        self.optimizer_type = optimizer_type
        self.optimizer_hyperparams = optimizer_hyperparams
        self.input_length = input_length
        self.hidden_nodes = hidden_nodes
        self.hidden_activations = hidden_activations
        self.pretrain_iters = pretrain_iters
        self.batch_size = batch_size
        self.eval_batch_size = eval_batch_size
        self.cache_eval_actions = cache_eval_actions

        self.valuation_prior = None
        self.log_dir = None  # is set dynamically in each run
        self.fig = None  # is set dynamically in each run
        self.seeds = list(range(n_runs))

        # Setup the experiment
        self.setup_experiment_domain()
        self.setup_bidders()
        self.setup_learning_environment()
        self.setup_learners()
        self.setup_eval_environment()
        self.setup_name()


    #ToDO This is a temporary measure
    @abstractmethod
    def setup_name(self):
        """"""
        pass

    @abstractmethod
    def setup_experiment_domain(self):
        """This method should set up the set of Players/Bidders
        """
        pass

    @abstractmethod
    def setup_bidders(self):
        """
        """
        pass

    @abstractmethod
    def setup_learning_environment(self):
        """This method should set up the environment that is used for learning. """
        pass

    @abstractmethod
    def setup_learners(self):
        """This method should set up learners for each of the models that are learnable."""
        pass

    @abstractmethod
    def setup_eval_environment(self):
        """Sets up an environment used for evaluation of learning agents (e.g.) vs known BNE"""
        pass

    @abstractmethod
    def optimal_bid(self, valuation):
        """Defines optimal BNE strategy in this setting"""
        pass

    @staticmethod
    def get_risk_profile(risk) -> str:
        if risk == 1.0:
            return 'risk_neutral'
        elif risk == 0.5:
            return 'risk_averse'
        else:
            return 'other'

    def plot(self, fig, plot_data, writer: SummaryWriter or None, e=None):
        """This method should implement a vizualization of the experiment at the current state"""
        pass

    # ToDO Move to logger
    def _process_figure(self, fig, writer=None, e=None):
        """displays, logs and/or saves figure built in plot method"""

        if self.logger.logging_options['save_figure_to_disc_png']:
            plt.savefig(os.path.join(self.log_dir, 'png', f'epoch_{e:05}.png'))

        if self.logger.logging_options['save_figure_to_disc_svg']:
            plt.savefig(os.path.join(self.log_dir, 'svg', f'epoch_{e:05}.svg'),
                        format='svg', dpi=1200)
        if writer:
            writer.add_figure('eval/bid_function', fig, e)
        if self.logger.logging_options['show_plot_inline']:
            # display.display(plt.gcf())
            plt.show()

    def log_once(self, writer, e):
        """Logging function called once at the beginning of the experiment."""
        """Everything that should be logged only once on initialization."""
        # writer.add_scalar('debug/total_model_parameters', n_parameters, e)
        # writer.add_text('hyperparams/neural_net_spec', str(self.model), 0)
        # writer.add_scalar('debug/eval_batch_size', eval_batch_size, e)
        writer.add_graph(self.model, self.env.agents[0].valuations)

    def log_metrics(self, writer, e):
        """Logging function called after each learning iteration"""
        writer.add_scalar('eval/utility', self.utility, e)
        writer.add_scalar('debug/norm_parameter_update', self.update_norm, e)
        writer.add_scalar('eval/utility_vs_bne', self.utility_vs_bne, e)
        writer.add_scalar('eval/epsilon_relative', self.epsilon_relative, e)
        writer.add_scalar('eval/epsilon_absolute', self.epsilon_absolute, e)
        writer.add_scalar('eval/L_2', self.L_2, e)
        writer.add_scalar('eval/L_inf', self.L_inf, e)

    def log_hyperparams(self, writer, e):
        """Logging function called when hyperparameters have changed"""
        pass

    @abstractmethod
    def training_loop(self, writer, e):
        """Main training loop to be executed in each iteration."""
        pass

    # ToDO Move logging to logger
    def run(self, epochs, run_comment=None):
        """Runs the experiment implemented by this class for `epochs` number of iterations."""

        if os.name == 'nt':
            raise ValueError('The run_name may not contain : on Windows!')
        run_name = time.strftime('%Y-%m-%d %a %H:%M:%S')
        if run_comment:
            run_name = run_name + ' - ' + str(run_comment)

        self.log_dir = os.path.join(self.logger.logging_options['log_root'], self.base_dir, run_name)
        os.makedirs(self.log_dir, exist_ok=False)
        if self.logger.logging_options['save_figure_to_disc_png']:
            os.mkdir(os.path.join(self.log_dir, 'png'))
        if self.logger.logging_options['save_figure_to_disc_svg']:
            os.mkdir(os.path.join(self.log_dir, 'svg'))

        # disable this to continue training?
        e = 0
        self.overhead_mins = 0.0

        print('Started run. Logging to {}'.format(self.log_dir))
        self.fig = plt.figure()

        torch.cuda.empty_cache()

        with SummaryWriter(self.log_dir, flush_secs=30) as writer:
            self.log_once(writer, 0)
            self.log_hyperparams(writer, 0)

            for e in range(e, e + epochs + 1):
                self.training_loop(writer, e)
