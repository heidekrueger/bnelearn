"""Defines experiment class"""
import os.path
import time
import warnings
from abc import ABC, abstractmethod

import matplotlib.pyplot as plt
import torch
from torch.utils.tensorboard import SummaryWriter

# pylint: disable=unnecessary-pass,unused-argument

class Experiment(ABC):
    """Abstract Class representing an experiment"""

    def __init__(self, name, mechanism, n_players, logging_options):
        self.n_players = n_players
        self.mechanism = mechanism

        self.base_dir = os.path.join(*name)
        self._logging_options = logging_options # TODO: add error handling?

        self.log_dir = None # is set dynamically in each run
        self.fig = None     # is set dynamically in each run

        ## Setup the experiment
        self.setup_players()
        self.setup_learning_environment()
        self.setup_learners()
        self.setup_eval_environment()


    @abstractmethod
    def setup_players(self):
        """This method should set up the set of Players/Bidders
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

    @staticmethod
    def equilibrium_strategy(inputs):
        """Defines optimal BNE strategy in this setting"""
        pass

    @abstractmethod
    def setup_eval_environment(self):
        """Sets up an environment used for evaluation of learning agents (e.g.) vs known BNE"""
        pass


    def plot(self, fig, plot_data, writer: SummaryWriter or None, e=None):
        """This method should implement a vizualization of the experiment at the current state"""
        warnings.warn('no plotting method set!')

    def _process_figure(self, fig, writer = None, e=None):
        """displays, logs and/or saves figure built in plot method"""

        if self._logging_options['save_figure_to_disc_png']:
            plt.savefig(os.path.join(self.log_dir, 'png', f'epoch_{e:05}.png'))

        if self._logging_options['save_figure_to_disc_svg']:
            plt.savefig(os.path.join(self.log_dir, 'svg', f'epoch_{e:05}.svg'),
                        format='svg', dpi=1200)
        if writer:
            writer.add_figure('eval/bid_function', fig, e)
        if self._logging_options['show_plot_inline']:
            #display.display(plt.gcf())
            plt.show()

    def log_once(self, writer, e):
        """Logging function called once at the beginning of the experiment."""
        pass

    def log_metrics(self, writer, e):
        """Logging function called after each learning iteration"""
        pass


    def log_hyperparams(self, writer, e):
        """Logging function called when hyperparameters have changed"""
        pass

    @abstractmethod
    def training_loop(self, writer, e):
        """Main training loop to be executed in each iteration."""
        pass

    def run(self, epochs, run_comment = None):
        """Runs the experiment implemented by this class for `epochs` number of iterations."""

        if os.name == 'nt':
            raise ValueError('The run_name may not contain : on Windows!')
        run_name = time.strftime('%Y-%m-%d %a %H:%M:%S')
        if run_comment:
            run_name = run_name + ' - ' + str(run_comment)

        self.log_dir = os.path.join(self._logging_options['log_root'], self.base_dir, run_name)
        os.makedirs(self.log_dir, exist_ok=False)
        if self._logging_options['save_figure_to_disc_png']:
            os.mkdir(os.path.join(self.log_dir, 'png'))
        if self._logging_options['save_figure_to_disc_svg']:
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

            for e in range(e, e+epochs+1):
                self.training_loop(writer, e)
