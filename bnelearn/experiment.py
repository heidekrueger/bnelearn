"""Defines experiment class"""
import os.path
import time
import warnings
from abc import ABC, abstractmethod

import matplotlib.pyplot as plt
import torch
from torch.utils.tensorboard import SummaryWriter


class Experiment(ABC):
    """Abstract Class representing an experiment"""

    def __init__(self, name, options, device = 'cuda', specific_gpu = 0, seed = None, log_root = None):

        self.base_dir = os.path.join(*name)
        self.log_root = os.path.abspath(log_root) if log_root else '.'

        self.device = device
        self.specific_gpu = specific_gpu
        self.seed = seed
        

        # Set class-specific options
        self.__init_options()

        ## Setup the experiment
        self.setup_game()
        self.setup_players()
        self.setup_learning_environment()
        self.setup_learners()
        self.setup_eval_environment()



    def __init_options(self):
        pass

    @abstractmethod
    def setup_game(self):
        pass

    @abstractmethod
    def setup_players(self):
        pass

    @abstractmethod
    def setup_learning_environment(self):
        pass

    @abstractmethod
    def setup_learners(self):
        pass

    @staticmethod
    def equilibrium_strategy(inputs):
        pass

    @abstractmethod
    def setup_eval_environment(self):
        pass


    def plot(self,fig, plot_data, writer: SummaryWriter or None,
             plot_points_limit: int = 100, save_data_to_disc = False,
             save_png_to_disc = False):
        warnings.warn('no plotting method set!')

    def log_once(self, writer, e):
        pass

    def log_metrics(self, writer, e):
        pass


    def log_hyperparams(self, writer, e):
        pass

    @abstractmethod
    def training_loop(self, writer, e):
        pass

    def run(self, epochs, run_comment = None):
        if os.name == 'nt': raise ValueError('The run_name may not contain : on Windows! (change datetime format to fix this)')
        run_name = time.strftime('%Y-%m-%d %a %H:%M')
        if run_comment:
            run_name = run_name + ' - ' + str(run_comment)

        self.logdir = os.path.join(self.log_root, self.base_dir, run_name)

        # disable this to continue training?
        e = 0
        self.overhead_mins = 0.0

        print(self.logdir)
        self.fig = plt.figure()

        torch.cuda.empty_cache()

        with SummaryWriter(self.logdir, flush_secs=30) as writer:
            self.log_once(writer, 0)
            self.log_hyperparams(writer, 0)

            for e in range(e, e+epochs+1):
                self.training_loop(writer, e)
