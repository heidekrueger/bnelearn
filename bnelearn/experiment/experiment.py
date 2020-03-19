"""Defines experiment class"""

import warnings
from abc import ABC, abstractmethod

import torch

# pylint: disable=unnecessary-pass,unused-argument

from bnelearn.experiment.gpu_controller import GPUController
from bnelearn.experiment.learning_configuration import LearningConfiguration
from bnelearn.experiment.logger import Logger


class Experiment(ABC):
    """Abstract Class representing an experiment"""

    def __init__(self, mechanism_type, gpu_config: GPUController, logger: Logger, l_config: LearningConfiguration,
                 risk: float = 1.0):

        self.l_config = l_config
        self.mechanism_type = mechanism_type
        self.gpu_config = gpu_config
        self.risk = risk
        self.risk_profile = Experiment.get_risk_profile(risk)
        self.logger = logger

        self.base_dir = None
        self.model = None

        # setup bidders
        self.common_prior = None
        self.positive_output_point = None
        self.plot_xmin = None
        self.plot_xmax = None
        self.plot_ymin = None
        self.plot_ymax = None
        self.valuation_prior = None
        self.model_sharing = None
        self.bidders = None

        # setup learner
        self.learner = None

        # setup learning environment
        self.env = None
        self.mechanism = None

        # setup eval environment
        self.bne_env = None
        self.bne_utility = None

        # setup the experiment, don't mess with the order
        self.setup_bidders()
        self.setup_learning_environment()
        self.setup_learners()
        self.setup_eval_environment()
        self.setup_name()

    # ToDO This is a temporary measure
    @abstractmethod
    def setup_name(self):
        """"""
        pass

    @abstractmethod
    def strat_to_bidder(self, strategy, batch_size, player_position=None, cache_actions=False):
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

    @abstractmethod
    def training_loop(self, epoch):
        """Main training loop to be executed in each iteration."""
        pass

    # ToDO Move logging to logger
    def run(self, epochs, n_runs: int = 1, run_comment=None):
        """Runs the experiment implemented by this class for `epochs` number of iterations."""

        seeds = list(range(n_runs))
        for seed in seeds:
            print('Running experiment {}'.format(seed))
            if seed is not None:
                torch.random.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)

                self.logger.log_experiment(model=self.model, env=self.env, run_comment=run_comment,
                                           plot_xmin=self.plot_xmin, plot_xmax=self.plot_xmax,
                                           plot_ymin=self.plot_ymin, plot_ymax=self.plot_ymax,
                                           batch_size=self.l_config.batch_size, optimal_bid=self.optimal_bid)

                # disable this to continue training?
                epoch = 0

                for epoch in range(epochs, epoch + epochs + 1):
                    self.training_loop(epoch=epoch)

            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            if torch.cuda.memory_allocated() > 0:
                warnings.warn('Theres a memory leak')
