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

    def __init__(self, gpu_config: GPUController, experiment_params: dict,
                 logger: Logger, l_config: LearningConfiguration):

        # Configs
        self.l_config = l_config
        self.gpu_config = gpu_config
        self.logger = logger

        # Experiment params
        self.experiment_params = experiment_params
        self.n_players = experiment_params['n_players']
        self.common_prior = experiment_params['common_prior']
        self.u_lo = experiment_params['u_lo']
        self.u_hi = experiment_params['u_hi']
        self.plot_xmin = min(experiment_params['u_lo'])
        self.plot_xmax = max(experiment_params['u_hi']) * 1.2
        self.plot_ymin = min(experiment_params['u_lo'])
        self.plot_ymax = max(experiment_params['u_hi']) * 1.2
        self.valuation_prior = experiment_params['valuation_prior']
        self.model_sharing = experiment_params['model_sharing']
        self.risk = experiment_params['risk']
        self.risk_profile = Experiment.get_risk_profile(self.risk)
        self.mechanism_type = experiment_params['payment_rule']

        # Misc
        self.base_dir = None
        self.models = None

        # setup bidders        
        self.positive_output_point = None
        self.bidders = None

        # setup learner
        self.learner = None

        # setup learning environment
        self.env = None
        self.mechanism = None

        # setup eval environment
        self.bne_env = None
        self.bne_utility = None

    def _run_setup(self):
        # setup the experiment, don't mess with the order
        self._setup_bidders()
        self._setup_learning_environment()
        self._setup_learners()
        self._setup_eval_environment()
        self._setup_name()

    # ToDO This is a temporary measure
    @abstractmethod
    def _setup_name(self):
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

    @abstractmethod
    def _setup_eval_environment(self):
        """Sets up an environment used for evaluation of learning agents (e.g.) vs known BNE"""
        pass

    @abstractmethod
    def _optimal_bid(self, valuation, player_position):
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
    def _training_loop(self, epoch):
        """Main training loop to be executed in each iteration."""
        pass

    def run(self, epochs, n_runs: int = 1, run_comment=None):
        """Runs the experiment implemented by this class for `epochs` number of iterations."""

        seeds = list(range(n_runs))
        for seed in seeds:
            print('Running experiment {}'.format(seed))
            if seed is not None:
                torch.random.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)

                self.logger.log_experiment(experiment_params=self.experiment_params, models=self.models, max_epochs = epochs, env=self.env,
                                           run_comment=run_comment, plot_xmin=self.plot_xmin, plot_xmax=self.plot_xmax,
                                           plot_ymin=self.plot_ymin, plot_ymax=self.plot_ymax,
                                           batch_size=self.l_config.batch_size, optimal_bid=self._optimal_bid)

                # disable this to continue training?
                epoch = 0

                for epoch in range(epoch, epoch + epochs + 1):
                    self._training_loop(epoch=epoch)

            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            # if torch.cuda.memory_allocated() > 0:
            #    warnings.warn('Theres a memory leak')
