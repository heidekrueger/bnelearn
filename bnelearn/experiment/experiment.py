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
from bnelearn.experiment.learning_configuration import LearningConfiguration
from bnelearn.experiment.logger import Logger



class Experiment(ABC):
    """Abstract Class representing an experiment"""

    # abstract fields that must be set in subclass init
    _bidder2model: List[int] = NotImplemented
    n_models: int = NotImplemented
    # TODO: make all fields that MUST be set in subclass abstract members

    def __init__(self, gpu_config: GPUController, experiment_params: dict,
                 l_config: LearningConfiguration, known_bne=False):

        # Configs
        self.l_config = l_config
        self.gpu_config = gpu_config


        # Experiment params
        # TODO: consolidate these!
        self.experiment_params = experiment_params
        self.n_players = experiment_params['n_players']

        # TODO: decouple --> logic should be in subclasses ?
        #if 'valuation_prior' in experiment_params.keys():
        #    self.valuation_prior = experiment_params['valuation_prior']
        #if 'payment_rule' in experiment_params.keys():
        #    self.mechanism_type = experiment_params['payment_rule']

        # TODO: these may possibly stay here, uncommented for now because of added complexity (due to separate regret logging implementation)
        if 'regret_batch_size' in experiment_params.keys():
            self.regret_batch_size = experiment_params['regret_batch_size']
        if 'regret_grid_size' in experiment_params.keys():
            self.regret_grid_size = experiment_params['regret_grid_size']

        # Misc
        self.base_dir = None
        self.models: Iterable[torch.nn.Module] = None


        self.mechanism: Mechanism = None
        self.bidders: Iterable[Bidder] = None
        self.env: Environment = None
        self.learners: Iterable[Learner] = None

        # TODO: remove this? move all logging logic into experiment itself?
        self.logger: Logger = None

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
