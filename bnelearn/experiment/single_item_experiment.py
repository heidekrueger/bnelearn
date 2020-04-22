"""This module implements Experiments on single items"""

import os
import warnings
from abc import ABC
from typing import Callable
from functools import partial
import torch
import numpy as np
from scipy import integrate

from bnelearn.bidder import Bidder
from bnelearn.environment import  AuctionEnvironment
from bnelearn.experiment import Experiment, GPUController
from bnelearn.experiment.configurations import LearningConfiguration, LoggingConfiguration

from bnelearn.learner import ESPGLearner
from bnelearn.mechanism import FirstPriceSealedBidAuction, VickreyAuction
from bnelearn.strategy import NeuralNetStrategy, ClosureStrategy


###############################################################################
#######   Known equilibrium bid functions                                ######
###############################################################################
# Define known BNE functions top level, so they may be pickled for parallelization
# These are called millions of times, so each implementation should be
# setting specific, i.e. there should be NO setting checks at runtime.

def _optimal_bid_single_item_FPSB_generic_prior_risk_neutral(
        valuation: torch.Tensor or np.ndarray or float, n_players: int, prior_cdf: Callable, **kwargs) -> torch.Tensor:
    if not isinstance(valuation, torch.Tensor):
        # For float and numpy --> convert to tensor (relevant for plotting)
        valuation = torch.tensor(valuation, dtype=torch.float)
    # For float / 0d tensors --> unsqueeze to allow list comprehension below
    if valuation.dim() == 0:
        valuation.unsqueeze_(0)
    # shorthand notation for F^(n-1)
    Fpowered = lambda v: torch.pow(prior_cdf(v), n_players - 1)
    # do the calculations
    numerator = torch.tensor(
        [integrate.quad(Fpowered, 0, v)[0] for v in valuation],
            device=valuation.device
    ).reshape(valuation.shape)
    return valuation - numerator / Fpowered(valuation)

def _optimal_bid_FPSB_UniformSymmetricPriorSingleItem(valuation: torch.Tensor, n: int, r: float, u_lo, u_hi, **kwargs) -> torch.Tensor:
    return u_lo + (valuation - u_lo) * (n - 1) / (n - 1.0 + r)

def _truthful_bid(valuation: torch.Tensor, **kwargs) -> torch.Tensor:
    return valuation


# TODO: single item experiment should not be abstract and hold all logic for learning. Only bne needs to go into subclass
class SingleItemExperiment(Experiment, ABC):

    # known issue: pylint doesn't recognize this class as abstract: https://github.com/PyCQA/pylint/commit/4024949f6caf5eff5f3da7ab2b4c3cf2e296472b
    # pylint: disable=abstract-method

    def __init__(self, experiment_config: dict, learning_config: LearningConfiguration,
                  logging_config: LoggingConfiguration, gpu_config: GPUController, known_bne = False):
        super().__init__(experiment_config, learning_config, logging_config, gpu_config, known_bne)
        self.valuation_prior = None

    def _setup_mechanism(self):
        if self.payment_rule == 'first_price':
            self.mechanism = FirstPriceSealedBidAuction(cuda=self.gpu_config.cuda)
        elif self.payment_rule == 'second_price':
            self.mechanism = VickreyAuction(cuda=self.gpu_config.cuda)
        else:
            raise ValueError('Invalid Mechanism type!')

    def _setup_learners(self):
        self.learners = []
        for i in range(len(self.models)):
            self.learners.append(
                ESPGLearner(model=self.models[i], #(known pylint issue for typing.Iterable) pylint: disable=unsubscriptable-object
                            environment=self.env,
                            hyperparams=self.learning_config.learner_hyperparams,
                            optimizer_type=self.learning_config.optimizer,
                            optimizer_hyperparams=self.learning_config.optimizer_hyperparams,
                            strat_to_player_kwargs={"player_position": i}
                            )
                )

    def _setup_learning_environment(self):
        self.env = AuctionEnvironment(self.mechanism, agents=self.bidders,
                                      batch_size=self.learning_config.batch_size, n_players=self.n_players,
                                      strategy_to_player_closure=self._strat_to_bidder)

class SymmetricPriorSingleItemExperiment(SingleItemExperiment):
    """A Single Item Experiment that has the same valuation prior for all participating bidders.
    For risk-neutral agents, a unique BNE is known.
    """
    def __init__(self, experiment_config: dict, learning_config: LearningConfiguration,
                 logging_config: LoggingConfiguration, gpu_config: GPUController, known_bne = False):
        super().__init__(experiment_config, learning_config, logging_config, gpu_config, known_bne=known_bne)
        self.common_prior = None
        self.risk = float(experiment_config.risk)
        self.risk_profile = Experiment.get_risk_profile(self.risk)
        self.model_sharing = experiment_config.model_sharing
        if self.model_sharing:
            self.n_models = 1
            self._bidder2model = [0] * self.n_players
        else:
            self.n_models = self.n_players
            self._bidder2model = list(range(self.n_players))

        # if not given by subclass, implement generic optimal_bid if known
        known_bne = known_bne or \
            experiment_config.payment_rule == 'second_price' or \
            (experiment_config.payment_rule == 'first_price' and self.risk == 1.0)

    def _setup_bidders(self):
        print('Setting up bidders...')

        # Ensure nonnegative output somewhere on relevatn domain to avoid degenerate initializations
        positive_output_point = self.common_prior.mean

        self.models = []

        if self.model_sharing:
            self.models.append(NeuralNetStrategy(
                self.learning_config.input_length, hidden_nodes=self.learning_config.hidden_nodes,
                hidden_activations=self.learning_config.hidden_activations,
                ensure_positive_output=positive_output_point.unsqueeze(0)
            ).to(self.gpu_config.device))

            self.bidders = [self._strat_to_bidder(self.models[0], self.learning_config.batch_size, i)
                            for i in range(self.n_players)]
        else:
            self.bidders = []
            for i in range(self.n_players):
                self.models.append(NeuralNetStrategy(
                    self.learning_config.input_length, hidden_nodes=self.learning_config.hidden_nodes,
                    hidden_activations=self.learning_config.hidden_activations,
                    ensure_positive_output=positive_output_point.unsqueeze(0)
                ).to(self.gpu_config.device))
                self.bidders.append(self._strat_to_bidder(self.models[i], self.learning_config.batch_size, i))

        if self.learning_config.pretrain_iters > 0:
            print('\tpretraining...')
            for i in range(len(self.models)):
                self.models[i].pretrain(self.bidders[i].valuations, self.learning_config.pretrain_iters)

    def _set_symmetric_bne_closure(self):
        # set optimal_bid here, possibly overwritten by subclasses if more specific form is known
        if self.payment_rule == 'first_price' and  self.risk == 1:
            self._optimal_bid = partial(_optimal_bid_single_item_FPSB_generic_prior_risk_neutral,
                                    n_players = self.n_players, prior_cdf = self.common_prior.cdf)
        elif self.payment_rule == 'second_price':
            self._optimal_bid = _truthful_bid
        else:
            # This should never happen due to check in init
            raise ValueError("Trying to set up unknown BNE..." )

    def _get_analytical_bne_utility(self) -> torch.Tensor:
        """Calculates utility in BNE from known closed-form solution (possibly using numerical integration)"""
        if self.payment_rule == 'first_price':
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                # don't print scipy accuracy warnings
                bne_utility, error_estimate = integrate.dblquad(
                    lambda x, v: self.common_prior.cdf(x) ** (self.n_players - 1) * self.common_prior.log_prob(
                        v).exp(),
                    0, float('inf'),  # outer boundaries
                    lambda v: 0, lambda v: v)  # inner boundaries
                if error_estimate > 1e-6:
                    warnings.warn('Error in optimal utility might not be negligible')
        elif self.payment_rule == 'second_price':
            F = self.common_prior.cdf
            f = lambda x: self.common_prior.log_prob(torch.tensor(x)).exp()
            f1n = lambda x, n: n * F(x) ** (n - 1) * f(x)

            bne_utility, error_estimate = integrate.dblquad(
                lambda x, v: (v - x) * f1n(x, self.n_players - 1) * f(v),
                0, float('inf'),  # outer boundaries
                lambda v: 0, lambda v: v)  # inner boundaries

            if error_estimate > 1e-6:
                warnings.warn('Error bound on analytical bne utility is not negligible!')
        else:
            raise ValueError("Invalid auction mechanism.")

        return torch.tensor(bne_utility, device=self.gpu_config.device)

    def _setup_eval_environment(self):

        self._set_symmetric_bne_closure()

        # TODO: parallelism should be taken from elsewhere
        # TODO: existence of valuation_prior not guaranteed
        n_processes_optimal_strategy = 44 if self.valuation_prior != 'uniform' and \
                                             self.payment_rule != 'second_price' else 0
        bne_strategy = ClosureStrategy(self._optimal_bid, parallel=n_processes_optimal_strategy, mute=True)

        # define bne agents once then use them in all runs
        self.bne_env = AuctionEnvironment(
            self.mechanism,
            agents=[self._strat_to_bidder(bne_strategy,
                                          player_position=i,
                                          batch_size=self.logging_config.eval_batch_size,
                                          cache_actions=self.logging_config.cache_eval_actions)
                    for i in range(self.n_players)],
            batch_size=self.logging_config.eval_batch_size,
            n_players=self.n_players,
            strategy_to_player_closure=self._strat_to_bidder
        )

        # Calculate bne_utility via sampling and from known closed form solution and do a sanity check
        #TODO: This is not very precise. Instead we should consider taking the mean over all agents
        bne_utility_sampled = self.bne_env.get_reward(self.bne_env.agents[0], draw_valuations=True)
        bne_utility_analytical = self._get_analytical_bne_utility()

        print('Utility in BNE (sampled): \t{:.5f}'.format(bne_utility_sampled))
        print('Utility in BNE (analytic): \t{:.5f}'.format(bne_utility_analytical))
        # TODO: make atol dynamic based on batch size
        assert torch.allclose(bne_utility_analytical, bne_utility_sampled, atol=5e-2), \
            "Analytical BNE Utility does not match sampled utility from parent class! \n\t sampled {}, analytic {}".format(
                bne_utility_sampled, bne_utility_analytical)
        print('Using analytical BNE utility.')
        self.bne_utility = bne_utility_analytical
        self.bne_utilities = [self.bne_utility]*self.n_models

    def _strat_to_bidder(self, strategy, batch_size, player_position=0, cache_actions=False):
        return Bidder(self.common_prior, strategy, player_position, batch_size, cache_actions=cache_actions, risk=self.risk)

    def _get_logdir(self):
        name = ['single_item', self.payment_rule, self.valuation_prior,
                'symmetric', self.risk_profile, str(self.n_players) + 'p']
        return os.path.join(*name)

class UniformSymmetricPriorSingleItemExperiment(SymmetricPriorSingleItemExperiment):

    def __init__(self, experiment_config: dict, learning_config: LearningConfiguration,
                 logging_config: LoggingConfiguration, gpu_config: GPUController):
        known_bne = experiment_config.payment_rule in ('first_price', 'second_price')
        super().__init__(experiment_config, learning_config, logging_config, gpu_config, known_bne)
        assert experiment_config.u_lo is not None, """Prior boundaries not specified!"""
        assert experiment_config.u_hi is not None, """Prior boundaries not specified!"""
        self.valuation_prior = 'uniform'
        self.u_lo = float(experiment_config.u_lo)
        self.u_hi = float(experiment_config.u_hi)
        self.common_prior = torch.distributions.uniform.Uniform(low = self.u_lo, high=self.u_hi)
        
        self.plot_xmin = self.u_lo
        self.plot_xmax = self.u_hi
        self.plot_ymin = 0
        self.plot_ymax = self.u_hi * 1.05
        self._setup_mechanism_and_eval_environment()

    def _set_symmetric_bne_closure(self):
        # set optimal_bid here, possibly overwritten by subclasses if more specific form is known
        if self.payment_rule == 'first_price':
            self._optimal_bid = partial(_optimal_bid_FPSB_UniformSymmetricPriorSingleItem,
                                        n=self.n_players, r=self.risk, u_lo = self.u_lo, u_hi = self.u_hi)
        elif self.payment_rule == 'second_price':
            self._optimal_bid = _truthful_bid
        else:
            raise ValueError('unknown mechanistm_type')

    def _get_analytical_bne_utility(self):
        if self.payment_rule == 'first_price':
            bne_utility = torch.tensor(
                (self.risk * (self.u_hi - self.u_lo) / (self.n_players - 1 + self.risk)) **
                    self.risk / (self.n_players + self.risk),
                device = self.gpu_config.device
                )
        elif self.payment_rule == 'second_price':
            F = self.common_prior.cdf
            f = lambda x: self.common_prior.log_prob(torch.tensor(x)).exp()
            f1n = lambda x, n: n * F(x) ** (n - 1) * f(x)

            bne_utility, error_estimate = integrate.dblquad(
                lambda x, v: (v - x) * f1n(x, self.n_players - 1) * f(v),
                0, float('inf'),  # outer boundaries
                lambda v: 0, lambda v: v)  # inner boundaries

            bne_utility = torch.tensor(bne_utility, device=self.gpu_config.device)
            if error_estimate > 1e-6:
                warnings.warn('Error bound on analytical bne utility is not negligible!')
        else:
            raise ValueError("Invalid auction mechanism.")

        return bne_utility

class GaussianSymmetricPriorSingleItemExperiment(SymmetricPriorSingleItemExperiment):
    def __init__(self, experiment_config: dict, learning_config: LearningConfiguration,
                 logging_config: LoggingConfiguration, gpu_config: GPUController):
        
        super().__init__(experiment_config, learning_config, logging_config, gpu_config)
        assert experiment_config.valuation_mean is not None, """Valuation mean and/or std not specified!"""
        assert experiment_config.valuation_std is not None, """Valuation mean and/or std not specified!"""
        self.valuation_prior = 'normal'
        self.valuation_mean = experiment_config.valuation_mean
        self.valuation_std = experiment_config.valuation_std
        self.common_prior = torch.distributions.normal.Normal(loc=self.valuation_mean, scale=self.valuation_std)
        self.plot_xmin = int(max(0, self.valuation_mean - 3 * self.valuation_std))
        self.plot_xmax = int(self.valuation_mean + 3 * self.valuation_std)
        self.plot_ymin = 0
        self.plot_ymax = 20 if self.payment_rule == 'first_price' else self.plot_xmax
        self._setup_mechanism_and_eval_environment()

class TwoPlayerAsymmetricUniformPriorSingleItemExperiment(SingleItemExperiment):
    def __init__(self, experiment_config: dict, learning_config: LearningConfiguration,
                 logging_config: LoggingConfiguration, gpu_config: GPUController):
        super().__init__(experiment_config, learning_config, logging_config, gpu_config)
        #TODO: implement optimal bid etc

        raise NotImplementedError()

    def _get_logdir(self):
        NotImplemented

    def _strat_to_bidder(self, strategy, batch_size, player_position=None, cache_actions=False):
        pass

    def _setup_bidders(self):
        pass

    def _setup_eval_environment(self):
        pass



    def _training_loop(self, epoch):
        pass
