"""This module implements Experiments on single items"""

import os
import warnings
from abc import ABC
from typing import Callable, List
from functools import partial
import torch
import numpy as np
from scipy import integrate, interpolate
from scipy import optimize

from bnelearn.bidder import Bidder
from bnelearn.environment import AuctionEnvironment
from bnelearn.experiment import Experiment, GPUConfiguration
from bnelearn.experiment.configurations import ModelConfiguration, LearningConfiguration, LoggingConfiguration, \
    ExperimentConfiguration

from bnelearn.mechanism import FirstPriceSealedBidAuction, VickreyAuction
from bnelearn.strategy import ClosureStrategy


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


def _optimal_bid_FPSB_UniformSymmetricPriorSingleItem(valuation: torch.Tensor, n: int, r: float, u_lo, u_hi,
                                                      **kwargs) -> torch.Tensor:
    return u_lo + (valuation - u_lo) * (n - 1) / (n - 1.0 + r)


def _truthful_bid(valuation: torch.Tensor, **kwargs) -> torch.Tensor:
    return valuation


def _optimal_bid_2P_asymmetric_uniform_risk_neutral(valuation: torch.Tensor or float, player_position: int,
                                                    u_lo: List, u_hi: List):
    """
    Optimal bid in this experiment when bidders share same lower bound.
    Source: https://link.springer.com/article/10.1007/BF01271133
    """

    if not isinstance(valuation, torch.Tensor):
        valuation = torch.tensor(valuation, dtype=torch.float)
    # unsqueeze if simple float
    if valuation.dim() == 0:
        valuation.unsqueeze_(0)

    c = 1 / (u_hi[0] - u_lo[0]) ** 2 - 1 / (u_hi[1] - u_lo[0]) ** 2
    factor = 2 * player_position - 1  # -1 for 0 (weak player), +1 for 1 (strong player)
    denominator = 1.0 + torch.sqrt(1 + factor * c * (valuation - u_lo[0]) ** 2)
    bid = u_lo[0] + (valuation - u_lo[0]) / denominator
    return torch.max(bid, torch.zeros_like(bid))


def _optimal_bid_2P_asymmetric_uniform_risk_neutral_multi_lower(u_lo: List, u_hi: List):
    """
    Optimal bid in this experiment when bidders do NOT share same lower bound.
    Source: Equilibrium 1 of https://link.springer.com/article/10.1007/s40505-014-0049-1
    """
    eps = 1e-8
    interpol_points = 256

    # 1. Solve implicit bid function
    v1 = np.linspace(u_lo[0] + eps, u_hi[0] - eps, interpol_points)
    v2 = np.linspace(u_lo[1] + eps, u_hi[1] - eps, interpol_points)

    def inverse_bid_player_1(bid):
        return 36 / ((2 * bid - 6) * (1 / 5) * np.exp(9 / 4 + 6 / (6 - 2 * bid)) + 24 - 4 * bid)

    def inverse_bid_player_2(bid):
        return 6 + 36 / ((2 * bid - 6) * 20 * np.exp(-9 / 4 - 6 / (6 - 2 * bid)) - 4 * bid)

    u_lo_cut = 0
    for i in range(interpol_points):
        if v1[i] > u_lo[1] / 2:
            u_lo_cut = i
            break

    b1 = np.copy(v1)  # truthful at beginning
    b1[u_lo_cut:] = np.array([optimize.broyden1(lambda x: inverse_bid_player_1(x) - v, v)
                              for v in v1[u_lo_cut:]])
    b2 = np.array([optimize.broyden1(lambda x: inverse_bid_player_2(x) - v, v)
                   for v in v2])

    opt_bid_function = [
        interpolate.interp1d(v1, b1, kind=1),
        interpolate.interp1d(v2, b2, kind=1)
    ]

    # 2. return interpolation of bid function
    def _optimal_bid(valuation: torch.Tensor or float, player_position: int):
        if not isinstance(valuation, torch.Tensor):
            valuation = torch.tensor(valuation, dtype=torch.float)
        # unsqueeze if simple float
        if valuation.dim() == 0:
            valuation.unsqueeze_(0)
        bid = torch.tensor(
            opt_bid_function[player_position](valuation.cpu().numpy()),
            device=valuation.device,
            dtype=valuation.dtype
        )
        return bid

    return _optimal_bid


def _optimal_bid_2P_asymmetric_uniform_risk_neutral_multi_lower_2(
        valuation: torch.Tensor or float, player_position: int,
        u_lo: List, u_hi: List
):
    """
    Optimal bid in this experiment when bidders do NOT share same lower bound.
    Source: Equilibrium 2 of https://link.springer.com/article/10.1007/s40505-014-0049-1
    """
    if not isinstance(valuation, torch.Tensor):
        valuation = torch.tensor(valuation, dtype=torch.float)
    # unsqueeze if simple float
    if valuation.dim() == 0:
        valuation.unsqueeze_(0)

    if player_position == 0:
        bids = torch.zeros_like(valuation)
        bids[valuation > 4] = valuation[valuation > 4] / 2 + 2
        bids[valuation <= 4] = valuation[valuation <= 4] / 4 + 3
    else:
        bids = valuation / 2 + 1

    return bids


def _optimal_bid_2P_asymmetric_uniform_risk_neutral_multi_lower_3(
        valuation: torch.Tensor or float, player_position: int,
        u_lo: List, u_hi: List
):
    """
    Optimal bid in this experiment when bidders do NOT share same lower bound.
    Source: Equilibrium 3 of https://link.springer.com/article/10.1007/s40505-014-0049-1
    """
    if not isinstance(valuation, torch.Tensor):
        valuation = torch.tensor(valuation, dtype=torch.float)
    # unsqueeze if simple float
    if valuation.dim() == 0:
        valuation.unsqueeze_(0)

    if player_position == 0:
        bids = valuation / 5 + 4
    else:
        bids = 5 * torch.ones_like(valuation)

    return bids


# TODO: single item experiment should not be abstract and hold all logic for learning.
# Only bne needs to go into subclass
class SingleItemExperiment(Experiment, ABC):

    # known issue: pylint doesn't recognize this class as abstract:
    # https://github.com/PyCQA/pylint/commit/4024949f6caf5eff5f3da7ab2b4c3cf2e296472b
    # pylint: disable=abstract-method

    def __init__(self, experiment_config: ExperimentConfiguration):
        self.experiment_config = experiment_config
        if not hasattr(self, 'payment_rule'):
            self.payment_rule = self.experiment_config.model_config.payment_rule
        if not hasattr(self, 'valuation_prior'):
            self.valuation_prior = 'unknown'

        self.n_items = 1
        self.input_length = 1
        super().__init__(experiment_config=experiment_config)

    def _setup_mechanism(self):
        if self.payment_rule == 'first_price':
            self.mechanism = FirstPriceSealedBidAuction(cuda=self.gpu_config.cuda)
        elif self.payment_rule == 'second_price':
            self.mechanism = VickreyAuction(cuda=self.gpu_config.cuda)
        else:
            raise ValueError('Invalid Mechanism type!')

    @staticmethod
    def get_risk_profile(risk) -> str:
        """Used for logging and checking existence of bne"""
        if risk == 1.0:
            return 'risk_neutral'
        elif risk == 0.5:
            return 'risk_averse'
        else:
            return 'other'


class SymmetricPriorSingleItemExperiment(SingleItemExperiment):
    """A Single Item Experiment that has the same valuation prior for all participating bidders.
    For risk-neutral agents, a unique BNE is known.
    """

    def __init__(self, experiment_config: ExperimentConfiguration):
        self.experiment_config = experiment_config
        self.n_players = self.experiment_config.run_config.n_players
        self.n_items = 1

        self.common_prior = self.experiment_config.model_config.common_prior
        self.positive_output_point = torch.stack([self.common_prior.mean] * self.n_items)

        self.risk = float(self.experiment_config.model_config.risk)
        self.risk_profile = self.get_risk_profile(self.risk)

        # if not given by subclass, implement generic optimal_bid if known
        self.experiment_config.model_config.known_bne = \
            self.experiment_config.model_config.known_bne or \
            self.experiment_config.model_config.payment_rule == 'second_price' \
            or (self.experiment_config.model_config.payment_rule == 'first_price' and self.risk == 1.0)

        self.model_sharing = self.experiment_config.model_config.model_sharing
        if self.model_sharing:
            self.n_models = 1
            self._bidder2model = [0] * self.n_players
        else:
            self.n_models = self.n_players
            self._bidder2model = list(range(self.n_players))

        super().__init__(experiment_config=experiment_config)

    def _set_symmetric_bne_closure(self):
        # set optimal_bid here, possibly overwritten by subclasses if more specific form is known
        if self.payment_rule == 'first_price' and self.risk == 1:
            self._optimal_bid = partial(_optimal_bid_single_item_FPSB_generic_prior_risk_neutral,
                                        n_players=self.n_players, prior_cdf=self.common_prior.cdf)
        elif self.payment_rule == 'second_price':
            self._optimal_bid = _truthful_bid
        else:
            # This should never happen due to check in init
            raise ValueError("Trying to set up unknown BNE...")

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

        # TODO: parallelism should be taken from elsewhere. Should be moved to config. Assigned @Stefan
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
        # TODO: This is not very precise. Instead we should consider taking the mean over all agents
        bne_utility_sampled = self.bne_env.get_reward(self.bne_env.agents[0], draw_valuations=True)
        bne_utility_analytical = self._get_analytical_bne_utility()

        print('Utility in BNE (sampled): \t{:.5f}'.format(bne_utility_sampled))
        print('Utility in BNE (analytic): \t{:.5f}'.format(bne_utility_analytical))
        # TODO: make atol dynamic based on batch size to avoid false positives in test runs.
        if not torch.allclose(bne_utility_analytical, bne_utility_sampled, atol=5e-2):
            warnings.warn(
                "Analytical BNE Utility does not match sampled utility from parent class! \n\t sampled {}, analytic {}"
                    .format(bne_utility_sampled, bne_utility_analytical))
        print('Using analytical BNE utility.')
        self.bne_utility = bne_utility_analytical
        self.bne_utilities = [self.bne_utility] * self.n_models

    def _strat_to_bidder(self, strategy, batch_size, player_position=0, cache_actions=False):
        return Bidder(self.common_prior, strategy, player_position, batch_size, cache_actions=cache_actions,
                      risk=self.risk)

    def _get_logdir_hierarchy(self):
        name = ['single_item', self.payment_rule, self.valuation_prior,
                'symmetric', self.risk_profile, str(self.n_players) + 'p']
        return os.path.join(*name)


class UniformSymmetricPriorSingleItemExperiment(SymmetricPriorSingleItemExperiment):

    def __init__(self, experiment_config: ExperimentConfiguration):
        self.experiment_config = experiment_config

        assert self.experiment_config.model_config.u_lo is not None, """Prior boundaries not specified!"""
        assert self.experiment_config.model_config.u_hi is not None, """Prior boundaries not specified!"""

        self.experiment_config.model_config.known_bne = \
            self.experiment_config.model_config.payment_rule in ('first_price', 'second_price')

        self.valuation_prior = 'uniform'
        self.u_lo = self.experiment_config.model_config.u_lo
        self.u_hi = self.experiment_config.model_config.u_hi
        self.experiment_config.model_config.common_prior = \
            torch.distributions.uniform.Uniform(low=self.u_lo, high=self.u_hi)

        # ToDO Implicit list to float type conversion
        self.plot_xmin = self.u_lo
        self.plot_xmax = self.u_hi
        self.plot_ymin = 0
        self.plot_ymax = self.u_hi * 1.05

        super().__init__(experiment_config=experiment_config)

    def _set_symmetric_bne_closure(self):
        # set optimal_bid here, possibly overwritten by subclasses if more specific form is known
        if self.payment_rule == 'first_price':
            self._optimal_bid = partial(_optimal_bid_FPSB_UniformSymmetricPriorSingleItem,
                                        n=self.n_players, r=self.risk, u_lo=self.u_lo, u_hi=self.u_hi)
        elif self.payment_rule == 'second_price':
            self._optimal_bid = _truthful_bid
        else:
            raise ValueError('unknown mechanistm_type')

    def _get_analytical_bne_utility(self):
        if self.payment_rule == 'first_price':
            bne_utility = torch.tensor(
                (self.risk * (self.u_hi - self.u_lo) / (self.n_players - 1 + self.risk)) **
                self.risk / (self.n_players + self.risk),
                device=self.gpu_config.device
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
    def __init__(self, experiment_config: ExperimentConfiguration):
        self.experiment_config = experiment_config
        assert self.experiment_config.model_config.valuation_mean is not None, """Valuation mean and/or std not 
        specified! """
        assert self.experiment_config.model_config.valuation_std is not None, """Valuation mean and/or std not 
        specified! """
        self.valuation_prior = 'normal'
        self.valuation_mean = self.experiment_config.model_config.valuation_mean
        self.valuation_std = self.experiment_config.model_config.valuation_std
        self.experiment_config.model_config.common_prior = \
            torch.distributions.normal.Normal(loc=self.valuation_mean, scale=self.valuation_std)

        self.plot_xmin = int(max(0, self.valuation_mean - 3 * self.valuation_std))
        self.plot_xmax = int(self.valuation_mean + 3 * self.valuation_std)
        self.plot_ymin = 0
        self.plot_ymax = 20 if self.experiment_config.model_config.payment_rule == 'first_price' else self.plot_xmax

        super().__init__(experiment_config=experiment_config)


class TwoPlayerAsymmetricUniformPriorSingleItemExperiment(SingleItemExperiment):
    def __init__(self, experiment_config: ExperimentConfiguration):
        self.experiment_config = experiment_config

        if self.experiment_config.model_config.model_sharing is not None:
            assert not self.experiment_config.model_config.model_sharing, "Model sharing not available in this setting!"
        self.model_sharing = False

        self.payment_rule = 'first_price'
        self.valuation_prior = 'uniform'
        self.risk = float(self.experiment_config.model_config.risk)
        self.risk_profile = self.get_risk_profile(self.risk)

        self.n_players = 2
        self.n_items = 1
        self.n_models = self.n_players
        self._bidder2model: List[int] = list(range(self.n_players))

        if not isinstance(self.experiment_config.model_config.u_lo, list):
            self.u_lo = [float(self.experiment_config.model_config.u_lo)] * self.n_players
        else:
            self.u_lo: List[float] = [float(self.experiment_config.model_config.u_lo[i]) for i in range(self.n_players)]
        self.u_hi: List[float] = [float(self.experiment_config.model_config.u_hi[i]) for i in range(self.n_players)]
        assert self.u_hi[0] < self.u_hi[1], "First Player must be the weaker player"
        self.positive_output_point = torch.tensor([min(self.u_hi)] * self.n_items)

        self.plot_xmin = min(self.u_lo)
        self.plot_xmax = max(self.u_hi)
        self.plot_ymin = self.plot_xmin * 0.90
        self.plot_ymax = self.plot_xmax * 1.05

        self.experiment_config.model_config.known_bne = True  # TODO: check additional requirements, i.e. risk

        assert self.risk == 1.0, "BNE only known for risk neutral bidders."

        super().__init__(experiment_config=experiment_config)

    def _get_logdir_hierarchy(self):
        name = ['single_item', self.payment_rule, self.valuation_prior,
                'asymmetric', self.risk_profile, str(self.n_players) + 'p']
        return os.path.join(*name)

    def _strat_to_bidder(self, strategy, batch_size, player_position=None, cache_actions=False):
        return Bidder.uniform(self.u_lo[player_position], self.u_hi[player_position], strategy,
                              player_position=player_position, batch_size=batch_size)

    def _setup_eval_environment(self):

        if len(set(self.u_lo)) != 1:  # BNE for differnt u_lo for each player
            print('Warning: only one of multiple BNE selected!')  # TODO @Nils
            # BNE 1
            # # self._optimal_bid = _optimal_bid_2P_asymmetric_uniform_risk_neutral_multi_lower(
            #     u_lo=self.u_lo, u_hi=self.u_hi
            # )

            # BNE 2
            self._optimal_bid = partial(_optimal_bid_2P_asymmetric_uniform_risk_neutral_multi_lower_2,
                                        u_lo=self.u_lo, u_hi=self.u_hi)

            # # BNE 3
            # self._optimal_bid = partial(_optimal_bid_2P_asymmetric_uniform_risk_neutral_multi_lower_3,
            #                             u_lo=self.u_lo, u_hi=self.u_hi)

        else:  # BNE for fixed u_lo for all players
            self._optimal_bid = partial(_optimal_bid_2P_asymmetric_uniform_risk_neutral,
                                        u_lo=self.u_lo, u_hi=self.u_hi)

        bne_strategies = [ClosureStrategy(partial(self._optimal_bid, player_position=i))
                          for i in range(self.n_players)]

        self.bne_env = AuctionEnvironment(
            mechanism=self.mechanism,
            agents=[self._strat_to_bidder(bne_strategies[i], player_position=i,
                                          batch_size=self.logging_config.eval_batch_size)
                    for i in range(self.n_players)],
            n_players=self.n_players,
            batch_size=self.logging_config.eval_batch_size,
            strategy_to_player_closure=self._strat_to_bidder
        )

        bne_utilities_sampled = torch.tensor(
            [self.bne_env.get_reward(a, draw_valuations=True) for a in self.bne_env.agents])

        print(('Utilities in BNE (sampled):' + '\t{:.5f}' * self.n_players + '.').format(*bne_utilities_sampled))
        print("No closed form solution for BNE utilities available in this setting. Using sampled value as baseline.")

        print('Debug: eval_batch size:{}'.format(self.bne_env.batch_size))
        if self.u_lo == 5. and self.u_hi[0] == 15. and self.u_hi[1] == 25. and self.bne_env.batch_size <= 2 ** 22:
            # replace by known optimum with higher precision
            bne_utilities_sampled = torch.tensor([0.9694, 5.0688])  # calculated using 100x batch size above
            print("\tReplacing sampled bne utilities by precalculated utilities with higher precision: {}".format(
                bne_utilities_sampled))

        self.bne_utilities = bne_utilities_sampled
