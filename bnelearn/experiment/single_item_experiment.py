"""This module implements Experiments on single items"""

import os
import warnings
from abc import ABC
from functools import partial
from typing import List

import torch
from bnelearn.bidder import Bidder
from bnelearn.environment import AuctionEnvironment
from bnelearn.experiment.configurations import ExperimentConfig
from bnelearn.experiment.equilibria import (
    bne_fpsb_ipv_asymmetric_uniform_overlapping_priors_risk_neutral,
    bne1_kaplan_zhamir,
    bne2_kaplan_zhamir,
    bne3_kaplan_zhamir,
    bne_fpsb_ipv_symmetric_uniform_prior,
    bne_2p_affiliated_values,
    bne_3p_mineral_rights,
    bne_fpsb_ipv_symmetric_generic_prior_risk_neutral, truthful_bid)
from bnelearn.mechanism import FirstPriceSealedBidAuction, VickreyAuction
from bnelearn.sampler import (AffiliatedValuationObservationSampler,
                              CompositeValuationObservationSampler,
                              MineralRightsValuationObservationSampler,
                              SymmetricIPVSampler, UniformSymmetricIPVSampler)
from bnelearn.strategy import ClosureStrategy
from bnelearn.util.distribution_util import copy_dist_to_device
from scipy import integrate

from .experiment import Experiment


# TODO: single item experiment should not be abstract and hold all logic for learning.
# Only bne needs to go into subclass
class SingleItemExperiment(Experiment, ABC):

    # known issue: pylint doesn't recognize this class as abstract:
    # https://github.com/PyCQA/pylint/commit/4024949f6caf5eff5f3da7ab2b4c3cf2e296472b
    # pylint: disable=abstract-method

    def __init__(self, config: ExperimentConfig):
        self.config = config

        # TODO Stefan: Can we get rid of this procedural code?
        if not hasattr(self, 'payment_rule'):
            self.payment_rule = self.config.setting.payment_rule
        if not hasattr(self, 'valuation_prior'):
            self.valuation_prior = 'unknown'

        self.observation_size = self.valuation_size = self.action_size = 1
        if self.config.logging.eval_batch_size < 2 ** 20:
            print(f"Using small eval_batch_size of {self.config.logging.eval_batch_size}. Use at least 2**22 for proper experiment runs!")
        super().__init__(config=config)

    def _setup_mechanism(self):
        if self.payment_rule == 'first_price':
            self.mechanism = FirstPriceSealedBidAuction(cuda=self.hardware.cuda)
        elif self.payment_rule == 'second_price':
            self.mechanism = VickreyAuction(cuda=self.hardware.cuda)
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

    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.n_players = self.config.setting.n_players

        # instance property will be set in super().__init__ call.
        action_size = 1

        # TODO: common_prior possibly on wrnog device now
        self.common_prior = self.config.setting.common_prior
        self.positive_output_point = torch.stack([self.common_prior.mean] * action_size)

        self.risk = float(self.config.setting.risk)
        self.risk_profile = self.get_risk_profile(self.risk)

        self.model_sharing = self.config.learning.model_sharing
        if self.model_sharing:
            self.n_models = 1
            self._bidder2model = [0] * self.n_players
        else:
            self.n_models = self.n_players
            self._bidder2model = list(range(self.n_players))

        super().__init__(config=config)

    def _setup_sampler(self):
        self.sampler = SymmetricIPVSampler(
            self.common_prior, self.n_players, self.valuation_size,
            self.config.learning.batch_size, self.config.hardware.device
        )

    def _check_and_set_known_bne(self):
        if self.payment_rule == 'first_price' and self.risk == 1:
            # cdf_cpu = copy_dist_to_device(self.common_prior, 'cpu').cdf
            self._optimal_bid = partial(bne_fpsb_ipv_symmetric_generic_prior_risk_neutral,
                                        n_players=self.n_players, prior_cdf=self.common_prior.cdf)
            return True
        elif self.payment_rule == 'second_price':
            self._optimal_bid = truthful_bid
            return True
        else:
            # no bne found, defer to parent
            return super()._check_and_set_known_bne()

    def _get_analytical_bne_utility(self) -> torch.Tensor:
        """Calculates utility in BNE from known closed-form solution (possibly using numerical integration)"""
        # Note: GPU integrals via torchquad do not support non-rectangular bounds
        # could still loop over bounds but not worth it(?)
        if self.payment_rule == 'first_price' and self.risk == 1:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                # don't print scipy accuracy warnings
                bne_utility, error_estimate = integrate.dblquad(
                    lambda x, v: self.common_prior.cdf(torch.tensor(x)) ** (self.n_players - 1) \
                        * self.common_prior.log_prob(torch.tensor(v)).exp(),
                    0, float('inf'),  # outer boundaries
                    lambda v: 0, lambda v: v)  # inner boundaries
                if error_estimate > 1e-6:
                    warnings.warn('Error in optimal utility might not be negligible')
        elif self.payment_rule == 'second_price':
            F = lambda x: self.common_prior.cdf(torch.tensor(x))
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

        return torch.tensor(bne_utility, device=self.hardware.device)

    def _setup_eval_environment(self):
        """Determines whether a bne exists and sets up eval environment."""

        assert self.known_bne
        assert  hasattr(self, '_optimal_bid')
        print("Setting up the evaluation environment..." + \
            "\tDepending on your and hardware and the eval_batch_size, this may take a while," +\
                "-- sequential numeric integration on the cpu is required in this environment.")

        n_processes_optimal_strategy = self.config.hardware.max_cpu_threads if self.valuation_prior != 'uniform' and \
                                                self.payment_rule != 'second_price' else 0
        bne_strategy = ClosureStrategy(self._optimal_bid, parallel=n_processes_optimal_strategy, mute=True)


        # define bne agents once then use them in all runs
        self.bne_env = AuctionEnvironment(
            mechanism = self.mechanism,
            agents=[self._strat_to_bidder(bne_strategy,
                                          player_position=i,
                                          batch_size=self.logging.eval_batch_size,
                                          enable_action_caching=self.logging.cache_eval_actions)
                    for i in range(self.n_players)],
            valuation_observation_sampler = self.sampler,
            batch_size=self.logging.eval_batch_size,
            n_players=self.n_players,
            strategy_to_player_closure=self._strat_to_bidder
        )

        # Calculate bne_utility via sampling and from known closed form solution and do a sanity check
        # TODO: This is not very precise. Instead we should consider taking the mean over all agents
        bne_utility_sampled = self.bne_env.get_reward(self.bne_env.agents[0], redraw_valuations=True)
        bne_utility_analytical = self._get_analytical_bne_utility()

        print('Utility in BNE (sampled): \t{:.5f}'.format(bne_utility_sampled))
        print('Utility in BNE (analytic): \t{:.5f}'.format(bne_utility_analytical))

        # don't print the warning for small batch_sizes (i.e. in test suite)
        if self.logging.eval_batch_size > 2**16 and \
            not torch.allclose(bne_utility_analytical, bne_utility_sampled, atol=5e-2):
            warnings.warn(
                "Analytical BNE Utility does not match sampled utility from parent class! \n\t sampled {}, analytic {}"
                    .format(bne_utility_sampled, bne_utility_analytical))
        print('Using analytical BNE utility.')
        self.bne_utility = bne_utility_analytical
        self.bne_utilities = [self.bne_utility] * self.n_models

    def _strat_to_bidder(self, strategy, batch_size, player_position=0, enable_action_caching=False):
        return Bidder(strategy, player_position, batch_size, enable_action_caching=enable_action_caching,
                      risk=self.risk)

    def _get_logdir_hierarchy(self):
        name = ['single_item', self.payment_rule, self.valuation_prior,
                'symmetric', self.risk_profile, str(self.n_players) + 'p']
        return os.path.join(*name)


class UniformSymmetricPriorSingleItemExperiment(SymmetricPriorSingleItemExperiment):

    def __init__(self, config: ExperimentConfig):
        self.config = config

        u_lo = self.config.setting.u_lo
        u_hi = self.config.setting.u_hi

        assert u_lo is not None and len(set(u_lo)) == 1, "Invalid prior boundaries. u_lo should be a list of length 1."
        assert u_hi is not None and len(set(u_hi)) == 1, "Invalid prior boundaries. u_hi should be a list of length 1."

        self.valuation_prior = 'uniform'
        self.u_lo = torch.tensor(u_lo[0], dtype=torch.float32,
            device=self.config.hardware.device)
        self.u_hi = torch.tensor(u_hi[0], dtype=torch.float32,
            device=self.config.hardware.device)
        self.config.setting.common_prior = \
            torch.distributions.uniform.Uniform(low=self.u_lo, high=self.u_hi)

        # ToDO Implicit list to float type conversion
        self.plot_xmin = self.u_lo.cpu()
        self.plot_xmax = self.u_hi.cpu()
        self.plot_ymin = 0
        self.plot_ymax = self.u_hi.cpu() * 1.05

        super().__init__(config=config)

    def _check_and_set_known_bne(self):
        if self.payment_rule == 'first_price':
            self._optimal_bid = partial(bne_fpsb_ipv_symmetric_uniform_prior,
                                        n=self.n_players, r=self.risk, u_lo=self.u_lo, u_hi=self.u_hi)
            return True
        elif self.payment_rule == 'second_price':
            self._optimal_bid = truthful_bid
            return True
        else: # no bne found, defer to parent
            return super()._check_and_set_known_bne()

    def _get_analytical_bne_utility(self):
        """Get bne utility from known closed-form solution for higher precision."""
        if self.payment_rule == 'first_price':
            bne_utility = torch.tensor(
                (self.risk * (self.u_hi - self.u_lo) / (self.n_players - 1 + self.risk)) **
                self.risk / (self.n_players + self.risk),
                device=self.hardware.device
            )
        elif self.payment_rule == 'second_price':
            F = self.common_prior.cdf
            f = lambda x: self.common_prior.log_prob(torch.tensor(x)).exp()
            f1n = lambda x, n: n * F(x) ** (n - 1) * f(x)

            bne_utility, error_estimate = integrate.dblquad(
                lambda x, v: (v - x) * f1n(x, self.n_players - 1) * f(v),
                0, float('inf'),  # outer boundaries
                lambda v: 0, lambda v: v)  # inner boundaries

            bne_utility = torch.tensor(bne_utility, device=self.hardware.device)
            if error_estimate > 1e-6:
                warnings.warn('Error bound on analytical bne utility is not negligible!')
        else:
            raise ValueError("Invalid auction mechanism.")

        return bne_utility


class GaussianSymmetricPriorSingleItemExperiment(SymmetricPriorSingleItemExperiment):
    def __init__(self, config: ExperimentConfig):
        self.config = config
        assert self.config.setting.valuation_mean is not None, """Valuation mean and/or std not specified! """
        assert self.config.setting.valuation_std is not None, """Valuation mean and/or std not specified! """
        self.valuation_prior = 'normal'
        self.valuation_mean = torch.tensor(
            self.config.setting.valuation_mean, dtype=torch.float32,
            device=self.config.hardware.device)

        self.plot_xmin = int(max(0, self.valuation_mean - 3 * self.valuation_std))
        self.plot_xmax = int(self.valuation_mean + 3 * self.valuation_std)
        self.plot_ymax = 20 if self.config.setting.payment_rule == 'first_price' else self.plot_xmax

        super().__init__(config=config)

    def _setup_sampler(self):
        self.sampler = GaussianSymmetricIPVSampler(
            mean=self.valuation_mean, stddev=self.valuation_std,
            n_players=self.n_players, valuation_size=self.valuation_size,
            default_batch_size=self.config.learning.batch_size,
            default_device=self.config.hardware.device
        )

class TwoPlayerAsymmetricUniformPriorSingleItemExperiment(SingleItemExperiment):
    def __init__(self, config: ExperimentConfig):
        self.config = config

        if self.config.learning.model_sharing is not None:
            assert not self.config.learning.model_sharing, "Model sharing not available in this setting!"
        self.model_sharing = False

        self.payment_rule = 'first_price'
        self.valuation_prior = 'uniform'
        self.risk = float(self.config.setting.risk)
        self.risk_profile = self.get_risk_profile(self.risk)

        n_items = 1
        self.n_players = 2
        self.n_models = self.n_players
        self._bidder2model: List[int] = list(range(self.n_players))

        self.u_lo: List[float] = [float(self.config.setting.u_lo[i]) for i in range(self.n_players)]
        self.u_hi: List[float] = [float(self.config.setting.u_hi[i]) for i in range(self.n_players)]
        assert self.u_hi[0] < self.u_hi[1], "First Player must be the weaker player"
        self.positive_output_point = torch.tensor([min(self.u_hi)] * n_items)

        self.plot_xmin = min(self.u_lo)
        self.plot_xmax = max(self.u_hi)
        self.plot_ymin = self.plot_xmin * 0.90
        self.plot_ymax = self.plot_xmax * 1.05

        super().__init__(config=config)

    def _setup_sampler(self):

        default_batch_size = self.learning.batch_size
        device = self.hardware.device
        # setup individual samplers for each bidder
        bidder_samplers = [
            UniformSymmetricIPVSampler(
                self.u_lo[i], self.u_hi[i], 1,
                self.valuation_size, default_batch_size, device)
            for i in range(self.n_players)]

        self.sampler = CompositeValuationObservationSampler(
            self.n_players, self.valuation_size, self.observation_size, bidder_samplers,
            default_batch_size, device
            )

    def _get_logdir_hierarchy(self):
        name = ['single_item', self.payment_rule, self.valuation_prior,
                'asymmetric', self.risk_profile, str(self.n_players) + 'p']
        return os.path.join(*name)

    def _strat_to_bidder(self, strategy, batch_size, player_position=None, **strat_to_player_kwargs):
        return Bidder(strategy, player_position=player_position, batch_size=batch_size, **strat_to_player_kwargs)

    def _check_and_set_known_bne(self):
        """Checks whether a BNE is known for this experiment and sets the corresponding
           `_optimal_bid` function.
        """
        if self.risk == 1.0:
            if self.u_lo[0] != self.u_lo[1]:  # Agents do not share same u_lo
                # Check for bounds match from Kaplan & Zamir [2015]
                if self.setting.u_lo == [0, 6] and self.setting.u_hi == [5, 7]:
                    self._optimal_bid = [
                        # BNE 1
                        bne1_kaplan_zhamir(u_lo=self.u_lo, u_hi=self.u_hi),
                        # BNE 2
                        partial(bne2_kaplan_zhamir, u_lo=self.u_lo, u_hi=self.u_hi),
                        # BNE 3
                        partial(bne3_kaplan_zhamir, u_lo=self.u_lo, u_hi=self.u_hi)
                    ]
                    return True
            else:  # BNE for shared u_lo for all players from Plum [1992]
                self._optimal_bid = [partial(bne_fpsb_ipv_asymmetric_uniform_overlapping_priors_risk_neutral,
                                             u_lo=self.u_lo, u_hi=self.u_hi)]
                return True

        # Found no BNE
        return super()._check_and_set_known_bne()

    def _setup_eval_environment(self):
        assert self.known_bne
        assert hasattr(self, '_optimal_bid')

        bne_strategies = [None] * len(self._optimal_bid)
        self.bne_env = [None] * len(self._optimal_bid)
        bne_utilities_sampled = [None] * len(self._optimal_bid)
        for i, bid_function in enumerate(self._optimal_bid):
            bne_strategies[i] = [ClosureStrategy(partial(bid_function, player_position=j))
                                 for j in range(self.n_players)]

            self.bne_env[i] = AuctionEnvironment(
                mechanism=self.mechanism,
                agents=[self._strat_to_bidder(bne_strategies[i][p], player_position=p,
                                              batch_size=self.logging.eval_batch_size,
                                              enable_action_caching=self.config.logging.cache_eval_actions)
                        for p in range(self.n_players)],
                valuation_observation_sampler=self.sampler,
                n_players=self.n_players,
                batch_size=self.logging.eval_batch_size,
                strategy_to_player_closure=self._strat_to_bidder
            )

            bne_utilities_sampled[i] = torch.tensor(
                [self.bne_env[i].get_reward(a, redraw_valuations=True) for a in self.bne_env[i].agents])

            print(('Utilities in BNE{} (sampled):' + '\t{:.5f}' * self.n_players + '.') \
                .format(i + 1,*bne_utilities_sampled[i]))

        if len(set(self.u_lo)) == 1:
            print("No closed form solution for BNE utilities available in this setting. Using sampled value as baseline.")

        print('Debug: eval_batch size: {}'.format(self.bne_env[0].batch_size))

        # TODO Stefan: generalize using analytical/hardcoded utilities over all settings!
        # In case of 'canonica' overlapping setting, use precomputed bne-utils with higher precision.
        if len(self.bne_env) == 1 and \
            self.u_lo[0] == self.u_lo[1] == 5.0 and self.u_hi[0] ==15. and self.u_hi[1] == 25. and \
            self.bne_env[0].batch_size <= 2**22:
            # replace by known optimum with higher precision
            bne_utilities_sampled[0] = torch.tensor([0.9694, 5.0688]) # calculated using 100x batch size above
            print(f"\tReplacing sampled bne utilities by precalculated utilities with higher precision: {bne_utilities_sampled[0]}")

        self.bne_utilities = bne_utilities_sampled


class MineralRightsExperiment(SingleItemExperiment):
    """A Single Item Experiment that has the same valuation prior for all participating bidders.
    For risk-neutral agents, a unique BNE is known.
    """

    def __init__(self, config: ExperimentConfig):
        self.n_players = config.setting.n_players

        u_lo = config.setting.u_lo
        u_hi = config.setting.u_hi

        assert len(set(u_lo)) == 1, "Symmetric prior supported only!"
        assert len(set(u_hi)) == 1, "Symmetric prior supported only!"

        self.n_items = 1

        self.valuation_prior = 'uniform'
        self.u_lo = float(u_lo[0])
        self.u_hi = float(u_hi[0])
        self.common_prior = torch.distributions.uniform.Uniform(low=self.u_lo, high=self.u_hi)
        self.positive_output_point = torch.tensor([(self.u_lo+self.u_hi)/2] * self.n_items)

        self.risk = float(config.setting.risk)
        self.risk_profile = self.get_risk_profile(self.risk)

        self.correlation_groups = config.setting.correlation_groups

        assert len(config.setting.correlation_coefficients) == 1

        self.model_sharing = config.learning.model_sharing
        if self.model_sharing:
            self.n_models = 1
            self._bidder2model = [0] * self.n_players
        else:
            self.n_models = self.n_players
            self._bidder2model = list(range(self.n_players))

        # plot limits
        self.plot_xmin = self.u_lo
        self.plot_xmax = self.u_hi * 2
        self.plot_ymin = 0
        self.plot_ymax = self.u_hi * 1.1

        super().__init__(config)

    def _setup_sampler(self):

        default_batch_size = self.learning.batch_size
        device = self.hardware.device
        # setup individual samplers for each bidder
        self.sampler = MineralRightsValuationObservationSampler(
            n_players=self.n_players,
            valuation_size=self.valuation_size,
            common_value_lo=self.u_lo,
            common_value_hi=self.u_hi,
            default_batch_size=default_batch_size,
            default_device=device
        )

    def _setup_mechanism(self):
        if self.payment_rule == 'second_price':
            self.mechanism = VickreyAuction(random_tie_break=False, cuda=self.hardware.cuda)
        else:
            raise ValueError('Invalid Mechanism type!')

    def _check_and_set_known_bne(self):
        if self.payment_rule == 'second_price' and self.n_players == 3:
            self._optimal_bid = partial(bne_3p_mineral_rights)
            return True
        else:
            return super()._check_and_set_known_bne()

    def _setup_eval_environment(self):
        assert self.known_bne
        assert hasattr(self, '_optimal_bid')

        if self.n_players == 3:
            bne_strategy = ClosureStrategy(self._optimal_bid)

            # define bne agents once then use them in all runs
            agents = [
                self._strat_to_bidder(
                    strategy=bne_strategy,
                    player_position=i,
                    batch_size=self.config.logging.eval_batch_size,
                    enable_action_caching=self.config.logging.cache_eval_actions
                )
                for i in range(self.n_players)
            ]
            for a in agents:
                a._grid_lb = 0
                a._grid_ub = 2

            self.bne_env = AuctionEnvironment(
                mechanism=self.mechanism,
                agents=agents,
                valuation_observation_sampler=self.sampler,
                batch_size=self.config.logging.eval_batch_size,
                n_players=self.n_players,
                strategy_to_player_closure=self._strat_to_bidder
            )

            # Calculate bne_utility via sampling and from known closed form solution and do a sanity check
            self.bne_utilities = torch.zeros((self.n_players,), device=self.config.hardware.device)
            for i, a in enumerate(self.bne_env.agents):
                self.bne_utilities[i] = self.bne_env.get_reward(agent=a, redraw_valuations=True)

            print('Utility in BNE (sampled): \t{}'.format(self.bne_utilities))
            self.bne_utility = self.bne_utilities.mean()

        else:
            self.known_bne = False

    def _strat_to_bidder(self, **kwargs):
        return Bidder(risk=self.risk, **kwargs)

    def _get_logdir_hierarchy(self):
        name = ['single_item', self.payment_rule, 'interdependent', self.valuation_prior,
                'symmetric', str(self.risk) + 'risk', str(self.n_players) + 'players']
        return os.path.join(*name)



class AffiliatedObservationsExperiment(SingleItemExperiment):
    """A Single Item Experiment that has the same valuation prior for all participating bidders.
    For risk-neutral agents, a unique BNE is known.
    """

    def __init__(self,  config: ExperimentConfig):
        self.n_players = config.setting.n_players

        u_lo = config.setting.u_lo
        u_hi = config.setting.u_hi

        assert len(set(u_lo)) == 1, "Symmetric prior supported only!"
        assert len(set(u_hi)) == 1, "Symmetric prior supported only!"

        self.n_items = 1

        self.valuation_prior = 'uniform'
        self.u_lo = float(u_lo[0])
        self.u_hi = float(u_hi[0])
        self.common_prior = torch.distributions.uniform.Uniform(low=self.u_lo, high=self.u_hi)
        self.positive_output_point = torch.stack([self.common_prior.mean] * self.n_items)

        self.risk = float(config.setting.risk)
        self.risk_profile = self.get_risk_profile(self.risk)

        self.correlation_groups = config.setting.correlation_groups
        assert self.correlation_groups == [[0, 1]], \
            "other settings not implemented properly yet"
        assert len(config.setting.correlation_coefficients) == 1

        self.model_sharing = config.learning.model_sharing
        if self.model_sharing:
            self.n_models = 1
            self._bidder2model = [0] * self.n_players
        else:
            self.n_models = self.n_players
            self._bidder2model = list(range(self.n_players))

        # plot limits
        self.plot_xmin = self.u_lo
        self.plot_xmax = self.u_hi
        self.plot_ymin = 0
        self.plot_ymax = self.u_hi

        super().__init__(config)

    def _setup_sampler(self):

        default_batch_size = self.learning.batch_size
        device = self.hardware.device
        # setup individual samplers for each bidder
        self.sampler = AffiliatedValuationObservationSampler(
            n_players=self.n_players,
            valuation_size=self.valuation_size,
            u_lo=self.u_lo, u_hi=self.u_hi,
            default_batch_size=default_batch_size,
            default_device=device
        )

    def _setup_mechanism(self):
        if self.payment_rule == 'first_price':
            self.mechanism = FirstPriceSealedBidAuction(cuda=self.hardware.cuda)
        else:
            raise ValueError('Invalid Mechanism type!')

    def _check_and_set_known_bne(self):
        if self.payment_rule == 'first_price' and self.n_players == 2:
            self._optimal_bid = partial(bne_2p_affiliated_values)
            return True
        else:
            return super()._check_and_set_known_bne()

    def _setup_eval_environment(self):
        assert self.known_bne
        assert hasattr(self, '_optimal_bid')

        bne_strategy = ClosureStrategy(self._optimal_bid)

        # define bne agents once then use them in all runs
        agents = [
            self._strat_to_bidder(
                strategy=bne_strategy,
                player_position=i,
                batch_size=self.config.logging.eval_batch_size,
                enable_action_caching=self.config.logging.cache_eval_actions
            )
            for i in range(self.n_players)
        ]
        for a in agents:
            a._grid_lb = 0
            a._grid_ub = 1.5

        self.bne_env = AuctionEnvironment(
            mechanism=self.mechanism,
            agents=agents,
            valuation_observation_sampler=self.sampler,
            batch_size=self.config.logging.eval_batch_size,
            n_players=self.n_players,
            strategy_to_player_closure=self._strat_to_bidder
        )

        # Calculate bne_utility via sampling and from known closed form solution and do a sanity check
        self.bne_utilities = torch.zeros((3,), device=self.config.hardware.device)
        for i, a in enumerate(self.bne_env.agents):
            self.bne_utilities[i] = self.bne_env.get_reward(agent=a, redraw_valuations=True)

        print('Utility in BNE (sampled): \t{}'.format(self.bne_utilities.tolist()))
        self.bne_utility = self.bne_utilities.mean()

    def _strat_to_bidder(self, **kwargs):
        return Bidder(risk=self.risk, **kwargs)

    def _get_logdir_hierarchy(self):
        name = ['single_item', self.payment_rule, 'interdependent', self.valuation_prior,
                'symmetric', str(self.risk) + 'risk', str(self.n_players) + 'players']
        return os.path.join(*name)
