"""
This module implements combinatorial experiments. Currently, this is only Local Global experiments as
considered by Bosshard et al. (2018).

Limitations and comments:
    - Currently implemented for only uniform valuations
    - Strictly speaking Split Award might belong here (however, implmentation closer to multi-unit)

TODO:
    - Check if truthful bidding is BNE in LLLLGG with VCG
"""
import os
from abc import ABC
from functools import partial
from typing import Iterable, List
import math
import warnings
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import torch

from bnelearn.mechanism import (
    LLGAuction, LLGFullAuction, LLLLGGAuction
)
from bnelearn.bidder import Bidder, CombinatorialBidder
from bnelearn.environment import AuctionEnvironment
from bnelearn.experiment.configurations import ExperimentConfig
from .experiment import Experiment
from bnelearn.strategy import ClosureStrategy
from bnelearn.correlation_device import (
    IndependentValuationDevice,
    BernoulliWeightsCorrelationDevice,
    ConstantWeightsCorrelationDevice
)

import bnelearn.util.logging as logging_utils

class LocalGlobalExperiment(Experiment, ABC):
    """
    This class represents Local Global experiments in general as considered by Bosshard et al. (2018).
    It serves only to provide common logic and parameters for LLG and LLLLGG.
    """

    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.n_players = self.config.setting.n_players
        self.n_local = self.config.setting.n_local
        self.n_items = self.config.setting.n_items

        self.risk = float(config.setting.risk)

        assert self.config.setting.u_lo is not None, """Missing prior information!"""
        assert self.config.setting.u_hi is not None, """Missing prior information!"""
        u_lo = self.config.setting.u_lo
        # Frontend could either provide single number u_lo that is shared or a list for each player.
        if isinstance(u_lo, Iterable): # pylint: disable=isinstance-second-argument-not-valid-type
            assert len(u_lo) == self.n_players
            u_lo = [float(l) for l in u_lo]
        else:
            u_lo = [float(u_lo)] * self.n_players
        self.u_lo = u_lo

        u_hi = self.config.setting.u_hi
        assert isinstance(u_hi, Iterable) # pylint: disable=isinstance-second-argument-not-valid-type
        assert len(u_hi) == self.n_players
        assert u_hi[1:self.config.setting.n_local] == \
               u_hi[:self.config.setting.n_local - 1], "local bidders should be identical"
        assert u_hi[0] < \
               u_hi[self.config.setting.n_local], "local bidders must be weaker than global bidder"
        self.u_hi = [float(h) for h in u_hi]

        self.positive_output_point = torch.tensor([min(self.u_hi)] * self.input_length)

        self.model_sharing = self.config.learning.model_sharing
        if self.model_sharing:
            self.n_models = 2
            self._bidder2model: List[int] = [0] * self.config.setting.n_local \
                                            + [1] * (self.n_players - self.config.setting.n_local)
        else:
            self.n_models = self.n_players
            self._bidder2model: List[int] = list(range(self.n_players))

        super().__init__(config=config)

        self.plot_xmin = min(u_lo)
        self.plot_xmax = max(u_hi)
        self.plot_ymin = self.plot_xmin
        self.plot_ymax = self.plot_xmax * 1.05

    def _get_model_names(self):
        if self.model_sharing:
            global_name = 'global' if self.n_players - self.n_local == 1 else 'globals'
            return ['locals', global_name]
        else:
            return super()._get_model_names()

    def _strat_to_bidder(self, strategy, batch_size, player_position=0, cache_actions=False):
        correlation_type = 'additive' if hasattr(self, 'correlation_groups') else None
        return Bidder.uniform(self.u_lo[player_position], self.u_hi[player_position], strategy,
                              player_position=player_position, batch_size=batch_size,
                              n_items=self.input_length, correlation_type=correlation_type,
                              risk=self.risk, cache_actions=cache_actions)


class LLGExperiment(LocalGlobalExperiment):
    """
    A combinatorial experiment with 2 local and 1 global bidder and 2 items; but each bidders bids on 1 bundle only.
    Local bidder 1 bids only on the first item, the second only on the second and global only on both.
    Ausubel and Baranov (2018) provide closed form solutions for the 3 core selecting rules.

    Supports arbitrary number of local bidders, not just two.
    """

    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.n_local = self.config.setting.n_players - 1

        self.gamma = self.correlation = float(config.setting.gamma)

        if hasattr(config.setting, 'regret'):
            self.regret = float(config.setting.regret)
        else:
            self.regret = 0  # default quasi-linear utility

        if config.setting.correlation_types == 'Bernoulli_weights':
            self.CorrelationDevice = BernoulliWeightsCorrelationDevice
        elif config.setting.correlation_types == 'constant_weights':
            self.CorrelationDevice = ConstantWeightsCorrelationDevice
        elif config.setting.correlation_types == 'independent':
            pass
        else:
            raise NotImplementedError('Correlation not implemented.')

        if self.gamma > 0.0:
            self.correlation_groups = [list(range(self.n_local)), [self.n_local]]
            self.correlation_coefficients = [self.gamma] * (self.n_local + 1)
            self.correlation_coefficients[-1] = 0  # global bidder is independent
            self.correlation_devices = [
                self.CorrelationDevice(
                    common_component_dist=torch.distributions.Uniform(config.setting.u_lo[0],
                                                                      config.setting.u_hi[0]),
                    batch_size=config.learning.batch_size,
                    n_items=1,
                    correlation=self.gamma),
                IndependentValuationDevice()]

        self.input_length = 1
        self.config.setting.n_players = self.config.setting.n_players
        self.config.setting.n_local = self.n_local
        self.config.setting.n_items = 1
        super().__init__(config=config)

    def _setup_mechanism(self):
        self.mechanism = LLGAuction(rule=self.payment_rule)

    def _optimal_bid(self, valuation, player_position): # pylint: disable=method-hidden
        """Core selecting and vcg equilibria for the Bernoulli weigths model in Ausubel & Baranov (2019)

           Note: for gamma=0 or gamma=1, these are identical to the constant weights model.
        """
        if not isinstance(valuation, torch.Tensor):
            valuation = torch.tensor(valuation)

        ### Global bidder: all core-selecting rules are strategy proof for global player
        if self.payment_rule in ['vcg', 'proxy', 'nearest_zero', 'nearest_bid',
                                 'nearest_vcg'] and player_position == self.n_players - 1:
            return valuation

        ### Local bidders: vcg => truthfull bidding
        if self.payment_rule in ['vcg'] and player_position in list(range(self.n_players - 1)):
            return valuation

        assert self.risk == 1.0, 'BNE known for risk-neutral only (or in VCG)'

        ### Local bidders:
        if self.config.setting.correlation_types in ['Bernoulli_weights', 'independent'] or \
            (self.config.setting.correlation_types == 'constant_weights' and self.gamma in [0, 1]):
            ## perfect correlation
            if self.gamma == 1.0: # limit case, others not well defined
                sigma = 1.0 # TODO: implement for other valuation profiles!
                bid = valuation
                if self.payment_rule == 'nearest_vcg':
                    bid.mul_(sigma / (1 + sigma - 2**(-sigma)))
                elif self.payment_rule == 'nearest_bid':
                    bid.mul_(sigma / (1 + sigma))
                # truthful for vcg and proxy/nearest-zero
                return bid
            ## no or imperfect correlation
            if self.payment_rule in ['proxy', 'nearest_zero']:
                bid_if_positive = 1 + torch.log(valuation * (1.0 - self.gamma) + self.gamma) / (1.0 - self.gamma)
                return torch.max(torch.zeros_like(valuation), bid_if_positive)
            if self.payment_rule == 'nearest_bid':
                return (np.log(2) - torch.log(2.0 - (1. - self.gamma) * valuation)) / (1. - self.gamma)
            if self.payment_rule == 'nearest_vcg':
                bid_if_positive = 2. / (2. + self.gamma) * (
                    valuation - (3. - np.sqrt(9 - (1. - self.gamma) ** 2)) / (1. - self.gamma))
                return torch.max(torch.zeros_like(valuation), bid_if_positive)
            warnings.warn('optimal bid not implemented for this payment rule')
        else:
            warnings.warn('optimal bid not implemented for this correlation type')

        self.known_bne = False

    def _check_and_set_known_bne(self):
        # TODO: This is not exhaustive, other criteria must be fulfilled for the bne to be known!
        #  (i.e. uniformity, bounds, etc)
        known_bne = None
        if self.config.setting.payment_rule == 'vcg':
            return True
        elif self.config.setting.n_players != 3:
            known_bne = False
        elif self.risk != 1.0:
            known_bne = False
        elif self.regret != 0.0:
            known_bne = False
        elif self.config.setting.payment_rule in \
            ['nearest_bid', 'nearest_zero', 'proxy', 'nearest_vcg']:
            if self.config.setting.correlation_types in ['Bernoulli_weights', 'independent'] or \
                (self.config.setting.correlation_types == 'constant_weights' and self.gamma in [0, 1]):
                return True
            else:
                known_bne = False

        if known_bne is None:
            known_bne = super()._check_and_set_known_bne()
        if not known_bne:
            self.logging.log_metrics['l2'] = False
            self.logging.log_metrics['opt'] = False

        return known_bne

    def _setup_eval_environment(self):

        assert self.known_bne
        assert hasattr(self, '_optimal_bid')

        bne_strategies = [
            ClosureStrategy(partial(self._optimal_bid, player_position=i))  # pylint: disable=no-member
            for i in range(self.n_players)]

        bne_env_corr_devices = None
        if self.correlation_groups:
            bne_env_corr_devices = [
                self.CorrelationDevice(
                    common_component_dist=torch.distributions.Uniform(self.config.setting.u_lo[0],
                                                                      self.config.setting.u_hi[0]),
                    batch_size=self.config.logging.eval_batch_size,
                    n_items=1,
                    correlation=self.gamma),
                IndependentValuationDevice()]

        self.known_bne = True
        bne_env = AuctionEnvironment(
            mechanism=self.mechanism,
            agents=[self._strat_to_bidder(bne_strategies[i], player_position=i,
                                          batch_size=self.config.logging.eval_batch_size)
                    for i in range(self.n_players)],
            n_players=self.n_players,
            batch_size=self.config.logging.eval_batch_size,
            strategy_to_player_closure=self._strat_to_bidder,
            correlation_groups=self.correlation_groups,
            correlation_devices=bne_env_corr_devices
        )

        self.bne_env = bne_env
        self.bne_utilities_new_sample = torch.tensor(
            [bne_env.get_reward(a, draw_valuations=True) for a in bne_env.agents])

        bne_utilities_database = logging_utils.access_bne_utility_database(self, self.bne_utilities_new_sample)
        if bne_utilities_database:
            self.bne_utilities = bne_utilities_database
        else:
            self.bne_utilities = self.bne_utilities_new_sample

        print(f'Setting up BNE env with batch size 2**{np.log2(self.config.logging.eval_batch_size)}.')
        print(('Utilities in BNE (sampled):' + '\t{:.5f}' * self.n_players + '.').format(*self.bne_utilities_new_sample))
        print("No closed form solution for BNE utilities available in this setting. Using sampled value as baseline.")

    def _get_logdir_hierarchy(self):
        name = [self.n_local * 'L' + 'G', self.payment_rule]
        if self.gamma > 0:
            name += [self.config.setting.correlation_types, f"gamma_{self.gamma:.3}"]
        else:
            name += ['independent']
        if self.risk != 1.0:
            name += ['risk_{}'.format(self.risk)]
        if self.regret != 0.0:
            name += ['regret_{}'.format(self.regret)]
        return os.path.join(*name)

    def _strat_to_bidder(self, strategy, batch_size, player_position=0, cache_actions=False):
        correlation_type = 'additive' if hasattr(self, 'correlation_groups') else None
        return Bidder.uniform(self.u_lo[player_position], self.u_hi[player_position], strategy,
                              player_position=player_position, batch_size=batch_size,
                              n_items=self.input_length, correlation_type=correlation_type,
                              risk=self.risk, cache_actions=cache_actions, regret=self.regret)


class LLGFullExperiment(LocalGlobalExperiment):
    """A combinatorial experiment with 2 local and 1 global bidder and 2 items.

    Essentially, this is a general CA with 3 bidders and 2 items.

    Each bidders bids on all bundles. Local bidder 1 has only a value for the
    first item, the second only for the second and global only on both. This
    experiment is therfore more general than the `LLGExperiment` and includes
    the specifc payment rule from Beck & Ott, where the 2nd local bidder is
    favored (pays VCG prices).

    """
    def __init__(self, config: ExperimentConfig):
        self.config = config
        assert self.config.setting.n_players == 3, \
            "Incorrect number of players specified."

        self.gamma = self.correlation = float(config.setting.gamma)
        if config.setting.correlation_types != 'independent' or \
            self.gamma > 0.0:
            # Should be similar to reduced LLG setting, but we have to consider
            # asymmetry of local bidders.
            raise NotImplementedError('Correlation not implemented.')

        self.input_length = 1
        self.config.setting.n_local = 2
        self.config.setting.n_items = 3
        super().__init__(config=config)

    def _setup_mechanism(self):
        self.mechanism = LLGFullAuction(rule=self.payment_rule,
                                        cuda=self.hardware.device)

    def _check_and_set_known_bne(self):
        return self.payment_rule in ['vcg', 'mrcs_favored']

    def _optimal_bid(self, valuation, player_position):  # pylint: disable=method-hidden
        """Equilibrium bid functions.

        Payment rule `mrcs_favored` is from Beck & Ott (minimum revenue core
        selecting with one player favored).
        """
        if not isinstance(valuation, torch.Tensor):
            valuation = torch.as_tensor(valuation, device=self.config.hardware.device)

        assert self.risk == 1.0, 'BNE known for risk-neutral only (or in VCG)'

        if self.payment_rule in ['vcg', 'mrcs_favored']:
            if player_position == 1:
                return torch.cat([
                    torch.zeros_like(valuation),  # item A
                    valuation,  # item B
                    valuation], axis=1)  # bundle {A, B}
            if player_position == 2:
                return torch.cat([
                    torch.zeros_like(valuation),  # TODO ?
                    torch.zeros_like(valuation),
                    valuation], axis=1)

        ### Favored bidder 1:
        if self.config.setting.correlation_types in ['independent'] and player_position == 0:
            if self.payment_rule == 'vcg':
                return torch.cat([
                    valuation,
                    torch.zeros_like(valuation),
                    valuation], axis=1)
            if self.payment_rule == 'mrcs_favored':
                # Beck & Ott provide no solution: Here we take real part of
                # complex solution (sqrt of negative values), see
                # https://www.wolframalpha.com/input/?i=0+%3D+12*v-+15*z+-+1+%2B+%289*z+-+1+-+3*v%29*sqrt%281+-+6*z%2B+6*v%29+solve+for+z
                v = torch.as_tensor(valuation, device=valuation.device,
                                    dtype=torch.cfloat)
                sqrt = torch.sqrt(
                    - 254016 * torch.pow(v, 5) - 45045 * torch.pow(v, 4) \
                    + 47892 * torch.pow(v, 3) + 118676 * torch.pow(v, 2) \
                    - 74560 * v + 11520
                )
                outer = torch.pow(
                    + 23328 * torch.pow(v, 3) - 47871 * torch.pow(v, 2) \
                    + 81 * np.sqrt(3) * sqrt + 6534 * v + 2884,
                    1./3.
                )
                z = torch.real(outer / (81* 2**(2./3.)) \
                    - (-1296* torch.pow(v, 2) - 2196 * v + 956) \
                    / (162 * 2**(1./3.) * outer) + (1./81.) * (45 * v - 2))

                b_A = torch.zeros_like(valuation)
                mask = 2 - 2 * math.sqrt(6.) / 3. < valuation
                b_A[mask] = z[mask] \
                    - (2 - torch.sqrt(1 - 6 * z[mask] + 6 * valuation[mask])) \
                    / 3.
                b_AB = 0.5 * valuation.detach().clone()
                b_AB[mask] = z[mask]
                bids = torch.cat([b_A, torch.zeros_like(valuation), b_AB], axis=1)
                return bids

            warnings.warn('optimal bid not implemented for this payment rule')
        else:
            warnings.warn('optimal bid not implemented for this correlation type')

        self.known_bne = False

    def relevant_actions(self):
        if self.config.setting.correlation_types in ['independent'] and \
            self.payment_rule in ['vcg', 'mrcs_favored']:
            return torch.tensor(
                [[1, 0, 0], [0, 1, 0], [0, 0, 1]], # TODO rather: [[1, 0, 1], [0, 1, 1], [0, 0, 1]]
                device=self.config.hardware.device,
                dtype=torch.bool
            )
        return super().relevant_actions()

    def _evaluate_and_log_epoch(self, epoch: int) -> float:
        # TODO keep track of time as in super()
        for name, agent in zip(['local 1', 'local 2', 'global'], self.env.agents):
            self.writer.add_histogram(
                tag="allocations/" + name,
                values=self.env.get_allocation(agent),
                bins=2*self.n_items-1,
                global_step=epoch
            )
        return super()._evaluate_and_log_epoch(epoch)

    def _setup_eval_environment(self):
        assert self.known_bne
        assert hasattr(self, '_optimal_bid')

        bne_strategies = [
            ClosureStrategy(partial(self._optimal_bid, player_position=i))  # pylint: disable=no-member
            for i in range(self.n_players)]

        self.known_bne = True
        self.bne_env = AuctionEnvironment(
            mechanism=self.mechanism,
            agents=[self._strat_to_bidder(bne_strategies[i], player_position=i,
                                          batch_size=self.config.logging.eval_batch_size)
                    for i in range(self.n_players)],
            n_players=self.n_players,
            batch_size=self.config.logging.eval_batch_size,
            strategy_to_player_closure=self._strat_to_bidder
        )
        self.bne_utilities = torch.tensor(
            [self.bne_env.get_reward(a, draw_valuations=True) for a in self.bne_env.agents])

        # Compare estimated util to that of Beck & Ott table 2
        if self.payment_rule == 'mrcs_favored':
            max_diff_to_estimate = float(max(
                torch.abs(self.bne_utilities - torch.tensor([0.154, 0.093, 0.418]))
            ))
            print(f'Max difference to BNE estimate is {round(max_diff_to_estimate, 4)}.')

    def _get_logdir_hierarchy(self):
        name = ['LLGFull', self.payment_rule]
        if self.gamma > 0:
            name += [self.config.setting.correlation_types,
                     f"gamma_{self.gamma:.3}"]
        else:
            name += ['independent']
        if self.risk != 1.0:
            name += ['risk_{}'.format(self.risk)]
        return os.path.join(*name)

    def _strat_to_bidder(self, strategy, batch_size, player_position=0,
                         cache_actions=False):
        correlation_type = 'additive' if (hasattr(self, 'correlation_groups') \
            and self.config.setting.correlation_types != 'independent') else None
        return CombinatorialBidder.uniform(
            self.u_lo[player_position],
            self.u_hi[player_position],
            strategy=strategy,
            player_position=player_position,
            batch_size=batch_size,
            n_items=self.input_length,
            correlation_type=correlation_type,
            risk=self.risk,
            cache_actions=cache_actions
        )

    def _plot(self, **kwargs):  # pylint: disable=arguments-differ
        kwargs['x_label'] = ['item A', 'item B', 'bundle']
        kwargs['labels'] = ['local 1', 'local 2', 'global']

        # handle dim-missmatch that agents only value 1 bundle but bid for 3
        plot_data = list(kwargs['plot_data'])
        if plot_data[0].shape != plot_data[1].shape:
            plot_data[0] = plot_data[0].repeat(1, 1, self.n_items)
            kwargs['plot_data'] = plot_data

        super()._plot(**kwargs)


class LLLLGGExperiment(LocalGlobalExperiment):
    """
    A combinatorial experiment with 4 local and 2 global bidder and 6 items; but each bidders bids on 2 bundles only.
        Local bidder 1 bids on the bundles {(item_1,item_2),(item_2,item_3)}
        Local bidder 2 bids on the bundles {(item_3,item_4),(item_4,item_5)}
        ...
        Gloabl bidder 1 bids on the bundles {(item_1,item_2,item_3,item_4), (item_5,item_6,item_7,item_8)}
        Gloabl bidder 1 bids on the bundles {(item_3,item_4,item_5,item_6), (item_1,item_2,item_7,item_8)}
    No BNE are known (but VCG).
    Bosshard et al. (2018) consider this setting with nearest-vcg and first-price payments.

    TODO:
        - Implement eval_env for VCG
    """

    def __init__(self, config: ExperimentConfig):
        self.config = config
        assert self.config.setting.n_players == 6, "not right number of players for setting"
        self.input_length = 2

        self.config.running.n_players = 6
        self.config.setting.n_local = 4
        self.config.setting.n_items = 2
        super().__init__(config=config)

    def _setup_mechanism(self):
        self.mechanism = LLLLGGAuction(rule=self.payment_rule, core_solver=self.setting.core_solver,
                                       parallel=self.hardware.max_cpu_threads, cuda=self.hardware.cuda)

    def _get_logdir_hierarchy(self):
        name = ['LLLLGG', self.payment_rule, str(self.n_players) + 'p']
        return os.path.join(*name)

    def _plot(self, plot_data, writer: SummaryWriter or None, epoch=None,
              fmts=['o'], **kwargs):
        super()._plot(plot_data=plot_data, writer=writer, epoch=epoch,
                      fmts=fmts, **kwargs)
        super()._plot_3d(plot_data=plot_data, writer=writer, epoch=epoch,
                         figure_name=kwargs['figure_name'])
