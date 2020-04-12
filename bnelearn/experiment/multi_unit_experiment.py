import os
import time
import warnings
from abc import ABC
from itertools import product

import torch
import numpy as np
from scipy import integrate, interpolate

from bnelearn.bidder import Bidder, ReverseBidder
from bnelearn.environment import AuctionEnvironment
from bnelearn.experiment import LearningConfiguration, GPUController, Logger, Experiment
from bnelearn.learner import ESPGLearner
from bnelearn.mechanism import (
    MultiItemVickreyAuction, MultiItemUniformPriceAuction, MultiItemDiscriminatoryAuction,
    FPSBSplitAwardAuction, Mechanism
)
from bnelearn.strategy import NeuralNetStrategy, ClosureStrategy
from bnelearn.util import metrics


class MultiUnitExperiment(Experiment, ABC):
    def __init__(self, experiment_params: dict, mechanism: Mechanism, gpu_config: GPUController, logger: Logger,
                 l_config: LearningConfiguration):

        # ToDO How about to assigning expetiment parameters to the object and just using them from the dictionary?
        self.n_players = experiment_params['n_players']
        self.n_items = experiment_params['n_items']
        self.u_lo = experiment_params['u_lo']
        self.u_hi = experiment_params['u_hi']
        self.BNE1 = experiment_params['BNE1']
        self.model_sharing = experiment_params['model_sharing']

        if 'BNE2' in experiment_params.keys():
            self.BNE2 = experiment_params['BNE2']
        if 'constant_marginal_values' in experiment_params.keys():
            self.constant_marginal_values = experiment_params['constant_marginal_values']
        else:
            self.constant_marginal_values = None
        if 'item_interest_limit' in experiment_params.keys():
            self.item_interest_limit = experiment_params['item_interest_limit']
        else:
            self.item_interest_limit = None
        if 'efficiency_parameter' in experiment_params.keys():
            self.efficiency_parameter = experiment_params['efficiency_parameter']
        if 'pretrain_transform' in experiment_params.keys():
            self.pretrain_transform = experiment_params['pretrain_transform']
        else:
            self.pretrain_transform = self.default_pretrain_transform
        if 'input_length' in experiment_params.keys():
            self.input_length = experiment_params['input_length']
        else:
            experiment_params['input_length'] = self.n_items
            self.input_length = self.n_items

        self.n_parameters = list()

        print('\nhyperparams\n-----------')
        for k in l_config.learner_hyperparams.keys():
            print('{}: {}'.format(k, l_config.learner_hyperparams[k]))
        print('-----------\n')

        super().__init__(gpu_config, experiment_params, logger, l_config)

        self.mechanism = mechanism

    def _strat_to_bidder(self, strategy, batch_size, player_position=0, cache_actions=False):
        """
        Standard strat_to_bidder method.
        """
        return Bidder.uniform(
            lower=self.u_lo[player_position], upper=self.u_hi[player_position],
            strategy=strategy,
            n_items=self.n_items,
            item_interest_limit=self.item_interest_limit,
            descending_valuations=True,
            constant_marginal_values=self.constant_marginal_values,
            player_position=player_position,
            batch_size=batch_size
        )

    def _setup_bidders(self):
        epo_n = 2  # for ensure positive output of initialization

        n_models = 1 if self.model_sharing else self.n_players
        for i in range(n_models):
            ensure_positive_output = torch.zeros(epo_n, self.input_length).uniform_(self.u_lo[i], self.u_hi[i]) \
                .sort(dim=1, descending=True)[0]
            self.models = [
                NeuralNetStrategy(
                    self.input_length,
                    hidden_nodes=self.l_config.hidden_nodes,
                    hidden_activations=self.l_config.hidden_activations,
                    ensure_positive_output=ensure_positive_output,
                    output_length=self.n_items
                ).to(self.gpu_config.device)
            ]

        # Pretrain
        pretrain_points = round(100 ** (1 / self.input_length))
        # pretrain_valuations = multi_unit_valuations(
        #     device = device,
        #     bounds = [param_dict["u_lo"], param_dict["u_hi"][0]],
        #     dim = param_dict["n_items"],
        #     batch_size = pretrain_points,
        #     selection = 'random' if param_dict["exp_no"] != 6 else split_award_dict
        # )
        pretrain_valuations = self._strat_to_bidder(
            lambda x: x, self.l_config.batch_size, 0).draw_valuations_()[:pretrain_points, :]

        self.n_parameters = list()
        for model in self.models:
            self.n_parameters.append(sum([p.numel() for p in model.parameters()]))
            model.pretrain(pretrain_valuations, self.l_config.pretrain_iters,
                           self.pretrain_transform)
        self.experiment_params['n_parameters'] = self.n_parameters

        # I see no other way to get this info to the logger
        self.experiment_params['n_parameters'] = self.n_parameters
        self.bidders = [
            self._strat_to_bidder(self.models[0 if self.model_sharing else i], self.l_config.batch_size, i)
            for i in range(self.n_players)
        ]

    def _setup_learning_environment(self):
        self.env = AuctionEnvironment(
            self.mechanism,
            agents=self.bidders,
            n_players=self.n_players,
            batch_size=self.l_config.batch_size,
            strategy_to_player_closure=self._strat_to_bidder
        )

    def _setup_learners(self):
        self.learners = [
            ESPGLearner(
                model=self.models[i],
                environment=self.env,
                hyperparams=self.l_config.learner_hyperparams,
                optimizer_type=self.l_config.optimizer,
                optimizer_hyperparams=self.l_config.optimizer_hyperparams,
                strat_to_player_kwargs={"player_position": i}
            )
            for i, model in enumerate(self.models)
        ]

    def _setup_eval_environment(self):
        self.bne_strategies = [
            # n_players is now not used inside optimal_bid function
            # ClosureStrategy(
            #    partial(
            #        _optimal_bid(self.mechanism, param_dict),
            #        player_position=i
            #    )
            # )
            # for i in range(self.n_players)
            ClosureStrategy(self._optimal_bid) for i in range(self.n_players)
        ]

        self.bne_env = AuctionEnvironment(
            self.mechanism,
            agents=[
                self._strat_to_bidder(bne_strategy, self.l_config.batch_size, i)
                for i, bne_strategy in enumerate(self.bne_strategies)
            ],
            n_players=self.n_players,
            batch_size=self.l_config.batch_size,
            strategy_to_player_closure=self._strat_to_bidder
        )

    def _get_logdir(self):
        auction_type_str = str(type(self.mechanism))
        auction_type_str = str(auction_type_str[len(auction_type_str) - auction_type_str[::-1].find('.'):-2])

        name = ['expiriments_nils', auction_type_str, str(self.n_players) + 'players_' + str(self.n_items) + 'items']
        return os.path.join(*name)

    def _training_loop(self, epoch, logger):
        start_time = time.time()

        # calculate utility vs BNE
        bne_utilities = list()
        for agent in self.bne_env.agents:
            u = self.bne_env.get_reward(agent, draw_valuations=True)
            bne_utilities.append(u)

        torch.cuda.empty_cache()
        self.env.prepare_iteration()

        # record utilities and do optimizer step
        utilities = list()
        for i, learner in enumerate(self.learners):
            u = learner.update_strategy_and_evaluate_utility()
            utilities.append(u)

        # log relative utility loss induced by not playing the BNE
        against_bne_utilities = list()
        for i, model in enumerate(self.models):
            u = self.bne_env.get_strategy_reward(model, player_position=i, draw_valuations=True)
            against_bne_utilities.append(u)

        # calculate regret
        grid = torch.linspace(
            0, self.u_hi[0],
            round((self.experiment_params['regret_batch_size']) ** (1 / self.n_items)),
            device = self.gpu_config.device
        )
        bid_profile = torch.zeros(self.l_config.batch_size, self.n_players, self.n_items,
                                  device=self.gpu_config.device)
        for pos, bid in self.env._generate_agent_actions():
            bid_profile[:, pos, :] = bid
        regret = [None] * len(self.models)
        for i in range(len(self.models)):
            regret[i] = metrics.ex_post_regret(
                self.mechanism,
                bid_profile.detach(),
                self.bidders[i],
                grid, half_precision=True,
                player_position = i
            ).mean()
        # print('regret:', regret)

        elapsed = time.time() - start_time

        log_params = {
            'elapsed': elapsed,
            'optima_bid': self._optimal_bid,
            'optima_bid_2': self._optimal_bid_2,
            'bne_utilities': bne_utilities,
            'utilities': utilities,
            'against_bne_utilities': against_bne_utilities,
            'regret': regret
        }
        logger.log_training_iteration(log_params=log_params, epoch=epoch, bidders=self.bidders)

    def _optimal_bid(self, valuation: torch.Tensor or np.ndarray or float, player_position: int=0):
        if not isinstance(valuation, torch.Tensor):
            valuation = torch.tensor(valuation, dtype=torch.float, device=self.gpu_config.device)
        else:
            valuation = valuation.clone().detach()

        # # unsqueeze if simple float
        # if valuation.dim() == 0:
        #     valuation.unsqueeze_(0)

        # elif valuation.shape[1] == 1:
        #     valuation = torch.cat((valuation, self.efficiency_parameter * valuation), 1)

        return valuation

    def _optimal_bid_2(self, valuation: torch.Tensor or np.ndarray or float, player_position: int=0):
        if not isinstance(valuation, torch.Tensor):
            valuation = torch.tensor(valuation, dtype=torch.float, device=self.gpu_config.device)
        else:
            valuation = valuation.clone().detach()

        return valuation

    @staticmethod
    def default_pretrain_transform(input_tensor):
        return torch.clone(input_tensor)


# exp_no==0
class MultiItemVickreyAuction2x2(MultiUnitExperiment):
    class_experiment_params = {
        'n_items': 2,
        'n_players': 2,
        'BNE1': 'Truthful'
    }

    def __init__(self, experiment_params: dict, gpu_config: GPUController, logger: Logger,
                 l_config: LearningConfiguration):
        mechanism = MultiItemVickreyAuction(cuda=gpu_config.cuda)
        super().__init__({**self.class_experiment_params, **experiment_params}, mechanism=mechanism,
                         gpu_config=gpu_config, logger=logger, l_config=l_config)
        self._run_setup()

    def _optimal_bid_2(self, valuation: torch.Tensor or np.ndarray or float, player_position: int = 0):
        warnings.warn("No 2nd explicit BNE known.", Warning)
        return super()._optimal_bid_2(valuation, player_position)


# exp_no==1, BNE continua
class MultiItemUniformPriceAuction2x2(MultiUnitExperiment):
    class_experiment_params = {
        'n_items': 2,
        'n_players': 2,
        'BNE1': 'BNE1',
        'BNE2': 'BNE2'
    }

    def __init__(self, experiment_params: dict, gpu_config: GPUController, logger: Logger,
                 l_config: LearningConfiguration):
        mechanism = MultiItemUniformPriceAuction(cuda=gpu_config.cuda)
        super().__init__({**self.class_experiment_params, **experiment_params}, mechanism=mechanism,
                         gpu_config=gpu_config, logger=logger, l_config=l_config)
        self._run_setup()

    @staticmethod
    def default_pretrain_transform(input_tensor):
        output_tensor = torch.clone(input_tensor)
        output_tensor[:, 1] = 0
        return output_tensor

    def _optimal_bid(self, valuation, player_position=None):
        valuation = super()._optimal_bid(valuation)

        if self.n_players == 2 and self.n_items == 3 \
                and self.u_lo == 0 and self.u_hi == 1 \
                and self.item_interest_limit == 2:
            opt_bid = torch.clone(valuation)
            opt_bid[:, 1] = opt_bid[:, 1] ** 2
            opt_bid[:, 2] = 0
            return opt_bid
        elif self.n_players == 2 and self.n_items == 2:
            opt_bid = torch.clone(valuation)
            opt_bid[:, 1] = 0
            return opt_bid
        elif self.n_players == self.n_items:
            opt_bid = torch.zeros_like(valuation)
            opt_bid[:, 0] = self.u_hi
            return opt_bid
        elif self.n_players > self.n_items:  # & cdf of v_1 is strictly increasing
            opt_bid = torch.clone(valuation)
            opt_bid[:, 1:] = 0
            warnings.warn("Only BNE bidding for 0th item known.", Warning)
            return opt_bid
        else:
            warnings.warn("No explict BNE for MultiItemUniformPriceAuction known.", Warning)
            return valuation

    def _optimal_bid_2(self, valuation: torch.Tensor or np.ndarray or float, player_position: int = 0):
        super()._optimal_bid_2(valuation, player_position)

        if self.experiment_params["n_players"] == 2 and self.experiment_params["n_items"] == 2:
            opt_bid = torch.clone(valuation)
            opt_bid[:, 0] = self.experiment_params["u_hi"][0]
            opt_bid[:, 1] = 0
            return opt_bid
        else:
            warnings.warn("No 2nd explicit BNE known.", Warning)
            return valuation


# exp_no==2
class MultiItemUniformPriceAuction2x3limit2(MultiUnitExperiment):
    class_experiment_params = {
        'n_items': 3,
        'n_players': 2,
        'BNE1': 'BNE1',
        'BNE2': 'Truthful',
        'item_interest_limit': 2
    }

    def __init__(self, experiment_params: dict, gpu_config: GPUController, logger: Logger,
                 l_config: LearningConfiguration):
        mechanism = MultiItemUniformPriceAuction(cuda=gpu_config.cuda)

        super().__init__({**self.class_experiment_params, **experiment_params}, mechanism=mechanism,
                         gpu_config=gpu_config, logger=logger, l_config=l_config)
        self._run_setup()

    # ToDO Once again in this method using u_hi/u_lo[0]. Is it appropriate?
    def _optimal_bid(self, valuation, player_position=None):
        valuation = super()._optimal_bid(valuation)

        if self.n_players == 2 and self.n_items == 3 \
                and self.u_lo[0] == 0 and self.u_hi[0] == 1 \
                and self.item_interest_limit == 2:
            opt_bid = torch.clone(valuation)
            opt_bid[:, 1] = opt_bid[:, 1] ** 2
            opt_bid[:, 2] = 0
            return opt_bid
        elif self.n_players == 2 and self.n_items == 2:
            opt_bid = torch.clone(valuation)
            opt_bid[:, 1] = 0
            return opt_bid
        elif self.n_players == self.n_items:
            opt_bid = torch.zeros_like(valuation)
            opt_bid[:, 0] = self.u_hi[0]
            return opt_bid
        elif self.n_players > self.n_items:  # & cdf of v_1 is strictly increasing
            opt_bid = torch.clone(valuation)
            opt_bid[:, 1:] = 0
            warnings.warn("Only BNE bidding for 0th item known.", Warning)
            return opt_bid
        else:
            warnings.warn("No explict BNE for MultiItemUniformPriceAuction known.", Warning)
            return valuation

    def _optimal_bid_2(self, valuation: torch.Tensor or np.ndarray or float, player_position: int = 0):
        warnings.warn("No 2nd explicit BNE known.", Warning)
        return super()._optimal_bid_2(valuation, player_position)


# exp_no==4
class MultiItemDiscriminatoryAuction2x2(MultiUnitExperiment):
    class_experiment_params = {
        'n_items': 2,
        'n_players': 2,
        'BNE1': 'BNE1',
        'BNE2': 'Truthful'
    }

    def __init__(self, experiment_params: dict, gpu_config: GPUController, logger: Logger,
                 l_config: LearningConfiguration):
        mechanism = MultiItemDiscriminatoryAuction(cuda=gpu_config.cuda)
        super().__init__({**self.class_experiment_params, **experiment_params}, mechanism=mechanism,
                         gpu_config=gpu_config, logger=logger, l_config=l_config)
        self._run_setup()

    def _optimal_bid(self, valuation, player_position=None):
        valuation = super()._optimal_bid(valuation)

        def b_approx(v, s, t):
            b = torch.clone(v)
            lin_e = np.array([[1, 1, 1], [2 * t, 1, 0], [t ** 2, t, 1]])
            lin_s = np.array([0.47, s / t, s])
            x = np.linalg.solve(lin_e, lin_s)
            b[v < t] *= s / t
            b[v >= t] = x[0] * b[v >= t] ** 2 + x[1] * b[v >= t] + x[2]
            return b

        b1 = lambda v: b_approx(v, s=0.42, t=0.90)
        b2 = lambda v: b_approx(v, s=0.35, t=0.55)

        opt_bid = valuation
        opt_bid[:, 0] = b1(opt_bid[:, 0])
        opt_bid[:, 1] = b2(opt_bid[:, 1])
        opt_bid = opt_bid.sort(dim=1, descending=True)[0]
        return opt_bid

    def _optimal_bid_2(self, valuation: torch.Tensor or np.ndarray or float, player_position: int = 0):
        warnings.warn("No 2nd explicit BNE known.", Warning)
        return super()._optimal_bid_2(valuation, player_position)


# exp_no==5
class MultiItemDiscriminatoryAuction2x2CMV(MultiUnitExperiment):
    class_experiment_params = {
        'n_items': 2,
        'n_players': 2,
        'BNE1': 'BNE1',
        'BNE2': 'Truthful',
        'constant_marginal_values': True
    }

    def __init__(self, experiment_params: dict, gpu_config: GPUController, logger: Logger,
                 l_config: LearningConfiguration):
        mechanism = MultiItemDiscriminatoryAuction(cuda=gpu_config.cuda)
        super().__init__({**self.class_experiment_params, **experiment_params}, mechanism=mechanism,
                         gpu_config=gpu_config, logger=logger, l_config=l_config)
        self._run_setup()

    def _optimal_bid(self, valuation, player_position=None):
        valuation = super()._optimal_bid(valuation)

        # ToDo Ugly if, right now uniform is hardcoded everywhere
        if isinstance(self._strat_to_bidder(None, 1).value_distribution,
                      torch.distributions.uniform.Uniform):
            return valuation / 2

        elif isinstance(self._strat_to_bidder(None, 1).value_distribution,
                        torch.distributions.normal.Normal):

            # TODO: just calc once and then interpolate with that?
            def muda_tb_cmv_bne(
                    value_pdf: callable,
                    value_cdf: callable = None,
                    lower_bound: int = 0,
                    epsabs=1e-3
            ):
                if value_cdf is None:
                    def _value_cdf(x):
                        return integrate.quad(value_pdf, lower_bound, x, epsabs=epsabs)[0]

                    value_cdf = _value_cdf

                def inner(s, x):
                    return integrate.quad(lambda t: value_pdf(t) / value_cdf(t),
                                          s, x, epsabs=epsabs)[0]

                def outer(x):
                    return integrate.quad(lambda s: np.exp(-inner(s, x)),
                                          lower_bound, x, epsabs=epsabs)[0]

                def bidding(x):
                    if not hasattr(x, '__iter__'):
                        return x - outer(x)
                    else:
                        return np.array([xi - outer(xi) for xi in x])

                return bidding

            dist = self._strat_to_bidder(None, 1, None).value_distribution
            bidding = muda_tb_cmv_bne(lambda x: torch.exp(dist.log_prob(x)).cpu().numpy(),
                                      lambda x: dist.cdf(x).cpu().numpy())

            opt_bid = np.zeros_like(valuation.cpu().numpy())
            for agent in range(self.n_players):
                opt_bid[agent] = bidding(valuation[agent, :])
            return torch.tensor(opt_bid)

    def _optimal_bid_2(self, valuation: torch.Tensor or np.ndarray or float, player_position: int = 0):
        warnings.warn("No 2nd explicit BNE known.", Warning)
        return super()._optimal_bid_2(valuation, player_position)


# exp_no==6, two BNE types, BNE continua
class FPSBSplitAwardAuction2x2(MultiUnitExperiment):
    class_experiment_params = {
        'n_items': 2,
        'n_players': 2,
        'BNE1': 'PD_Sigma_BNE',
        'BNE2': 'WTA_BNE',
        'is_FPSBSplitAwardAuction2x2': True
    }

    def __init__(self, experiment_params: dict, gpu_config: GPUController, logger: Logger,
                 l_config: LearningConfiguration):
        mechanism = FPSBSplitAwardAuction(cuda=gpu_config.cuda)
        super().__init__({**self.class_experiment_params, **experiment_params}, mechanism=mechanism,
                         gpu_config=gpu_config, logger=logger, l_config=l_config)

        self.efficiency_parameter = 0.3

        self.split_award_dict = {
            'split_award': True,
            'efficiency_parameter': self.efficiency_parameter,
            'input_length': self.input_length,
            'linspace': False
        }
        self._run_setup()

    # ToDO Efficiency parameter shouldn't be hardcoded here, but it is called in strategy class from pretrain without
    # the efficiency_parameter
    @staticmethod
    def default_pretrain_transform(input_tensor, efficiency_parameter=0.3):
        temp = input_tensor.clone().detach()
        if input_tensor.shape[1] == 1:
            output_tensor = torch.cat((
                temp,
                efficiency_parameter * temp
            ), 1)
        else:
            output_tensor = temp
        return output_tensor

    def _strat_to_bidder(self, strategy, batch_size, player_position=None, cache_actions=False):
        """
        Standard strat_to_bidder method.
        """
        # ToDO One more place with hardcoded u_lo/u_hi[0]
        return ReverseBidder.uniform(
            lower=self.u_lo[0], upper=self.u_hi[0],
            strategy=strategy,
            n_items=self.n_items,
            item_interest_limit=self.item_interest_limit,
            descending_valuations=True,
            constant_marginal_values=self.constant_marginal_values,
            player_position=player_position,
            efficiency_parameter=self.efficiency_parameter,
            batch_size=batch_size
        )

    def _optimal_bid(self, valuation, player_position=None, return_payoff_dominant=True):
        valuation = super()._optimal_bid(valuation)

        # sigma/pooling equilibrium
        if self.input_length == 1 and self.n_items == 2 and valuation.shape[1] == 1:
            valuation = torch.cat(
                (valuation, self.efficiency_parameter * valuation), axis=1
            )

        # ToDO Yet one more hardcoded u_lo/u/hi[0]
        sigma_bounds = torch.ones_like(valuation, device=valuation.device)
        sigma_bounds[:, 0] = self.efficiency_parameter * self.u_hi[0]
        sigma_bounds[:, 1] = (1 - self.efficiency_parameter) * self.u_lo[0]
        # [:,0]: lower bound and [:,1] upper

        _p_sigma = (1 - self.efficiency_parameter) * self.u_lo[0]

        # highest possible p_sigma

        def G(theta):
            return _p_sigma \
                   + (_p_sigma - self.u_hi[0] * self.efficiency_parameter
                      * FPSBSplitAwardAuction2x2.value_cdf(self.u_lo[0], self.u_hi[0])(theta)) \
                   / (1 - FPSBSplitAwardAuction2x2.value_cdf(self.u_lo[0], self.u_hi[0])(theta))

        wta_bounds = 2 * sigma_bounds
        wta_bounds[:, 1] = G(valuation[:, 0])

        # cutoff value: otherwise would go to inf
        # lim = 4 * param_dict["u_hi"][0]
        # wta_bounds[wta_bounds > lim] = lim

        if return_payoff_dominant:
            return torch.cat(
                (
                    wta_bounds[:, 1].unsqueeze(0).t_(),
                    sigma_bounds[:, 1].unsqueeze(0).t_()
                ),
                axis=1
            )
        else:
            return {'sigma_bounds': sigma_bounds, 'wta_bounds': wta_bounds}

    def _optimal_bid_2(self, valuation: torch.Tensor or np.ndarray or float, player_position: int = 0):
        super()._optimal_bid_2(valuation, player_position)

        # WTA equilibrium
        # Anton and Yao, 1992: Proposition 4

        if valuation.shape[1] == 1:
            valuation = torch.cat((valuation, self.experiment_params['efficiency_parameter'] * valuation),
                                  axis=1)
        opt_bid_batch_size = 2 ** 12
        opt_bid = np.zeros(shape=(opt_bid_batch_size, valuation.shape[1]))

        #ToDO u_lo/u_hi[0]
        if 'opt_bid_function' not in globals():
            # do one-time approximation via integration
            eps = 1e-4
            val_lin = np.linspace(self.experiment_params["u_lo"][0], self.experiment_params["u_hi"][0] \
                                  - eps, opt_bid_batch_size)

            def integral(theta):
                return np.array(
                        [integrate.quad(
                            lambda x: (1 - self.value_cdf(
                                self.experiment_params["u_lo"][0],
                                self.experiment_params["u_hi"][0]
                            )(x)) ** (self.experiment_params["n_players"] - 1),
                            v,
                            self.experiment_params["u_hi"][0],
                            epsabs = eps
                        )[0] for v in theta]
                    )

            def opt_bid_100(theta):
                return theta + (integral(theta) / (
                        (1 - self.value_cdf(
                            self.experiment_params["u_lo"][0],
                            self.experiment_params["u_hi"][0]
                        )(theta)) ** (self.experiment_params["n_players"] - 1))
                    )

            opt_bid[:, 0] = opt_bid_100(val_lin)
            opt_bid[:, 1] = opt_bid_100(val_lin) \
                            - self.experiment_params["efficiency_parameter"] \
                            * self.experiment_params["u_lo"][0]
            # or more

            global opt_bid_function
            opt_bid_function = [
                interpolate.interp1d(
                    val_lin, opt_bid[:, 0],
                    fill_value='extrapolate'
                ),
                interpolate.interp1d(
                    val_lin,
                    opt_bid[:, 1], fill_value='extrapolate'
                )
            ]

        # (re)use interpolation of opt_bid done on first batch
        opt_bid = torch.tensor(
            [
                opt_bid_function[0](valuation[:, 0].cpu().numpy()),
                opt_bid_function[1](valuation[:, 0].cpu().numpy())
            ],
            device=valuation.device
        ).t_()

        opt_bid[opt_bid < 0] = 0
        opt_bid[torch.isnan(opt_bid)] = 0

        return opt_bid

    @staticmethod
    def value_cdf(u_lo, u_hi):
        """
        CDF for uniform valuations on [u_lo, u_hi].
        """

        def cdf(v: torch.Tensor):
            warnings.warn("Uniform valuations only!", Warning)

            out = (v - u_lo) / (u_hi - u_lo)

            try:
                out[out < 0] = 0
                out[out > 1] = 1
            except:
                pass

            return out

        return cdf
