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
from bnelearn.experiment import GPUController, Logger, Experiment
from bnelearn.experiment.configurations import ExperimentConfiguration, LearningConfiguration, LoggingConfiguration
from bnelearn.learner import ESPGLearner
from bnelearn.mechanism import (
    MultiUnitVickreyAuction, MultiUnitUniformPriceAuction, MultiUnitDiscriminatoryAuction,
    FPSBSplitAwardAuction, Mechanism
)
from bnelearn.experiment.logger import MultiUnitAuctionLogger
from bnelearn.strategy import NeuralNetStrategy, ClosureStrategy
from bnelearn.util import metrics


class MultiUnitExperiment(Experiment, ABC):
    def __init__(self, experiment_config: dict, learning_config: LearningConfiguration,
                 logging_config: LoggingConfiguration, gpu_config: GPUController,
                 mechanism_type: Mechanism, known_bne=True):

        # ToDO How about to assigning expetiment parameters to the object and just using them from the dictionary?
        self.n_players = experiment_config.n_players
        self.n_units = experiment_config.n_units
        self.u_lo = experiment_config.u_lo
        self.u_hi = experiment_config.u_hi

        self.plot_frequency = logging_config.plot_frequency
        self.plot_xmin = min(self.u_lo)
        self.plot_xmax = max(self.u_hi)
        self.plot_ymin = 0
        self.plot_ymax = self.plot_xmax * 1.05
        self.BNE1 = experiment_config.BNE1
        self.model_sharing = experiment_config.model_sharing
        self.mechanism_type = mechanism_type

        if self.model_sharing:
            self.n_models = 1
            self._bidder2model = [0] * self.n_players
        else:
            self.n_models = self.n_players
            self._bidder2model = list(range(self.n_players))

        if experiment_config.BNE2 is not None:
            self.BNE2 = experiment_config.BNE2
        self.constant_marginal_values = experiment_config.constant_marginal_values
        self.item_interest_limit = experiment_config.item_interest_limit
        
        if experiment_config.efficiency_parameter is not None:
            self.efficiency_parameter = experiment_config.efficiency_parameter

        if experiment_config.pretrain_transform:
            self.pretrain_transform = experiment_config.pretrain_transform
        else:
            self.pretrain_transform = self.default_pretrain_transform
        
        self.input_length = experiment_config.input_length

        print('\nhyperparams\n-----------')
        for k in learning_config.learner_hyperparams.keys():
            print('{}: {}'.format(k, learning_config.learner_hyperparams[k]))
        print('-----------\n')

        super().__init__(experiment_config, learning_config, logging_config, gpu_config, known_bne)

    def _strat_to_bidder(self, strategy, batch_size, player_position=0, cache_actions=False):
        """
        Standard strat_to_bidder method.
        """
        return Bidder.uniform(
            lower=self.u_lo[player_position], upper=self.u_hi[player_position],
            strategy=strategy,
            n_items=self.n_units,
            item_interest_limit=self.item_interest_limit,
            descending_valuations=True,
            constant_marginal_values=self.constant_marginal_values,
            player_position=player_position,
            batch_size=batch_size
        )

    def _setup_bidders(self):
        epo_n = 2  # for ensure positive output of initialization
        
        for i in range(self.n_models):
            ensure_positive_output = torch.zeros(epo_n, self.input_length).uniform_(self.u_lo[i], self.u_hi[i]) \
                .sort(dim=1, descending=True)[0]
            self.models = [
                NeuralNetStrategy(
                    self.input_length,
                    hidden_nodes=self.learning_config.hidden_nodes,
                    hidden_activations=self.learning_config.hidden_activations,
                    ensure_positive_output=ensure_positive_output,
                    output_length=self.n_units
                ).to(self.gpu_config.device)
            ]

        # Pretrain
        pretrain_points = round(100 ** (1 / self.input_length))
        # pretrain_valuations = multi_unit_valuations(
        #     device = device,
        #     bounds = [param_dict["u_lo"], param_dict["u_hi"][0]],
        #     dim = param_dict["n_units"],
        #     batch_size = pretrain_points,
        #     selection = 'random' if param_dict["exp_no"] != 6 else split_award_dict
        # )
        pretrain_valuations = self._strat_to_bidder(
            ClosureStrategy(lambda x: x), self.learning_config.batch_size, 0).draw_valuations_()[:pretrain_points, :]

        for model in self.models:
            model.pretrain(pretrain_valuations, self.learning_config.pretrain_iters,
                           self.pretrain_transform)
        self.bidders = [
            self._strat_to_bidder(self.models[0 if self.model_sharing else i], self.learning_config.batch_size, i)
            for i in range(self.n_players)
        ]

    def _setup_logger(self, base_dir):
        """Creates logger for run.
        THIS IS A TEMPORARY WORKAROUND TODO
        """
        return MultiUnitAuctionLogger(exp=self, base_dir=base_dir, plot_epoch=self.plot_frequency)

    def _setup_mechanism(self):
        self.mechanism = self.mechanism_type(cuda=self.gpu_config.cuda)

    def _setup_learning_environment(self):
        self.env = AuctionEnvironment(
            mechanism=self.mechanism,
            agents=self.bidders,
            n_players=self.n_players,
            batch_size=self.learning_config.batch_size,
            strategy_to_player_closure=self._strat_to_bidder
        )

    def _setup_learners(self):
        self.learners = [
            ESPGLearner(
                model=model,
                environment=self.env,
                hyperparams=self.learning_config.learner_hyperparams,
                optimizer_type=self.learning_config.optimizer,
                optimizer_hyperparams=self.learning_config.optimizer_hyperparams,
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
            mechanism=self.mechanism,
            agents=[
                self._strat_to_bidder(bne_strategy, self.learning_config.batch_size, i)
                for i, bne_strategy in enumerate(self.bne_strategies)
            ],
            n_players=self.n_players,
            batch_size=self.learning_config.batch_size,
            strategy_to_player_closure=self._strat_to_bidder
        )

    def _get_logdir(self):
        auction_type_str = str(type(self.mechanism))
        auction_type_str = str(auction_type_str[len(auction_type_str) - auction_type_str[::-1].find('.'):-2])

        name = ['expiriments_nils', auction_type_str, str(self.n_players) + 'players_' + str(self.n_units) + 'Units']
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
            round((self.logging_config.regret_batch_size) ** (1 / self.n_units)),
            device = self.gpu_config.device
        )
        bid_profile = torch.zeros(self.learning_config.batch_size, self.n_players, self.n_units,
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
            valuation = valuation.detach()
        
        if valuation.dim() == 0:
            valuation.unsqueeze_(0) # unsqueeze if simple float
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
class MultiUnitVickreyAuction2x2(MultiUnitExperiment):
    class_experiment_config = {
        'n_units': 2,
        'n_players': 2,
        'BNE1': 'Truthful'
    }

    def __init__(self, experiment_config: dict, gpu_config: GPUController,
                 learning_config: LearningConfiguration):
        mechanism_type = MultiUnitVickreyAuction

        super().__init__({**self.class_experiment_config, **experiment_config}, mechanism_type=mechanism_type,
                         gpu_config=gpu_config, learning_config=learning_config, known_bne=True)
        self._setup_run()

    def _optimal_bid_2(self, valuation: torch.Tensor or np.ndarray or float, player_position: int = 0):
        warnings.warn("No 2nd explicit BNE known.", Warning)
        return super()._optimal_bid_2(valuation, player_position)


# exp_no==1, BNE continua
class MultiUnitUniformPriceAuction2x2(MultiUnitExperiment):
    class_experiment_config = {
        'n_units': 2,
        'n_players': 2,
        'BNE1': 'BNE1',
        'BNE2': 'BNE2'
    }

    def __init__(self, experiment_config: dict, gpu_config: GPUController,
                 learning_config: LearningConfiguration):
        mechanism_type = MultiUnitUniformPriceAuction

        super().__init__({**self.class_experiment_config, **experiment_config}, mechanism_type=mechanism_type,
                         gpu_config=gpu_config, learning_config=learning_config, known_bne=True)
        self._setup_run()

    @staticmethod
    def default_pretrain_transform(input_tensor):
        output_tensor = torch.clone(input_tensor)
        output_tensor[:, 1] = 0
        return output_tensor

    def _optimal_bid(self, valuation, player_position=None):
        valuation = super()._optimal_bid(valuation)
        opt_bid = torch.clone(valuation)
        opt_bid[:, 1] = 0
        return opt_bid

    def _optimal_bid_2(self, valuation: torch.Tensor or np.ndarray or float, player_position: int = 0):
        super()._optimal_bid_2(valuation, player_position)
        opt_bid = torch.clone(valuation)
        opt_bid[:, 0] = self.experiment_config["u_hi"][0]
        opt_bid[:, 1] = 0
        return opt_bid


# exp_no==2
class MultiUnitUniformPriceAuction2x3limit2(MultiUnitExperiment):
    class_experiment_config = {
        'n_units': 3,
        'n_players': 2,
        'BNE1': 'BNE1',
        'item_interest_limit': 2
    }

    def __init__(self, experiment_config: dict, gpu_config: GPUController,
                 learning_config: LearningConfiguration):
        mechanism_type = MultiUnitUniformPriceAuction

        super().__init__({**self.class_experiment_config, **experiment_config}, mechanism_type=mechanism_type,
                         gpu_config=gpu_config, learning_config=learning_config, known_bne=True)
        self._setup_run()

    # ToDO Once again in this method using u_hi/u_lo[0]. Is it appropriate?
    def _optimal_bid(self, valuation, player_position=None):
        valuation = super()._optimal_bid(valuation)
        opt_bid = torch.clone(valuation)
        opt_bid[:, 1] = opt_bid[:, 1] ** 2
        opt_bid[:, 2] = 0
        return opt_bid


# exp_no==4
class MultiUnitDiscriminatoryAuction2x2(MultiUnitExperiment):
    class_experiment_config = {
        'n_units': 2,
        'n_players': 2,
        'BNE1': 'BNE1',
    }

    def __init__(self, experiment_config: dict, gpu_config: GPUController,
                 learning_config: LearningConfiguration):
        mechanism_type = MultiUnitDiscriminatoryAuction

        super().__init__({**self.class_experiment_config, **experiment_config}, mechanism_type=mechanism_type,
                         gpu_config=gpu_config, learning_config=learning_config, known_bne=True)
        self._setup_run()

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


# exp_no==5
class MultiUnitDiscriminatoryAuction2x2CMV(MultiUnitExperiment):
    class_experiment_config = {
        'n_units': 2,
        'n_players': 2,
        'BNE1': 'BNE1',
        'constant_marginal_values': True
    }

    def __init__(self, experiment_config: dict, gpu_config: GPUController,
                 learning_config: LearningConfiguration):
        mechanism_type = MultiUnitDiscriminatoryAuction

        super().__init__({**self.class_experiment_config, **experiment_config}, mechanism_type=mechanism_type,
                         gpu_config=gpu_config, learning_config=learning_config, known_bne=True)
        self._setup_run()

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


# exp_no==6, two BNE types, BNE continua
class FPSBSplitAwardAuction2x2(MultiUnitExperiment):
    class_experiment_config = {
        'n_units': 2,
        'n_players': 2,
        'BNE1': 'PD_Sigma_BNE',
        'BNE2': 'WTA_BNE',
        'is_FPSBSplitAwardAuction2x2': True
    }

    def __init__(self, experiment_config: dict, gpu_config: GPUController,
                 learning_config: LearningConfiguration):
        mechanism_type = FPSBSplitAwardAuction

        assert all(u_lo > 0 for u_lo in experiment_config.u_lo), \
            '100% Unit must be valued > 0'

        super().__init__({**self.class_experiment_config, **experiment_config}, mechanism_type=mechanism_type,
                         gpu_config=gpu_config, learning_config=learning_config, known_bne=True)

        self.efficiency_parameter = 0.3

        self.split_award_dict = {
            'split_award': True,
            'efficiency_parameter': self.efficiency_parameter,
            'input_length': self.input_length,
            'linspace': False
        }
        self._setup_run()

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
            n_units=self.n_units,
            item_interest_limit=self.item_interest_limit,
            descending_valuations=True,
            constant_marginal_values=self.constant_marginal_values,
            player_position=player_position,
            efficiency_parameter=self.efficiency_parameter,
            batch_size=batch_size
        )

    def _optimal_bid(self, valuation, player_position=None, return_payoff_dominant=True):
        valuation = super()._optimal_bid(valuation)
        print(valuation)
        # sigma/pooling equilibrium
        if self.input_length == 1 and self.n_units == 2 and valuation.shape[1] == 1:
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
        print(torch.cat(
                (
                    wta_bounds[:, 1].unsqueeze(0).t_(),
                    sigma_bounds[:, 1].unsqueeze(0).t_()
                )))
        if return_payoff_dominant:
            return torch.cat(
                (
                    wta_bounds[:, 1].unsqueeze(0).t_(),
                    sigma_bounds[:, 1].unsqueeze(0).t_()
                ),
                axis=1
            )

        return {'sigma_bounds': sigma_bounds, 'wta_bounds': wta_bounds}

    def _optimal_bid_2(self, valuation: torch.Tensor or np.ndarray or float, player_position: int = 0):
        super()._optimal_bid_2(valuation, player_position)

        # WTA equilibrium
        # Anton and Yao, 1992: Proposition 4

        if valuation.shape[1] == 1:
            valuation = torch.cat((valuation, self.experiment_config['efficiency_parameter'] * valuation),
                                  axis=1)
        opt_bid_batch_size = 2 ** 12
        opt_bid = np.zeros(shape=(opt_bid_batch_size, valuation.shape[1]))

        #ToDO u_lo/u_hi[0]
        if 'opt_bid_function' not in globals():
            # do one-time approximation via integration
            eps = 1e-4
            val_lin = np.linspace(self.experiment_config["u_lo"][0], self.experiment_config["u_hi"][0] \
                                  - eps, opt_bid_batch_size)

            def integral(theta):
                return np.array(
                        [integrate.quad(
                            lambda x: (1 - self.value_cdf(
                                self.experiment_config["u_lo"][0],
                                self.experiment_config["u_hi"][0]
                            )(x)) ** (self.experiment_config["n_players"] - 1),
                            v,
                            self.experiment_config["u_hi"][0],
                            epsabs = eps
                        )[0] for v in theta]
                    )

            def opt_bid_100(theta):
                return theta + (integral(theta) / (
                        (1 - self.value_cdf(
                            self.experiment_config["u_lo"][0],
                            self.experiment_config["u_hi"][0]
                        )(theta)) ** (self.experiment_config["n_players"] - 1))
                    )

            opt_bid[:, 0] = opt_bid_100(val_lin)
            opt_bid[:, 1] = opt_bid_100(val_lin) \
                            - self.experiment_config["efficiency_parameter"] \
                            * self.experiment_config["u_lo"][0]
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
