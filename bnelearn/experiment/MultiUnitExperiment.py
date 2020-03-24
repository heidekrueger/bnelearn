import os
import warnings
from abc import ABC
from functools import partial
from itertools import product

import torch
import numpy as np
from scipy import integrate

from bnelearn.bidder import Bidder, ReverseBidder
from bnelearn.environment import AuctionEnvironment
from bnelearn.experiment import Experiment, LearningConfiguration, GPUController, Logger
from bnelearn.learner import ESPGLearner
from bnelearn.mechanism import MultiItemUniformPriceAuction, MultiItemDiscriminatoryAuction, FPSBSplitAwardAuction, \
    Mechanism
from bnelearn.strategy import NeuralNetStrategy, ClosureStrategy


class MultiUnitExperiment(Experiment, ABC):
    def __init__(self, n_players: int, mechanism: Mechanism, n_items: int, u_lo: float, u_hi: float, BNE1: str,
                 BNE2: str, gpu_config: GPUController, logger: Logger, l_config: LearningConfiguration,
                 item_interest_limit: int = None, efficiency_parameter=None):
        self.mechanism = mechanism
        self.n_players = n_players
        self.n_items = n_items
        self.u_lo = u_lo
        self.u_hi = u_hi
        self.BNE1 = BNE1
        self.BNE2 = BNE2
        self.item_interest_limit = item_interest_limit
        self.model_sharing = False
        self.input_length = self.n_items
        self.efficiency_parameter = efficiency_parameter
        self.item_interest_limit = item_interest_limit

        for vals in product(*l_config.learner_hyperparams.values()):
            self.seed, self.population_size, self.sigma, self.scale_sigma_by_model_size, self.normalize_gradients, \
            self.lr, self.weight_decay, self.momentum, self.pretrain_epoch, self.pretrain_transform = vals

        print('\nhyperparams\n-----------')
        for k in l_config.learner_hyperparams.keys():
            print('{}: {}'.format(k, eval(k)))
        print('-----------\n')

        self.learner_hyperparams = {
            'population_size': self.population_size,
            'sigma': self.sigma,
            'scale_sigma_by_model_size': self.scale_sigma_by_model_size,
            'normalize_gradients': self.normalize_gradients
        }

        super().__init__(n_players, gpu_config, logger, l_config)

    def _strat_to_bidder(self, strategy, batch_size, player_position=None, cache_actions=False):
        """
        Standard strat_to_bidder method.
        """
        return Bidder.uniform(
            lower=self.u_lo, upper=self.u_hi,
            strategy=strategy,
            n_items=self.n_items,
            item_interest_limit=self.item_interest_limit,
            descending_valuations=True,
            constant_marginal_values=True,
            player_position=player_position,
            batch_size=batch_size
        )

    def _setup_bidders(self):
        epo_n = 2  # for ensure positive output of initialization
        ensure_positive_output = torch.zeros(epo_n, self.input_length).uniform_(self.u_lo, self.u_hi) \
            .sort(dim=1, descending=True)[0]
        n_models = 1 if self.model_sharing else self.n_players
        self.models = [
            NeuralNetStrategy(
                self.input_length,
                hidden_nodes=self.l_config.hidden_nodes,
                hidden_activations=self.l_config.hidden_activations,
                ensure_positive_output=ensure_positive_output,
                output_length=self.n_items
            ).to(self.gpu_config.device)
            for _ in range(n_models)
        ]

        # Pretrain
        pretrain_points = round(100 ** (1 / self.input_length))
        # pretrain_valuations = multi_unit_valuations(
        #     device = device,
        #     bounds = [param_dict["u_lo"], param_dict["u_hi"]],
        #     dim = param_dict["n_items"],
        #     batch_size = pretrain_points,
        #     selection = 'random' if param_dict["exp_no"] != 6 else split_award_dict
        # )
        pretrain_valuations = self._strat_to_bidder(
            lambda x: x, self.l_config.batch_size, 0).draw_valuations_()[:pretrain_points, :]

        n_parameters = list()
        for model in self.models:
            n_parameters.append(sum([p.numel() for p in model.parameters()]))
            model.pretrain(pretrain_valuations, self.pretrain_epoch, self.pretrain_transform)

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
                model=self.model,
                environment=self.env,
                hyperparams=self.learner_hyperparams,
                optimizer_type=self.l_config.optimizer_type,
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

    def _setup_name(self):
        auction_type_str = str(type(self.mechanism))
        auction_type_str = str(auction_type_str[len(auction_type_str) - auction_type_str[::-1].find('.'):-2])

        name = ['expiriments_nils', auction_type_str, str(self.n_players) + 'players_' + str(self.n_items) + 'items']
        self.logger.base_dir = os.path.join(*name)

    def _training_loop(self, epoch):


        # calculate utility vs BNE
        bne_utilities = list()
        for agent in bne_env.agents:
            u = bne_env.get_reward(agent, draw_valuations=True)
            bne_utilities.append(u)
        print('bne_utilities', bne_utilities)

        with SummaryWriter(logdir, flush_secs=60) as writer:

            # torch.cuda.empty_cache()

            if logging:
                log_once(writer, 0, epoch, param_dict["n_players"], log_name,
                         n_parameters, seed, models, batch_size, learner_hyperparams,
                         optimizer_type, optimizer_hyperparams, pretrain_epoch, env)

            for e in range(epoch + 1):

                torch.cuda.empty_cache()

                # plotting
                if e % plot_epoch == 0 and logging:
                    plot_bid_function(
                        bidders,
                        optimal_bid(mechanism, param_dict, return_payoff_dominant=False),
                        optimal_bid_2(mechanism, param_dict),
                        log_name, logdir, writer, e=e,
                        bounds=[param_dict["u_lo"], param_dict["u_hi"]],
                        split_award={
                            'split_award': True,
                            "efficiency_parameter": param_dict["efficiency_parameter"],
                            "input_length": param_dict["input_length"] \
                            } if param_dict["exp_no"] == 6 else None,
                        save_fig_to_disc=save_figure_to_disc,
                        device=device
                    )
                    # if param_dict["n_items"] == 2 \
                    # and param_dict["n_players"] < 4 \
                    # and param_dict["exp_no"] != 6 \
                    # or param_dict["exp_no"] == 2:
                    #     plot_bid_function_3d(
                    #         writer, e, param_dict["exp_no"],
                    #         param_dict["n_items"], log_name, logdir, bidders,
                    #         batch_size, device, #bounds=[param_dict["u_lo"], param_dict["u_hi"]],
                    #         split_award = param_dict["exp_no"]==6,
                    #         save_fig_to_disc = save_figure_to_disc
                    #     )

                start_time = time.time()
                # torch.cuda.reset_max_memory_allocated(device=device)

                env.prepare_iteration()

                # record utilities and do optimizer step
                utilities = list()
                for i, learner in enumerate(learners):
                    u = learner.update_strategy_and_evaluate_utility()
                    utilities.append(u)
                # print('util:', np.round(u.detach().cpu().numpy(), 4), end='\t')

                elapsed = time.time() - start_time

                # memory = torch.cuda.max_memory_allocated(device=device) * (2**-17)

                # log relative utility loss induced by not playing the BNE
                against_bne_utilities = list()
                for i, model in enumerate(models):
                    u = bne_env.get_strategy_reward(model, player_position=i, draw_valuations=True)
                    against_bne_utilities.append(u)
                # print(' util_vs_bne:', np.round(u.detach().cpu().numpy(), 4), end='\t')

                # logging
                if logging:
                    log_metrics(
                        writer=writer,
                        utilities=utilities,
                        bne_utilities=bne_utilities,
                        against_bne_utilities=against_bne_utilities,
                        overhead=elapsed,
                        e=e,
                        log_name=log_name,
                        n_players=param_dict["n_players"],
                        models=models,
                        policy_metrics={
                            param_dict["BNE1"]: [
                                policy_metric(
                                    model.forward,
                                    optimal_bid(mechanism, param_dict),
                                    param_dict["n_items"],
                                    selection=split_award_dict \
                                        if param_dict["exp_no"] == 6 else 'random',
                                    bounds=[param_dict["u_lo"], param_dict["u_hi"]],
                                    item_interest_limit=param_dict["item_interest_limit"] if \
                                        "item_interest_limit" in param_dict.keys() else None,
                                    eval_points_max=2 ** 18,
                                    device=device
                                )
                                for model in models],
                            param_dict["BNE2"]: [
                                policy_metric(
                                    model.forward,
                                    optimal_bid_2(mechanism, param_dict),
                                    param_dict["n_items"],
                                    selection=split_award_dict \
                                        if param_dict["exp_no"] == 6 else 'random',
                                    bounds=[param_dict["u_lo"], param_dict["u_hi"]],
                                    item_interest_limit=param_dict["item_interest_limit"] if \
                                        "item_interest_limit" in param_dict.keys() else None,
                                    eval_points_max=2 ** 18,
                                    device=device
                                )
                                for model in models]
                        }
                    )

                print('epoch {}:\t{}s'.format(e, round(elapsed, 2)))

        #if logging:
        #    for i, model in enumerate(models):
        #        torch.save(model.state_dict(), os.path.join(logdir, 'saved_model_' + str(i) + '.pt'))


def _optimal_bid(self, valuation, player_position=None):
        if not isinstance(valuation, torch.Tensor):
            valuation = torch.tensor(valuation, dtype=torch.float, device=self.gpu_config.device)
        else:
            valuation = valuation.clone().detach()

            # unsqueeze if simple float
        if valuation.dim() == 0:
            valuation.unsqueeze_(0)

        elif valuation.shape[1] == 1:
            valuation = torch.cat((valuation, self.efficiency_parameter * valuation), 1)

        return valuation

    # ToDo Selection should be a dictionary
    @staticmethod
    def multi_unit_valuations(
            device=None,
            bounds=[0.0, 1.0],
            dim=2,
            batch_size=100,
            selection='random',
            item_interest_limit=False,
            sort=False,
    ):
        """Returns uniformly sampled valuations for multi unit auctions."""
        # for uniform vals and 2 items <=> F1(v)=v**2, F2(v)=2v-v**2

        eval_points_per_dim = round((2 * batch_size) ** (1 / dim))
        valuations = torch.zeros(eval_points_per_dim ** dim, dim, device=device)

        if selection == 'random':
            valuations.uniform_(bounds[0], bounds[1])
            valuations = valuations.sort(dim=1, descending=True)[0]

        elif 'split_award' in selection.keys():
            if 'linspace' in selection.keys() and selection['linspace']:
                valuations[:, 0] = torch.linspace(bounds[0], bounds[1],
                                                  eval_points_per_dim ** dim, device=device)
            else:
                valuations.uniform_(bounds[0], bounds[1])
            valuations[:, 1] = selection['efficiency_parameter'] * valuations[:, 0]
            # if 'input_length' in selection.keys():
            #     valuations = valuations[:,:selection['input_length']]

        else:
            lin = torch.linspace(bounds[0], bounds[1], eval_points_per_dim, device=device)
            mesh = torch.meshgrid([lin] * dim)
            for n in range(dim):
                valuations[:, n] = mesh[n].reshape(eval_points_per_dim ** dim)

            mask = valuations.sort(dim=1, descending=True)[0]
            mask = (mask == valuations).all(dim=1)
            valuations = valuations[mask]

        if isinstance(item_interest_limit, int):
            valuations[:, item_interest_limit:] = 0
        if sort:
            valuations = valuations.sort(dim=1)[0]

        return valuations


# exp_no==0
class MultiItemVickreyAuction(MultiUnitExperiment):
    def __init__(self, n_players: int, gpu_config: GPUController, logger: Logger, l_config: LearningConfiguration):
        mechanism = MultiItemUniformPriceAuction(cuda=gpu_config.cuda)
        super().__init__(n_players=n_players, mechanism=mechanism, n_items=2, u_lo=0, u_hi=1,
                         BNE1='Truthful', BNE2='Truthful', gpu_config=gpu_config, logger=logger, l_config=l_config)

    def _setup_bidders(self):
        pass

    def _setup_learning_environment(self):
        pass

    def _setup_learners(self):
        pass

    def _setup_eval_environment(self):
        pass

    def _training_loop(self, epoch):
        pass


# exp_no==1, BNE continua
class MultiItemUniformPriceAuction2x2(MultiUnitExperiment):
    def __init__(self, n_players: int, gpu_config: GPUController, logger: Logger, l_config: LearningConfiguration):
        mechanism = MultiItemUniformPriceAuction(cuda=gpu_config.cuda)
        super().__init__(n_players=n_players, mechanism=mechanism, n_items=2, u_lo=0, u_hi=1,
                         BNE1='BNE1', BNE2='Truthful', gpu_config=gpu_config, logger=logger, l_config=l_config)

    @staticmethod
    def exp_no_1_transform(input_tensor):
        output_tensor = torch.clone(input_tensor)
        output_tensor[:, 1] = 0
        return output_tensor

    def _setup_bidders(self):
        pass

    def _setup_learning_environment(self):
        pass

    def _setup_learners(self):
        pass

    def _setup_eval_environment(self):
        pass

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

    def _training_loop(self, epoch):
        pass


# exp_no==2
class MultiItemUniformPriceAuction2x3limit2(MultiUnitExperiment):
    def __init__(self, n_players: int, gpu_config: GPUController, logger: Logger, l_config: LearningConfiguration):
        mechanism = MultiItemUniformPriceAuction(cuda=gpu_config.cuda)
        super().__init__(n_players=n_players, mechanism=mechanism, n_items=3, u_lo=0, u_hi=1,
                         BNE1='BNE1', BNE2='Truthful', item_interest_limit=2,
                         gpu_config=gpu_config, logger=logger, l_config=l_config)

    def _setup_bidders(self):
        pass

    def _setup_learning_environment(self):
        pass

    def _setup_learners(self):
        pass

    def _setup_eval_environment(self):
        pass

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

    def _training_loop(self, epoch):
        pass


# exp_no==4
class MultiItemDiscriminatoryAuction2x2(MultiUnitExperiment):
    def __init__(self, n_players: int, gpu_config: GPUController, logger: Logger, l_config: LearningConfiguration):
        mechanism = MultiItemDiscriminatoryAuction(cuda=gpu_config.cuda)
        super().__init__(n_players=n_players, mechanism=mechanism, n_items=2, u_lo=0, u_hi=1,
                         BNE1='BNE1', BNE2='Truthful', gpu_config=gpu_config, logger=logger, l_config=l_config)

    def _setup_bidders(self):
        pass

    def _setup_learning_environment(self):
        pass

    def _setup_learners(self):
        pass

    def _setup_eval_environment(self):
        pass

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

    def _training_loop(self, epoch):
        pass


# exp_no==5
class MultiItemDiscriminatoryAuction2x2CMV(MultiUnitExperiment):
    def __init__(self, n_players: int, gpu_config: GPUController, logger: Logger, l_config: LearningConfiguration):
        mechanism = MultiItemDiscriminatoryAuction(cuda=gpu_config.cuda)
        super().__init__(n_players=n_players, mechanism=mechanism, n_items=2, u_lo=0, u_hi=1,
                         BNE1='BNE1', BNE2='Truthful', gpu_config=gpu_config, logger=logger, l_config=l_config)

    def _setup_bidders(self):
        pass

    def _setup_learning_environment(self):
        pass

    def _setup_learners(self):
        pass

    def _setup_eval_environment(self):
        pass

    def _optimal_bid(self, valuation, player_position=None):
        valuation = super()._optimal_bid(valuation)

        # ToDo Ugly if, right now uniform is hardcoded everywhere
        if isinstance(self._strat_to_bidder(None, 1, None).value_distribution,
                      torch.distributions.uniform.Uniform):
            return valuation / 2

        elif isinstance(self._strat_to_bidder(None, 1, None).value_distribution,
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

    def _training_loop(self, epoch):
        pass


# exp_no==6, two BNE types, BNE continua
class FPSBSplitAwardAuction2x2(MultiUnitExperiment):
    def __init__(self, n_players: int, gpu_config: GPUController, logger: Logger, l_config: LearningConfiguration):
        mechanism = FPSBSplitAwardAuction(cuda=gpu_config.cuda)
        super().__init__(n_players=n_players, mechanism=mechanism, n_items=2, u_lo=1.0, u_hi=1.4,
                         BNE1='PD_Sigma_BNE', BNE2='WTA_BNE', gpu_config=gpu_config, logger=logger, l_config=l_config)
        self.efficiency_parameter = 0.3
        self.input_length = self.n_items - 1
        self.constant_marginal_values = None
        self.return_payoff_dominant = True

        self.split_award_dict = {
            'split_award': True,
            'efficiency_parameter': self.efficiency_parameter,
            'input_length': self.input_length,
            'linspace': False
        }

    def exp_no_6_transform(self, input_tensor):
        temp = input_tensor.clone().detach()
        if input_tensor.shape[1] == 1:
            output_tensor = torch.cat((
                temp,
                self.efficiency_parameter * temp
            ), 1)
        else:
            output_tensor = temp
        return output_tensor

    def _strat_to_bidder(self, strategy, batch_size, player_position=None, cache_actions=False):
        """
        Standard strat_to_bidder method.
        """
        return ReverseBidder.uniform(
            lower=self.u_lo, upper=self.u_hi,
            strategy=strategy,
            n_items=self.n_items,
            # item_interest_limit=self.item_interest_limit,
            descending_valuations=True,
            # constant_marginal_values=self.constant_marginal_values,
            player_position=player_position,
            efficiency_parameter=self.efficiency_parameter,
            batch_size=batch_size
        )

    def _setup_learning_environment(self):
        pass

    def _setup_learners(self):
        pass

    def _setup_eval_environment(self):
        pass

    def _optimal_bid(self, valuation, player_position=None):
        valuation = super()._optimal_bid(valuation)

        # sigma/pooling equilibrium
        if self.input_length == 1 and self.n_items == 2 and valuation.shape[1] == 1:
            valuation = torch.cat(
                (valuation, self.efficiency_parameter * valuation), axis=1
            )

        sigma_bounds = torch.ones_like(valuation, device=valuation.device)
        sigma_bounds[:, 0] = self.efficiency_parameter * self.u_hi
        sigma_bounds[:, 1] = (1 - self.efficiency_parameter) * self.u_lo
        # [:,0]: lower bound and [:,1] upper

        _p_sigma = (1 - self.efficiency_parameter) * self.u_lo

        # highest possible p_sigma

        def G(theta):
            return _p_sigma \
                   + (_p_sigma - self.u_hi * self.efficiency_parameter \
                      * FPSBSplitAwardAuction2x2.value_cdf(self.u_lo, self.u_hi)(theta)) \
                   / (1 - FPSBSplitAwardAuction2x2.value_cdf(self.u_lo, self.u_hi)(theta))

        wta_bounds = 2 * sigma_bounds
        wta_bounds[:, 1] = G(valuation[:, 0])

        # cutoff value: otherwise would go to inf
        # lim = 4 * param_dict["u_hi"]
        # wta_bounds[wta_bounds > lim] = lim

        if self.return_payoff_dominant:
            return torch.cat((
                wta_bounds[:, 1].unsqueeze(0).t_(),
                sigma_bounds[:, 1].unsqueeze(0).t_()),
                axis=1
            )
        else:
            return {'sigma_bounds': sigma_bounds, 'wta_bounds': wta_bounds}

    def _training_loop(self, epoch):
        pass

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
