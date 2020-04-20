"""
In this file multi-unit experiments and their analytical BNEs (if known) are defiened.

TODO:
    - support for multiple BNE
    - valuations of plotiing BNE _optimal_bid_multidiscriminatory2x2 are wrong
    - base FPSBSplitAwardAuction2x2 on MultiUnitExperiment
"""

import os
import time
import warnings
from abc import ABC
from itertools import product
import bnelearn.util.metrics as metrics


import torch
import numpy as np
from scipy import integrate, interpolate

from bnelearn.bidder import Bidder, ReverseBidder
from bnelearn.environment import AuctionEnvironment
from bnelearn.experiment import GPUController, Experiment
from bnelearn.experiment.configurations import ExperimentConfiguration, LearningConfiguration, LoggingConfiguration
from bnelearn.learner import ESPGLearner
from bnelearn.mechanism import (
    MultiUnitVickreyAuction, MultiUnitUniformPriceAuction, MultiUnitDiscriminatoryAuction,
    FPSBSplitAwardAuction, Mechanism
)

from bnelearn.strategy import NeuralNetStrategy, ClosureStrategy
from bnelearn.util import metrics as metrics

import matplotlib.pyplot as plt
from timeit import default_timer as timer

from torch.utils.tensorboard import SummaryWriter


########################################################################################################################
###                                                 BNE STRATEGIES                                                   ###
########################################################################################################################
def multiunit_bne(experiment_config, payment_rule):
    """
    Method that returns the known BNE strategy as callable if available and None otherwise.
    """
    if payment_rule == 'vickrey':
        def truthful(valuation, player_position=None):
            return torch.clone(valuation)
        return truthful

    elif payment_rule == 'discriminatory':
        if experiment_config.n_units == 2 and experiment_config.n_players == 2:
            if not experiment_config.constant_marginal_values:
                print('BNE is only approximated roughly!')
                return _optimal_bid_multidiscriminatory2x2
            else:
                # TODO get valuation_cdf from experiment_config
                raise NotImplementedError
                # return _optimal_bid_multidiscriminatory2x2CMV(valuation_cdf)

    elif payment_rule == 'uniform':
        if experiment_config.n_units == 2 and experiment_config.n_players == 2:
            return _optimal_bid_multiuniform2x2
        elif (experiment_config.n_units == 3 and experiment_config.n_players == 2 and experiment_config.item_interest_limit == 2):
            return _optimal_bid_multiuniform3x2limit2

    return None

def _optimal_bid_multidiscriminatory2x2(valuation, player_position=None):

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
    opt_bid[:,0] = b1(opt_bid[:,0])
    opt_bid[:,1] = b2(opt_bid[:,1])
    opt_bid = opt_bid.sort(dim=1, descending=True)[0]
    return opt_bid

def _optimal_bid_multidiscriminatory2x2CMV(valuation_cdf):

    if isinstance(valuation_cdf, torch.distributions.uniform.Uniform):
        def _optimal_bid(valuation, player_position=None):
            return valuation / 2

    elif isinstance(valuation_cdf, torch.distributions.normal.Normal):

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

        def _optimal_bid(valuation, player_position=None): 
            opt_bid = np.zeros_like(valuation.cpu().numpy())
            for agent in range(self.n_players):
                opt_bid[agent] = bidding(valuation[agent,:])
            return torch.tensor(opt_bid)

    return _optimal_bid

def _optimal_bid_multiuniform2x2(valuation, player_position=None):
    opt_bid = torch.clone(valuation)
    opt_bid[:,1] = 0
    return opt_bid

def _optimal_bid_multiuniform3x2limit2(valuation, player_position=None):
    opt_bid = torch.clone(valuation)
    opt_bid[:,1] = opt_bid[:, 1] ** 2
    opt_bid[:,2] = 0
    return opt_bid

def _optimal_bid_splitaward2x2_1(experiment_config, player_position=None):
    """Pooling equilibrium as in Anton and Yao, 1992."""

    efficiency_parameter = experiment_config.efficiency_parameter
    u_lo = experiment_config.u_lo
    u_hi = experiment_config.u_hi

    value_cdf = torch.distributions.Uniform(u_lo[0], u_hi[0]).cdf

    def _optimal_bid(valuation, player_position=None, return_payoff_dominant=True):
        sigma_bounds = torch.ones_like(valuation, device=valuation.device)
        sigma_bounds[:,0] = efficiency_parameter * u_hi[0]
        sigma_bounds[:,1] = (1 - efficiency_parameter) * u_lo[0]
        # [:,0]: lower bound and [:,1] upper

        _p_sigma = (1 - efficiency_parameter) * u_lo[0] # highest possible p_sigma

        def G(theta):
            return _p_sigma + (_p_sigma - u_hi[0]*efficiency_parameter * value_cdf(theta)) \
                / (1 - value_cdf(theta))

        wta_bounds = 2 * sigma_bounds
        wta_bounds[:,1] = G(valuation[:,0])

        if return_payoff_dominant:
            return torch.cat(
                (wta_bounds[:,1].unsqueeze(0).t_(),
                    sigma_bounds[:,1].unsqueeze(0).t_()),
                axis=1
            )
        return {'sigma_bounds': sigma_bounds, 'wta_bounds': wta_bounds}

    return _optimal_bid

def _optimal_bid_splitaward2x2_2(experiment_config):
    """WTA equilibrium as in Anton and Yao, 1992: Proposition 4."""

    efficiency_parameter = experiment_config.efficiency_parameter
    u_lo = experiment_config.u_lo
    u_hi = experiment_config.u_hi
    n_players = experiment_config.n_players

    def value_cdf(value):
        value = np.array(value)
        result = (value - u_lo[0]) / (u_hi[0] - u_lo[0])
        return result.clip(0, 1)

    # CONSTANTS
    opt_bid_batch_size = 2 ** 12
    eps = 1e-4

    opt_bid = np.zeros((opt_bid_batch_size, experiment_config.n_units))

    # do one-time approximation via integration
    val_lin = np.linspace(u_lo[0], u_hi[0]-eps, opt_bid_batch_size)

    def integral(theta):
        return np.array(
                [integrate.quad(
                    lambda x: (1 - value_cdf(x))**(n_players - 1), v, u_hi[0],
                    epsabs = eps
                )[0] for v in theta]
            )

    def opt_bid_100(theta):
        return theta + (integral(theta) / (
                (1 - value_cdf(theta))**(n_players - 1))
            )

    opt_bid[:,0] = opt_bid_100(val_lin)
    opt_bid[:,1] = opt_bid_100(val_lin) - efficiency_parameter * u_lo[0] # or more

    opt_bid_function = [
        interpolate.interp1d(val_lin, opt_bid[:,0], fill_value='extrapolate'),
        interpolate.interp1d(val_lin, opt_bid[:,1], fill_value='extrapolate')
    ]

    # use interpolation of opt_bid done on first batch
    def _optimal_bid(valuation, player_position=None):
        bid = torch.tensor([
                opt_bid_function[0](valuation[:,0].cpu().numpy()),
                opt_bid_function[1](valuation[:,0].cpu().numpy())
            ],
            device = valuation.device
        ).t_()
        bid[bid < 0] = 0
        bid[torch.bid(opt_bid)] = 0
        return bid

    return _optimal_bid

########################################################################################################################


class MultiUnitExperiment(Experiment, ABC):
    """
    Experiment for the standard multi-unit auctions.
    """
    def __init__(self, experiment_config: ExperimentConfiguration, learning_config: LearningConfiguration,
                 logging_config: LoggingConfiguration, gpu_config: GPUController):

        # ToDO How about to assigning expetiment parameters to the object and just using them from the dictionary?
        self.n_players = experiment_config.n_players
        self.n_units = self.n_items = experiment_config.n_units
        self.payment_rule = experiment_config.payment_rule

        self.u_lo = experiment_config.u_lo
        self.u_hi = experiment_config.u_hi

        self.plot_frequency = logging_config.plot_frequency
        self.plot_xmin = self.plot_ymin = min(self.u_lo)
        self.plot_xmax = self.plot_ymax = max(self.u_hi)

        self.model_sharing = experiment_config.model_sharing

        if self.payment_rule in ('discriminatory', 'first_price'):
            self.mechanism_type = MultiUnitDiscriminatoryAuction
        elif self.payment_rule == 'vickrey':
            self.mechanism_type = MultiUnitVickreyAuction
        elif self.payment_rule == 'uniform':
            self.mechanism_type = MultiUnitUniformPriceAuction
        else:
            raise ValueError('payment rule unknown')

        if self.model_sharing:
            self.n_models = 1
            self._bidder2model = [0] * self.n_players
        else:
            self.n_models = self.n_players
            self._bidder2model = list(range(self.n_players))

        self.constant_marginal_values = experiment_config.constant_marginal_values
        self.item_interest_limit = experiment_config.item_interest_limit

        if experiment_config.pretrain_transform is not None:
            self.pretrain_transform = experiment_config.pretrain_transform
        else:
            self.pretrain_transform = self.default_pretrain_transform

        # check for available BNE strategy
        if not isinstance(self, SplitAwardExperiment):
            self._optimal_bid = multiunit_bne(experiment_config, self.payment_rule)
        else:
            if self.n_units == 2 and self.n_players == 2:
                self._optimal_bid = _optimal_bid_splitaward2x2_1(experiment_config)
                self._optimal_bid_2 = _optimal_bid_splitaward2x2_2(experiment_config) # TODO unused
            else:
                self._optimal_bid = None
        known_bne = self._optimal_bid is not None

        self.input_length =  experiment_config.input_length

        print('\n=== Hyperparameters ===')
        for k in learning_config.learner_hyperparams.keys():
            print('{}: {}'.format(k, learning_config.learner_hyperparams[k]))
        print('=======================\n')

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
            batch_size=batch_size,
            cache_actions=cache_actions
        )

    def _setup_bidders(self):
        epo_n = 2  # for ensure positive output of initialization

        self.models = [None] * self.n_models
        for i in range(self.n_models):
            ensure_positive_output = torch.zeros(epo_n, self.input_length).uniform_(self.u_lo[i], self.u_hi[i]) \
                .sort(dim=1, descending=True)[0]
            self.models[i] = NeuralNetStrategy(
                self.input_length,
                hidden_nodes=self.learning_config.hidden_nodes,
                hidden_activations=self.learning_config.hidden_activations,
                ensure_positive_output=ensure_positive_output,
                output_length=self.n_units
            ).to(self.gpu_config.device)

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
        name = ['MultiUnit', self.payment_rule, str(self.n_players) + 'players_' + str(self.n_units) + 'units']
        return os.path.join(*name)

    def _training_loop(self, epoch):
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
            'optimal_bid': self._optimal_bid,
            'bne_utilities': bne_utilities,
            'utilities': utilities,
            'against_bne_utilities': against_bne_utilities,
            'regret': regret
        }
        self.log_training_iteration(log_params=log_params, epoch=epoch, bidders=self.bidders)

    @staticmethod
    def default_pretrain_transform(input_tensor):
        return torch.clone(input_tensor)

    # #TODO: This is identical to the one in experiment apart from n_units. @Nils: Adapt experiment and delete this one
    # def log_experiment(self, run_comment, max_epochs, run=""):
    #     self.max_epochs = max_epochs
    #     # setting up plotting
            
    #     if self.logging_config.log_metrics['opt']:
    #         # TODO: apdapt interval to be model specific! (e.g. for LLG)
    #         self.v_opt = torch.stack([
    #             torch.linspace(self.plot_xmin, self.plot_xmax, self.plot_points,
    #                            device=self.gpu_config.device)
    #             ] * self.n_units,
    #             dim = 1
    #         )
    #         self.b_opt = self._optimal_bid(self.v_opt) # only one sym. BNE supported

    #     is_ipython = 'inline' in plt.get_backend()
    #     if is_ipython:
    #         from IPython import display
    #     plt.rcParams['figure.figsize'] = [8, 5]

    #     if os.name == 'nt':
    #         raise ValueError('The run_name may not contain : on Windows!')
    #     run_name = self.logging_config.file_name + '_' + str(run)
    #     if run_comment:
    #         run_name = run_name + ' - ' + str(run_comment)

    #     self.log_dir = os.path.join(self.log_root, self.log_dir, run_name)
    #     os.makedirs(self.log_dir, exist_ok=False)
    #     if self.logging_config.save_figure_to_disk_png:
    #         os.mkdir(os.path.join(self.log_dir, 'png'))
    #     if self.logging_config.save_figure_to_disk_svg:
    #         os.mkdir(os.path.join(self.log_dir, 'svg'))

    #     print('Started run. Logging to {}'.format(self.log_dir))
    #     self.fig = plt.figure()

    #     self.writer = SummaryWriter(self.log_dir, flush_secs=30)
    #     start_time = timer()
    #     self._log_experimentparams() # TODO: what to use
    #     self._log_hyperparams()
    #     elapsed = timer() - start_time
    #     self.overhead += elapsed

    def log_training_iteration(self, epoch, bidders, log_params: dict):

        valuations = torch.stack([b.draw_valuations_() for b in bidders], dim=1)
        bids = torch.stack([b.get_action() for b in bidders], dim=1)

        if self.logging_config.log_metrics['opt']:
            # TODO: only sym. case
            valuations = torch.cat([
                valuations[:self.plot_points,:,:], self.v_opt[:,:1,:],
            ], dim=1)
            bids = torch.cat([
                bids[:self.plot_points,:,:], self.b_opt[:,:1,:],
            ], dim=1)

        # plotting
        if epoch % self.logging_config.plot_frequency == 0:
            labels = ['NPGA'] * len(bidders)
            fmts = ['bo'] * len(bidders)
            if self.logging_config.log_metrics['opt']:
                labels.append('BNE')
                fmts.append('.')
            from bnelearn.experiment.multi_unit_experiment import SplitAwardExperiment
            if isinstance(self, SplitAwardExperiment):
                xlim = [
                    [self.u_lo[0], self.u_hi[0]],
                    [self.experiment_config.efficiency_parameter * self.u_lo[0],
                     self.experiment_config.efficiency_parameter * self.u_hi[0]]
                ]
                ylim = [
                    [0, 2 * self.u_hi[0]],
                    [0, 2 * self.u_hi[0]]
                ]
            else:
                xlim = ylim = None
            super()._plot(fig=self.fig, plot_data=(valuations, bids), writer=self.writer, xlim=xlim, ylim=ylim,
                          figure_name='bid_function', epoch=epoch, labels=labels, fmts=fmts)

        # TODO: dim_of_interest for multiple BNE
        log_params['rmse'] = {
            'BNE1': torch.tensor(
                [metrics.norm_actions(bids[:,i,:], self._optimal_bid(valuations[:,i,:]))
                 for i, model in enumerate(self.models)]
            )
        }

        log_params['rel_utility_loss'] = [
            1 - u / bne_u for u, bne_u
            in zip(log_params['against_bne_utilities'], log_params['bne_utilities'])
        ]
        self._log_metrics(epoch, log_params)

        print('epoch {}:\t{}s'.format(epoch, round(log_params['elapsed'], 2)))

        # TODO: unify model saving via switch
        if epoch == self.max_epochs:
            for i, model in enumerate(self.models):
                torch.save(model.state_dict(), os.path.join(self.log_dir, 'saved_model_' + str(i) + '.pt'))


    def _log_metrics(self, epoch, metrics_dict: dict):
        """Log scalar for each player"""

        agent_name_list = ['agent_{}'.format(i) for i in range(self.experiment_config.n_players)]

        for metric_key, metric_val in metrics_dict.items():
            if isinstance(metric_val, float):
                self.writer.add_scalar('eval/' + str(metric_key), metric_val, epoch)
            elif isinstance(metric_val, list):
                self.writer.add_scalars(
                    'eval/' + str(metric_key),
                    dict(zip(agent_name_list, metric_val)),
                    epoch
                )
            elif isinstance(metric_val, dict):
                for key, val in metric_val.items():
                    self.writer.add_scalars(
                        'eval/' + str(metric_key),
                        dict(zip([name + '/' + str(key) for name in agent_name_list], val)),
                        epoch
                    )

        # log model parameters
        model_paras = [torch.norm(torch.nn.utils.parameters_to_vector(model.parameters()), p=2)
                       for model in self.models]
        self.writer.add_scalars('eval/weight_norm', dict(zip(agent_name_list, model_paras)), epoch)
# exp_no==6, two BNE types, BNE continua
class SplitAwardExperiment(MultiUnitExperiment):
    """
    Experiment of the first-price sealed bid split-award auction.
    """
    def __init__(self, experiment_config: ExperimentConfiguration, learning_config: LearningConfiguration,
                 logging_config: LoggingConfiguration, gpu_config: GPUController):


        assert all(u_lo > 0 for u_lo in experiment_config.u_lo), \
            '100% Unit must be valued > 0'

        self.efficiency_parameter = experiment_config.efficiency_parameter
        self.input_length = experiment_config.input_length
        self.payment_rule = experiment_config.payment_rule
        if self.payment_rule == 'first_price':
            self.mechanism_type = FPSBSplitAwardAuction
        else:
            raise NotImplementedError('for the split-award auction only the ' + \
                'first-price payment rule is supported')

        super().__init__(experiment_config, learning_config, logging_config, gpu_config)

        self.plot_xmin = self.plot_ymin = 0
        self.plot_xmax = self.plot_ymax = 2 * max(self.u_hi)
        self._setup_run()

    def default_pretrain_transform(self, input_tensor):
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
        """Standard strat_to_bidder method, but with ReverseBidder"""
        return ReverseBidder.uniform(
            lower=self.u_lo[0], upper=self.u_hi[0],
            strategy=strategy,
            n_units=self.n_units,
            item_interest_limit=self.item_interest_limit,
            descending_valuations=True,
            constant_marginal_values=self.constant_marginal_values,
            player_position=player_position,
            efficiency_parameter=self.efficiency_parameter,
            batch_size=batch_size,
            cache_actions=cache_actions
        )

    def _get_logdir(self):
        name = ['SplitAward', self.payment_rule, str(self.n_players) + 'players_' +
                str(self.n_units) + 'units']
        return os.path.join(*name)