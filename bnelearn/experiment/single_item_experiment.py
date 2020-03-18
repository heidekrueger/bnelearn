import os
import warnings
from abc import ABC
import torch

from scipy import integrate

from bnelearn.bidder import Bidder
from bnelearn.environment import Environment, AuctionEnvironment
from bnelearn.experiment import Experiment, GPUController, Logger, LearningConfiguration

from bnelearn.learner import ESPGLearner
from bnelearn.mechanism import FirstPriceSealedBidAuction, VickreyAuction
from bnelearn.strategy import Strategy, NeuralNetStrategy, ClosureStrategy


# general logic and setup, plot
class SingleItemExperiment(Experiment, ABC):
    def __init__(self, mechanism_type, gpu_config: GPUController, logger: Logger, l_config: LearningConfiguration,
                 risk: float = 1.0):
        super().__init__(mechanism_type, gpu_config, logger, l_config, risk)


# implementation logic, e.g. model sharing. Model sharing should also override plotting function, etc.
class SymmetricPriorSingleItemExperiment(SingleItemExperiment, ABC):
    def __init__(self, mechanism_type, gpu_config: GPUController, logger: Logger, l_config: LearningConfiguration,
                 risk: float = 1.0):
        super().__init__(mechanism_type, gpu_config, logger, l_config, risk)


# implementation differences to symmetric case?
class AsymmetricPriorSingleItemExperiment(SingleItemExperiment, ABC):
    def __init__(self, mechanism_type, gpu_config: GPUController, logger: Logger, l_config: LearningConfiguration,
                 risk: float = 1.0):
        super().__init__(mechanism_type, gpu_config, logger, l_config, risk)


# known BNE
class UniformSymmetricPriorSingleItemExperiment(SymmetricPriorSingleItemExperiment):

    def __init__(self, mechanism_type, gpu_config: GPUController, logger: Logger, l_config: LearningConfiguration,
                 risk: float = 1.0):

        self.n_players = 2
        super().__init__(mechanism_type, gpu_config, logger, l_config, risk)

    def strat_to_bidder(self, strategy, batch_size, player_position=None, cache_actions=False):
        return Bidder.uniform(self.u_lo, self.u_hi, strategy, batch_size=batch_size,
                              player_position=player_position, cache_actions=cache_actions, risk=self.risk)

    def setup_bidders(self):
        # setup_experiment_domain
        self.u_lo = 0
        self.u_hi = 10
        self.common_prior = torch.distributions.uniform.Uniform(low=self.u_lo, high=self.u_hi)

        self.positive_output_point = self.u_hi  # is required  to set up bidders

        self.plot_xmin = self.u_lo
        self.plot_xmax = self.u_hi
        self.plot_ymin = 0
        self.plot_ymax = 10

        self.valuation_prior = 'uniform'  # for now, one of 'uniform' / 'normal', specific params defined in script
        self.model_sharing = True

        # symmetric and thus with model sharing
        print('Setting up bidders with model Sharing...')
        self.model = NeuralNetStrategy(
            self.l_config.input_length, hidden_nodes=self.l_config.hidden_nodes,
            hidden_activations=self.l_config.hidden_activations,
            ensure_positive_output=torch.tensor([float(self.positive_output_point)])
        ).to(self.gpu_config.device)

        self.bidders = [
            self.strat_to_bidder(self.model, self.l_config.batch_size, player_position)
            for player_position in range(self.n_players)]
        if self.l_config.pretrain_iters > 0:
            print('\tpretraining...')
            self.model.pretrain(self.bidders[0].valuations, self.l_config.pretrain_iters)

    def setup_learning_environment(self):
        if self.mechanism_type == 'first_price':
            self.mechanism = FirstPriceSealedBidAuction(cuda=self.gpu_config.cuda)
        elif self.mechanism_type == 'second_price':
            self.mechanism = VickreyAuction(cuda=self.gpu_config.cuda)

        self.env = AuctionEnvironment(self.mechanism, agents=self.bidders,
                                      batch_size=self.l_config.batch_size, n_players=self.n_players,
                                      strategy_to_player_closure=self.strat_to_bidder)

    def setup_learners(self):
        self.learner = ESPGLearner(
            model=self.model, environment=self.env, hyperparams=self.l_config.learner_hyperparams,
            optimizer_type=self.l_config.optimizer, optimizer_hyperparams=self.l_config.optimizer_hyperparams)

    def optimal_bid(self, valuation):
        if self.mechanism_type == 'second_price':
            return valuation
        elif self.mechanism_type == 'first_price':
            return self.u_lo + (valuation - self.u_lo) * (self.n_players - 1) / (self.n_players - 1.0 + self.risk)
        else:
            raise ValueError('Invalid Auction Mechanism')

    def setup_eval_environment(self):
        n_processes_optimal_strategy = 44 if self.valuation_prior != 'uniform' and \
                                             self.mechanism_type != 'second_price' else 0
        bneStrategy = ClosureStrategy(self.optimal_bid, parallel=n_processes_optimal_strategy)

        # define bne agents once then use them in all runs
        global_bne_env = AuctionEnvironment(
            self.mechanism,
            agents=[self.strat_to_bidder(bneStrategy,
                                         player_position=i,
                                         batch_size=self.l_config.eval_batch_size,
                                         cache_actions=self.l_config.cache_eval_actions)
                    for i in range(self.n_players)],
            batch_size=self.l_config.eval_batch_size,
            n_players=self.n_players,
            strategy_to_player_closure=self.strat_to_bidder
        )

        if self.mechanism_type == 'first_price':
            if self.valuation_prior == 'uniform':
                global_bne_utility = (self.risk * (self.u_hi - self.u_lo) / (self.n_players - 1 + self.risk)) ** \
                                     self.risk / (self.n_players + self.risk)
            elif self.valuation_prior == 'normal':
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    # don't print scipy accuracy warnings
                    global_bne_utility, error_estimate = integrate.dblquad(
                        lambda x, v: self.common_prior.cdf(x) ** (self.n_players - 1) * self.common_prior.log_prob(
                            v).exp(),
                        0, float('inf'),  # outer boundaries
                        lambda v: 0, lambda v: v)  # inner boundaries
                    if error_estimate > 1e-6:
                        warnings.warn('Error in optimal utility might not be negligible')
        elif self.mechanism_type == 'second_price':
            F = self.common_prior.cdf
            f = lambda x: self.common_prior.log_prob(torch.tensor(x)).exp()
            f1n = lambda x, n: n * F(x) ** (n - 1) * f(x)

            global_bne_utility, error_estimate = integrate.dblquad(
                lambda x, v: (v - x) * f1n(x, self.n_players - 1) * f(v),
                0, float('inf'),  # outer boundaries
                lambda v: 0, lambda v: v)  # inner boundaries

            if error_estimate > 1e-6:
                warnings.warn('Error bound on analytical bne utility is not negligible!')
        else:
            raise ValueError("Invalid auction mechanism.")

        global_bne_utility_sampled = global_bne_env.get_reward(global_bne_env.agents[0], draw_valuations=True)
        print("Utility in BNE (analytical): \t{:.5f}".format(global_bne_utility))
        print('Utility in BNE (sampled): \t{:.5f}'.format(global_bne_utility_sampled))

        # environment filled with optimal players for logging
        # use higher batch size for calculating optimum
        self.bne_env = global_bne_env
        self.bne_utility = global_bne_utility

    def training_loop(self, epoch):
        # do in every iteration
        # save current params to calculate update norm
        prev_params = torch.nn.utils.parameters_to_vector(self.model.parameters())
        # update model
        utility = self.learner.update_strategy_and_evaluate_utility()

        # everything after this is logging --> measure overhead
        self.logger.log_training_iteration(prev_params=prev_params, epoch=epoch, bne_env=self.bne_env,
                                           strat_to_bidder=self.strat_to_bidder,
                                           eval_batch_size=self.l_config.eval_batch_size, bne_utility=self.bne_utility,
                                           bidders=self.bidders, utility=utility)

    def setup_name(self):
        name = ['single_item', self.mechanism_type, self.valuation_prior,
                'symmetric', self.risk_profile, str(self.n_players) + 'p']
        self.base_dir = os.path.join(*name)  # ToDo Redundant?
        self.logger.base_dir = os.path.join(*name)


# known BNE + shared setup logic across runs (calculate and cache BNE
class GaussianSymmetricPriorSingleItemExperiment(SymmetricPriorSingleItemExperiment):
    def __init__(self, mechanism_type, gpu_config: GPUController, logger: Logger, l_config: LearningConfiguration,
                 risk: float = 1.0):
        super().__init__(mechanism_type, gpu_config, logger, l_config, risk)

    def setup_name(self):
        pass

    def setup_experiment_domain(self):
        pass

    def setup_bidders(self):
        pass

    def setup_learning_environment(self):
        pass

    def setup_learners(self):
        pass

    def setup_eval_environment(self):
        pass

    def training_loop(self, epoch):
        pass


# known BNE
class TwoPlayerUniformPriorSingleItemExperiment(AsymmetricPriorSingleItemExperiment):
    def __init__(self, mechanism_type, gpu_config: GPUController, logger: Logger, l_config: LearningConfiguration,
                 risk: float = 1.0):
        super().__init__(mechanism_type, gpu_config, logger, l_config, risk)
