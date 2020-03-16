import warnings
from abc import ABC
import torch
from pandas import np
from scipy import integrate
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from timeit import default_timer as timer
import bnelearn.util.metrics as metrics

from bnelearn.bidder import Bidder
from bnelearn.environment import Environment, AuctionEnvironment
from bnelearn.experiment import Experiment, GPUController, Logger, Plotter, Learner, os

from bnelearn.learner import ESPGLearner
from bnelearn.mechanism import FirstPriceSealedBidAuction, VickreyAuction
from bnelearn.strategy import Strategy, NeuralNetStrategy, ClosureStrategy


# general logic and setup, plot
class SingleItemExperiment(Experiment, ABC):
    def __init__(self, mechanism_type, n_players, gpu_config: GPUController, logger: Logger, plotter: Plotter,
                 learner_hyperparams: dict, optimizer_type: torch.optim, optimizer_hyperparams: dict, input_length: int,
                 hidden_nodes: list, hidden_activations: list, pretrain_iters: int = 500, batch_size: int = 2 ** 13,
                 eval_batch_size: int = 2 ** 12, cache_eval_actions: bool = True, risk: int = 1.0, n_runs: int = 1):
        super().__init__(mechanism_type, n_players, gpu_config, logger, plotter, learner_hyperparams, optimizer_type,
                         optimizer_hyperparams, input_length, hidden_nodes, hidden_activations, pretrain_iters,
                         batch_size, eval_batch_size, cache_eval_actions, risk, n_runs)


# implementation logic, e.g. model sharing. Model sharing should also override plotting function, etc.
class SymmetricPriorSingleItemExperiment(SingleItemExperiment, ABC):
    def __init__(self, mechanism_type, n_players, gpu_config: GPUController, logger: Logger, plotter: Plotter,
                 learner_hyperparams: dict, optimizer_type: torch.optim, optimizer_hyperparams: dict, input_length: int,
                 hidden_nodes: list, hidden_activations: list, pretrain_iters: int = 500, batch_size: int = 2 ** 13,
                 eval_batch_size: int = 2 ** 12, cache_eval_actions: bool = True, risk: int = 1.0, n_runs: int = 1):
        super().__init__(mechanism_type, n_players, gpu_config, logger, plotter, learner_hyperparams, optimizer_type,
                         optimizer_hyperparams, input_length, hidden_nodes, hidden_activations, pretrain_iters,
                         batch_size, eval_batch_size, cache_eval_actions, risk, n_runs)


# implementation differences to symmetric case?
class AsymmetricPriorSingleItemExperiment(SingleItemExperiment, ABC):
    def __init__(self, mechanism_type, n_players, gpu_config: GPUController, logger: Logger, plotter: Plotter,
                 learner_hyperparams: dict, optimizer_type: torch.optim, optimizer_hyperparams: dict, input_length: int,
                 hidden_nodes: list, hidden_activations: list, pretrain_iters: int = 500, batch_size: int = 2 ** 13,
                 eval_batch_size: int = 2 ** 12, cache_eval_actions: bool = True, risk: int = 1.0, n_runs: int = 1):
        super().__init__(mechanism_type, n_players, gpu_config, logger, plotter, learner_hyperparams, optimizer_type,
                         optimizer_hyperparams, input_length, hidden_nodes, hidden_activations, pretrain_iters,
                         batch_size, eval_batch_size, cache_eval_actions, risk, n_runs)


# known BNE
class UniformSymmetricPriorSingleItemExperiment(SymmetricPriorSingleItemExperiment):

    def __init__(self, mechanism_type, n_players, gpu_config: GPUController, logger: Logger, plotter: Plotter,
                 learner_hyperparams: dict, optimizer_type: torch.optim, optimizer_hyperparams: dict, input_length: int,
                 hidden_nodes: list, hidden_activations: list, pretrain_iters: int = 500, batch_size: int = 2 ** 13,
                 eval_batch_size: int = 2 ** 12, cache_eval_actions: bool = True, risk: int = 1.0, n_runs: int = 1):

        super().__init__(mechanism_type, n_players, gpu_config, logger, plotter, learner_hyperparams, optimizer_type,
                         optimizer_hyperparams, input_length, hidden_nodes, hidden_activations, pretrain_iters,
                         batch_size, eval_batch_size, cache_eval_actions, risk, n_runs)

        # plotting
        self.plot_points = min(150, self.batch_size)
        self.v_opt = np.linspace(self.plot_xmin, self.plot_xmax, 100)
        self.b_opt = self.optimal_bid(self.v_opt)

        is_ipython = 'inline' in plt.get_backend()
        if is_ipython:
            from IPython import display
        plt.rcParams['figure.figsize'] = [8, 5]




    def strat_to_bidder(self, strategy, batch_size, player_position=None, cache_actions=False):
        return Bidder.uniform(self.u_lo, self.u_hi, strategy, batch_size=batch_size,
                              player_position=player_position, cache_actions=cache_actions, risk=self.risk)

    def setup_experiment_domain(self):
        pass

    def setup_bidders(self):
        # setup_experiment_domain
        self.u_lo = 0
        self.u_hi = 10
        self.common_prior = torch.distributions.uniform.Uniform(low=self.u_lo, high=self.u_hi)

        self.positive_output_point = self.u_hi # is required  to set up bidders

        self.plot_xmin = self.u_lo
        self.plot_xmax = self.u_hi
        self.plot_ymin = 0
        self.plot_ymax = 10

        self.valuation_prior = 'uniform'  # for now, one of 'uniform' / 'normal', specific params defined in script
        self.model_sharing = True

        #symmetric and thus with model sharing
        print('Setting up bidders with model Sharing...')
        self.model = NeuralNetStrategy(
            self.input_length, hidden_nodes=self.hidden_nodes, hidden_activations=self.hidden_activations,
            ensure_positive_output=torch.tensor([float(self.positive_output_point)])
        ).to(self.gpu_config.device)

        self.bidders = [
            self.strat_to_bidder(self.model, self.batch_size, player_position)
            for player_position in range(self.n_players)]
        if self.pretrain_iters > 0:
            print('\tpretraining...')
            self.model.pretrain(self.bidders[0].valuations, self.pretrain_iters)

    def setup_learning_environment(self):
        if self.mechanism_type == 'first_price':
            self.mechanism = FirstPriceSealedBidAuction(cuda=self.gpu_config.cuda)
        elif self.mechanism_type == 'second_price':
            self.mechanism = VickreyAuction(cuda=self.gpu_config.cuda)

        self.env = AuctionEnvironment(self.mechanism, agents=self.bidders,
                                      batch_size=self.batch_size, n_players=self.n_players,
                                      strategy_to_player_closure=self.strat_to_bidder)

    def setup_learners(self):
        self.learner = ESPGLearner(
            model=self.model, environment=self.env, hyperparams=self.learner_hyperparams,
            optimizer_type=self.optimizer_type, optimizer_hyperparams=self.optimizer_hyperparams)

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
                                         batch_size=self.eval_batch_size,
                                         cache_actions=self.cache_eval_actions)
                    for i in range(self.n_players)],
            batch_size=self.eval_batch_size,
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

    def training_loop(self, writer, e):
        # do in every iteration
        # save current params to calculate update norm
        prev_params = torch.nn.utils.parameters_to_vector(self.model.parameters())
        # update model
        self.utility = self.learner.update_strategy_and_evaluate_utility()

        # everything after this is logging --> measure overhead
        start_time = timer()

        # calculate infinity-norm of update step
        new_params = torch.nn.utils.parameters_to_vector(self.model.parameters())
        self.update_norm = (new_params - prev_params).norm(float('inf'))
        # calculate utility vs bne
        self.utility_vs_bne = self.bne_env.get_reward(
            self.strat_to_bidder(self.model, batch_size=self.eval_batch_size),
            draw_valuations=False)  # False because expensive for normal priors
        self.epsilon_relative = 1 - self.utility_vs_bne / self.bne_utility
        self.epsilon_absolute = self.bne_utility - self.utility_vs_bne
        self.L_2 = metrics.norm_strategy_and_actions(self.model, self.bne_env.agents[0].actions,
                                                     self.bne_env.agents[0].valuations, 2)
        self.L_inf = metrics.norm_strategy_and_actions(self.model, self.bne_env.agents[0].actions,
                                                       self.bne_env.agents[0].valuations, float('inf'))
        self.log_metrics(writer, e)

        if e % self.logger.logging_options['plot_epoch'] == 0:
            # plot current function output
            # bidder = strat_to_bidder(model, batch_size)
            # bidder.draw_valuations_()
            v = self.bidders[0].valuations
            b = self.bidders[0].get_action()
            plot_data = (v, b)

            print(
                "Epoch {}: \tcurrent utility: {:.3f},\t utility vs BNE: {:.3f}, \tepsilon (abs/rel): ({:.5f}, {:.5f})".format(
                    e, self.utility, self.utility_vs_bne, self.epsilon_absolute, self.epsilon_relative))
            self.plot(self.fig, plot_data, writer, e)

        elapsed = timer() - start_time
        self.overhead_mins = self.overhead_mins + elapsed / 60
        writer.add_scalar('debug/overhead_mins', self.overhead_mins, e)

    def plot(self, fig, plot_data, writer: SummaryWriter or None, e=None):
        """This method should implement a vizualization of the experiment at the current state"""
        v, b = plot_data
        v = v.detach().cpu().numpy()[:self.plot_points]
        b = b.detach().cpu().numpy()[:self.plot_points]

        # create the plot
        fig = plt.gcf()
        plt.cla()
        plt.xlim(self.plot_xmin, self.plot_xmax)
        plt.ylim(self.plot_ymin, self.plot_ymax)
        plt.xlabel('valuation')
        plt.ylabel('bid')
        plt.text(self.plot_xmin + 0.05 * (self.plot_xmax - self.plot_xmin),
                 self.plot_ymax - 0.05 * (self.plot_ymax - self.plot_ymin),
                 'iteration {}'.format(e))
        plt.plot(v, b, 'o', self.v_opt, self.b_opt, 'r--')

        # show and/or log
        self._process_figure(fig, writer, e)

    def setup_name(self):
        name = ['single_item', self.mechanism_type, self.valuation_prior,
                'symmetric', self.risk_profile, str(self.n_players) + 'p']
        self.base_dir = os.path.join(*name)


# known BNE + shared setup logic across runs (calculate and cache BNE
class GaussianSymmetricPriorSingleItemExperiment(SymmetricPriorSingleItemExperiment):
    def __init__(self, name, mechanism_type, n_players, logging_options, gpu_config: GPUController, n_runs: int,
                 logger: Logger, plotter: Plotter, learner: Learner, strategy: Strategy, environment: Environment):
        super().__init__(name, mechanism_type, n_players, logging_options, gpu_config, n_runs, logger, plotter, learner,
                         strategy, environment)
        self.self.model_sharing = True

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

    def training_loop(self, writer, e):
        pass


# known BNE
class TwoPlayerUniformPriorSingleItemExperiment(AsymmetricPriorSingleItemExperiment):
    def __init__(self, name, mechanism_type, n_players, logging_options, gpu_config: GPUController, n_runs: int,
                 logger: Logger, plotter: Plotter, learner: Learner, strategy: Strategy, environment: Environment):
        super().__init__(name, mechanism_type, n_players, logging_options, gpu_config, n_runs, logger, plotter, learner,
                         strategy, environment)
