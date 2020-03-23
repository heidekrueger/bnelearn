from abc import ABC

import numpy as np
import scipy.integrate as integrate
from functools import partial

from bnelearn.experiment import Experiment
from bnelearn.mechanism.auctions_combinatorial import *

from bnelearn.bidder import Bidder
from bnelearn.environment import Environment, AuctionEnvironment
from bnelearn.experiment import Experiment, GPUController, Logger, LearningConfiguration

from bnelearn.learner import ESPGLearner
from bnelearn.mechanism import FirstPriceSealedBidAuction, VickreyAuction
from bnelearn.strategy import Strategy, NeuralNetStrategy, ClosureStrategy


# TODO: Currently only implemented for uniform val
class CombinatorialExperiment(Experiment, ABC):
    def __init__(self, mechanism_type, gpu_config: GPUController, logger: Logger, l_config: LearningConfiguration,
                 model_sharing=True):
        self.model_sharing = model_sharing
        self.global_bne_env = None
        self.global_bne_utility = None
        self.u_lo = 0
        self.u_hi = [1, 1, 2]
        super().__init__(mechanism_type, gpu_config, logger, l_config)
        self.plot_xmin = self.u_lo
        self.plot_xmax = np.max(self.u_hi)
        self.plot_ymin = 0
        self.plot_ymax = self.plot_xmax * 1.5

    def strat_to_bidder(self, strategy, player_position=0, batch_size=None):
        return Bidder.uniform(self.u_lo, self.u_hi[player_position], strategy, player_position=player_position,
                              batch_size=batch_size)


# mechanism/bidding implementation, plot, bnes
class LLGExperiment(CombinatorialExperiment):
    def __init__(self, mechanism_type, gpu_config: GPUController, logger: Logger, l_config: LearningConfiguration,
                 model_sharing=True):
        self.n_players = 3
        self.gamma = 0.0
        assert self.gamma == 0, "Gamma > 0 implemented yet!?"
        super().__init__(mechanism_type, gpu_config, logger, l_config, model_sharing)

    def setup_bidders(self):
        print('Setting up bidders...')
        self.models = []
        model_l1 = NeuralNetStrategy(
            self.l_config.input_length, hidden_nodes=self.l_config.hidden_nodes,
            hidden_activations=self.l_config.hidden_activations,
            ensure_positive_output=torch.tensor([float(self.u_hi[0])])
        ).to(self.gpu_config.device)
        self.models.append(model_l1)

        if not self.model_sharing:
            model_l2 = NeuralNetStrategy(
                self.l_config.input_length, hidden_nodes=self.l_config.hidden_nodes,
                hidden_activations=self.l_config.hidden_activations,
                ensure_positive_output=torch.tensor([float(self.u_hi[1])])
            ).to(self.gpu_config.device)
            self.models.append(model_l2)
        # global player
        model_g = NeuralNetStrategy(
            self.l_config.input_length, hidden_nodes=self.l_config.hidden_nodes,
            hidden_activations=self.l_config.hidden_activations,
            ensure_positive_output=torch.tensor([float(self.u_hi[2])])
        ).to(self.gpu_config.device)
        self.models.append(model_g)

        bidder_l1 = self.strat_to_bidder(model_l1, player_position=0, batch_size=self.l_config.batch_size)
        bidder_l2 = self.strat_to_bidder(model_l1, player_position=1, batch_size=self.l_config.batch_size) \
            if self.model_sharing else self.strat_to_bidder(model_l2, player_position=1,
                                                            batch_size=self.l_config.batch_size)
        bidder_g = self.strat_to_bidder(model_g, player_position=2, batch_size=self.l_config.batch_size)

        self.bidders = [bidder_l1, bidder_l2, bidder_g]
        self.n_parameters = [sum([p.numel() for p in model.parameters()]) for model in
                             [b.strategy for b in self.bidders]]

        if self.l_config.pretrain_iters > 0:
            print('\tpretraining...')
            for bidder in self.bidders:
                bidder.strategy.pretrain(bidder.valuations, self.l_config.pretrain_iters)

    def setup_experiment_domain(self):
        pass

    def setup_learning_environment(self):
        self.mechanism = LLGAuction(rule=self.mechanism_type, cuda=self.gpu_config.cuda)
        self.env = AuctionEnvironment(self.mechanism,
                                      agents=self.bidders,
                                      batch_size=self.l_config.batch_size,
                                      n_players=self.n_players,
                                      strategy_to_player_closure=self.strat_to_bidder)

    def setup_learners(self):
        learner_l1 = ESPGLearner(model=self.bidders[0].strategy,
                                 environment=self.env,
                                 hyperparams=self.l_config.learner_hyperparams,
                                 optimizer_type=self.l_config.optimizer,
                                 optimizer_hyperparams=self.l_config.optimizer_hyperparams,
                                 strat_to_player_kwargs={"player_position": 0}
                                 )
        learner_l2 = ESPGLearner(model=self.bidders[1].strategy,
                                 environment=self.env,
                                 hyperparams=self.l_config.learner_hyperparams,
                                 optimizer_type=self.l_config.optimizer,
                                 optimizer_hyperparams=self.l_config.optimizer_hyperparams,
                                 strat_to_player_kwargs={"player_position": 1}
                                 )
        learner_g = ESPGLearner(model=self.bidders[2].strategy,
                                environment=self.env,
                                hyperparams=self.l_config.learner_hyperparams,
                                optimizer_type=self.l_config.optimizer,
                                optimizer_hyperparams=self.l_config.optimizer_hyperparams,
                                strat_to_player_kwargs={"player_position": 2}
                                )
        self.learners = [learner_l1, learner_l2, learner_g]

    def optimal_bid(self, valuation, player_position):
        if not isinstance(valuation, torch.Tensor):
            valuation = torch.tensor(valuation)

        # all core-selecting rules are strategy proof for global player:
        if self.mechanism_type in ['vcg', 'proxy', 'nearest_zero', 'nearest_bid',
                                   'nearest_vcg'] and player_position == 2:
            return valuation
        # local bidders:
        if self.mechanism_type == 'vcg':
            return valuation
        if self.mechanism_type in ['proxy', 'nearest_zero']:
            bid_if_positive = 1 + torch.log(valuation * (1.0 - self.gamma) + self.gamma) / (1.0 - self.gamma)
            return torch.max(torch.zeros_like(valuation), bid_if_positive)
        if self.mechanism_type == 'nearest_bid':
            return (np.log(2) - torch.log(2.0 - (1. - self.gamma) * valuation)) / (1. - self.gamma)
        if self.mechanism_type == 'nearest_vcg':
            bid_if_positive = 2. / (2. + self.gamma) * (
                        valuation - (3. - np.sqrt(9 - (1. - self.gamma) ** 2)) / (1. - self.gamma))
            return torch.max(torch.zeros_like(valuation), bid_if_positive)
        raise ValueError('optimal bid not implemented for other rules')

    def setup_eval_environment(self):
        # TODO: Check if correct and finish
        grid_size = [50, 50, 100]
        bne_strategies = [
            ClosureStrategy(partial(self.optimal_bid, player_position=i))
            for i in range(self.n_players)
        ]

        self.global_bne_env = AuctionEnvironment(
            mechanism=self.mechanism,
            agents=[self.strat_to_bidder(bne_strategies[i], player_position=i, batch_size=self.l_config.eval_batch_size)
                    for i in range(self.n_players)],
            n_players=self.n_players,
            batch_size=self.l_config.eval_batch_size,
            strategy_to_player_closure=self.strat_to_bidder
        )

        # print("Utility in BNE (analytical): \t{:.5f}".format(bne_utility))
        global_bne_utility_sampled = torch.tensor(
            [self.global_bne_env.get_reward(a, draw_valuations=True) for a in self.global_bne_env.agents])
        print(('Utilities in BNE (sampled):' + '\t{:.5f}' * self.n_players + '.').format(*global_bne_utility_sampled))

        eps_abs = lambda us: global_bne_utility_sampled - us
        eps_rel = lambda us: 1 - us / global_bne_utility_sampled

        # environment filled with optimal players for logging
        # use higher batch size for calculating optimum
        # TODO:@Stefan: Check if this is correcgt!?
        self.bne_env = self.global_bne_env
        self.bne_utility = global_bne_utility_sampled

    def training_loop(self, epoch):
        # do in every iteration
        # save current params to calculate update norm
        prev_params = [torch.nn.utils.parameters_to_vector(model.parameters())
                       for model in self.models]
        # update models
        utilities = torch.tensor([
            learner.update_strategy_and_evaluate_utility()
            for learner in self.learners
        ])

        # TODO: Remove. This is in log_training_iteration!?
        # # play against bne
        # utilities_vs_bne = torch.tensor(
        #     [self.global_bne_env.get_strategy_reward(
        #         a.strategy, player_position=i, draw_valuations=True)
        #         for i,a in enumerate(self.env.agents)])

        # everything after this is logging --> measure overhead
        # TODO: Adjust this such that we log all models params, not just the first
        self.logger.log_training_iteration(prev_params=prev_params[0], epoch=epoch, bne_env=self.bne_env,
                                           strat_to_bidder=self.strat_to_bidder,
                                           eval_batch_size=self.l_config.eval_batch_size,
                                           bne_utility=self.bne_utility[0],
                                           bidders=self.bidders, utility=utilities[0])
        if epoch % 100:
            print(epoch)

    def setup_name(self):
        name = ['LLG', self.mechanism_type, str(self.n_players) + 'p']
        self.base_dir = os.path.join(*name)  # ToDo Redundant?
        self.logger.base_dir = os.path.join(*name)


# mechanism/bidding implementation, plot
class LLLLGGExperiment(CombinatorialExperiment):
    def __init__(self, name, mechanism_type, n_players, logging_options):
        super().__init__(name, mechanism_type, n_players, logging_options)
        self.n_players = 6

    def setup_experiment_domain(self):
        pass

    def _setup_learning_environment(self):
        pass

    def _setup_learners(self):
        pass

    def _setup_eval_environment(self):
        pass

    def _training_loop(self, epoch):
        pass
