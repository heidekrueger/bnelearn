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
    def __init__(self, gpu_config: GPUController, experiment_params: dict, logger: Logger, l_config: LearningConfiguration,
                 model_sharing=True):
        super().__init__(gpu_config, experiment_params, logger, l_config)
        self.model_sharing = model_sharing
        self.global_bne_env = None
        self.global_bne_utility = None

    def _strat_to_bidder(self, strategy, player_position=0, batch_size=None):
        return Bidder.uniform(self.u_lo[player_position], self.u_hi[player_position], strategy, player_position=player_position,
                              batch_size=batch_size, n_items = self.n_items)

    # Currently only working for LLG and LLLLGG.
    def _setup_bidders(self):
        print('Setting up bidders...')
        self.models = [None] * 2 if self.model_sharing else [None] * self.n_players

        for i in range(len(self.models)):
            self.models[i] = NeuralNetStrategy(
                self.l_config.input_length, hidden_nodes=self.l_config.hidden_nodes,
                hidden_activations=self.l_config.hidden_activations,
                output_length = self.n_items,
                ensure_positive_output=None
            ).to(self.gpu_config.device)

        self.bidders = []
        for i in range(self.n_players):
            if self.model_sharing:
                self.bidders.append(self._strat_to_bidder(self.models[int(i/self.n_local)], player_position=i,
                                                     batch_size=self.l_config.batch_size))
            else:
                self.bidders.append(self._strat_to_bidder(self.models[i], player_position=i,
                                                     batch_size=self.l_config.batch_size))

        self.n_parameters = [sum([p.numel() for p in model.parameters()]) for model in
                             [b.strategy for b in self.bidders]]

        if self.l_config.pretrain_iters > 0:
            print('\tpretraining...')
            for bidder in self.bidders:
                bidder.strategy.pretrain(bidder.valuations, self.l_config.pretrain_iters)

    def _setup_learners(self):
        self.learners = []
        for i in range(len(self.models)):
            self.learners.append(ESPGLearner(model=self.models[i],
                                 environment=self.env,
                                 hyperparams=self.l_config.learner_hyperparams,
                                 optimizer_type=self.l_config.optimizer,
                                 optimizer_hyperparams=self.l_config.optimizer_hyperparams,
                                 strat_to_player_kwargs={"player_position": i*self.n_local}
                                 ))


    def _training_loop(self, epoch):
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
                                           strat_to_bidder=self._strat_to_bidder,
                                           eval_batch_size=self.l_config.eval_batch_size,
                                           bne_utility=self.bne_utility[0],
                                           bidders=self.bidders, utility=utilities[0])
        if epoch % 100 == 0:
            print("epoch {}, utilities: ".format(epoch))
            for i in range(len(utilities)):
                print("{}: {:.5f}".format(i, utilities[i]))
# mechanism/bidding implementation, plot, bnes
class LLGExperiment(CombinatorialExperiment):
    def __init__(self, experiment_params:dict, gpu_config: GPUController, logger: Logger, l_config: LearningConfiguration,
                 model_sharing=True):
        # Experiment specific parameters
        self.n_local = 2
        self.gamma = 0.0
        assert self.gamma == 0, "Gamma > 0 implemented yet!?"
        experiment_params['n_players'] = 3
        self.n_items = 1
        super().__init__(gpu_config, experiment_params, logger, l_config, model_sharing)

        # Experiment general parameters
        self.n_players = 3

        self._run_setup()

    def _setup_learning_environment(self):
        self.mechanism = LLGAuction(rule=self.mechanism_type, cuda=self.gpu_config.cuda)
        self.env = AuctionEnvironment(self.mechanism,
                                      agents=self.bidders,
                                      batch_size=self.l_config.batch_size,
                                      n_players=self.n_players,
                                      strategy_to_player_closure=self._strat_to_bidder)



    def _optimal_bid(self, valuation, player_position):
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

    def _setup_eval_environment(self):
        # TODO: Check if correct and finish
        grid_size = [50, 50, 100]
        bne_strategies = [
            ClosureStrategy(partial(self._optimal_bid, player_position=i))
            for i in range(self.n_players)
        ]

        self.global_bne_env = AuctionEnvironment(
            mechanism=self.mechanism,
            agents=[self._strat_to_bidder(bne_strategies[i], player_position=i, batch_size=self.l_config.eval_batch_size)
                    for i in range(self.n_players)],
            n_players=self.n_players,
            batch_size=self.l_config.eval_batch_size,
            strategy_to_player_closure=self._strat_to_bidder
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

    def _setup_name(self):
        name = ['LLG', self.mechanism_type, str(self.n_players) + 'p']
        self.base_dir = os.path.join(*name)  # ToDo Redundant?
        self.logger.base_dir = os.path.join(*name)


# mechanism/bidding implementation, plot
class LLLLGGExperiment(CombinatorialExperiment):
    def __init__(self, experiment_params, gpu_config: GPUController, logger: Logger, l_config: LearningConfiguration,
                 model_sharing=True):
        self.n_local = 4
        self.gamma = 0.0
        experiment_params['n_players'] = 6
        self.n_items = 2
        super().__init__(gpu_config, experiment_params, logger, l_config, model_sharing)
        self.bne_utility = [9999] * 2 if experiment_params['model_sharing'] else [9999] * 6
        self._run_setup()


    def _setup_learning_environment(self):
        #TODO: We could handover self.mechanism in experiment and move _self_learning_environment up, since it is identical in most places
        self.mechanism = LLLLGGAuction(rule=self.mechanism_type, core_solver='NoCore', parallel=1, cuda=self.gpu_config.cuda)
        self.env = AuctionEnvironment(self.mechanism,
                                      agents=self.bidders,
                                      batch_size=self.l_config.batch_size,
                                      n_players=self.n_players,
                                      strategy_to_player_closure=self._strat_to_bidder)


    def _setup_eval_environment(self):
        # No bne eval known
        # Add _setup_eval_regret_environment(self)
        pass

    def _setup_name(self):
        name = ['LLLLGG', self.mechanism_type, str(self.n_players) + 'p']
        self.base_dir = os.path.join(*name)  # ToDo Redundant?
        self.logger.base_dir = os.path.join(*name)

    def _optimal_bid(self):
        # No bne eval known
        pass