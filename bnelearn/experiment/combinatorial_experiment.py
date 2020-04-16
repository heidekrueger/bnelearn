from abc import ABC

import numpy as np
import scipy.integrate as integrate
from functools import partial
from typing import Iterable, List

from bnelearn.experiment import Experiment
from bnelearn.mechanism.auctions_combinatorial import *

from bnelearn.bidder import Bidder
from bnelearn.environment import Environment, AuctionEnvironment
from bnelearn.experiment import Experiment, GPUController, Logger, LearningConfiguration
from bnelearn.experiment.logger import LLGAuctionLogger, LLLLGGAuctionLogger

from bnelearn.learner import ESPGLearner
from bnelearn.mechanism import FirstPriceSealedBidAuction, VickreyAuction
from bnelearn.strategy import Strategy, NeuralNetStrategy, ClosureStrategy


# TODO: Currently only implemented for uniform val
# TODO: Currently only implemented for LLG and LLLLGG
class CombinatorialExperiment(Experiment, ABC):

    def __init__(self, n_players, n_local, experiment_params, gpu_config, l_config, known_bne):
        self.n_local = n_players

        self.model_sharing = experiment_params['model_sharing']
        if self.model_sharing:
            self.n_models = 2
            self._bidder2model: List[int] = [0] * n_local + [1] * (n_players - n_local)
        else:
            self.n_models = self.n_players
            self._bidder2model: List[int] = list(range(self.n_players))

        assert all(key in experiment_params for key in ['u_lo', 'u_hi']), \
            """Missing prior information!"""

        u_lo = experiment_params['u_lo']
        if isinstance(u_lo, Iterable):
            assert len(u_lo) == n_players
            u_lo = [float(l) for l in u_lo]
        else:
            u_lo = [float(u_lo)] * n_players
        self.u_lo = u_lo

        u_hi = experiment_params['u_hi']
        assert isinstance(u_hi, Iterable)
        assert len(u_hi) == n_players
        assert u_hi[1:n_local] == u_hi[:n_local-1], "local bidders should be identical"
        assert u_hi[0] < u_hi[n_local], "local bidders must be weaker than global bidder"
        self.u_hi = [float(h) for h in u_hi]

        self.plot_xmin = min(u_lo)
        self.plot_xmax = max(u_hi)
        self.plot_ymin = self.plot_xmin
        self.plot_ymax = self.plot_xmax * 1.05


        super().__init__(experiment_params=experiment_params, gpu_config=gpu_config, l_config=l_config, known_bne=known_bne)



    def _strat_to_bidder(self, strategy, batch_size, player_position=0):
        # TODO: this probably isn't the right place...
        # The model should know who is using it # TODO: Stefan: In my oppinion, it shouldn't...
        return Bidder.uniform(self.u_lo[player_position], self.u_hi[player_position], strategy, player_position=player_position,
                              batch_size=batch_size, n_items = self.n_items)

    # Currently only working for LLG and LLLLGG.
    def _setup_bidders(self):
        print('Setting up bidders...')
        # TODO: this seems tightly coupled with setup learners... can we change this?
        self.models = [None] * self.n_models

        for i in range(len(self.models)):
            # get id of canonical bidder for this model
            b_id = self._model2bidder[i][0]
            positive_output_point = torch.tensor([self.u_hi[b_id]]*self.n_items, dtype= torch.float)

            self.models[i] = NeuralNetStrategy(
                self.l_config.input_length, hidden_nodes=self.l_config.hidden_nodes,
                hidden_activations=self.l_config.hidden_activations,
                ensure_positive_output=positive_output_point,
                output_length = self.n_items
            ).to(self.gpu_config.device)

        self.bidders = [
            self._strat_to_bidder(self.models[m_id], batch_size=self.l_config.batch_size, player_position=i)
            for i, m_id in enumerate(self._bidder2model)]


        self.n_parameters = [sum([p.numel() for p in model.parameters()]) for model in
                             self.models]

        if self.l_config.pretrain_iters > 0:
            print('\tpretraining...')
            #TODO: why is this on per bidder basis when everything else is on per model basis?
            for bidder in self.bidders:
                bidder.strategy.pretrain(bidder.valuations, self.l_config.pretrain_iters)

    def _setup_learners(self):
        # TODO: the current strat_to_player kwargs is weird. Cross-check this with how values are evaluated in learner.
        # ideally, we can abstract this function away and move the functionality to the base Experiment class.
        # Implementation in SingleItem case is identical except for player position argument below.
        self.learners = []
        for m_id, model in enumerate(self.models):
            self.learners.append(ESPGLearner(model=model,
                                 environment=self.env,
                                 hyperparams=self.l_config.learner_hyperparams,
                                 optimizer_type=self.l_config.optimizer,
                                 optimizer_hyperparams=self.l_config.optimizer_hyperparams,
                                 strat_to_player_kwargs={"player_position": self._model2bidder[m_id][0]}
                                 ))


class LLGExperiment(CombinatorialExperiment):
    def __init__(self, experiment_params:dict, gpu_config: GPUController, l_config: LearningConfiguration):

        # coupling between valuations only 0 implemented currently
        self.gamma = 0.0
        assert self.gamma == 0, "Gamma > 0 implemented yet!?"
        # Experiment specific parameters
        experiment_params['n_players'] = 3
        n_local =2
        self.n_players = experiment_params['n_players'] # TODO: this will also be set in superclass but le'ts use it below
        self.n_local = 2
        
        self.n_items = 1 # TODO: what does this do? can we get rid of it?
        self.payment_rule = experiment_params['payment_rule']


        # TODO: This is not exhaustive, other criteria must be fulfilled for the bne to be known! (i.e. uniformity, bounds, etc)
        known_bne = self.payment_rule in ['first_price', 'vcg', 'nearest_bid','nearest_zero', 'proxy', 'nearest_vcg']
        
        super().__init__(3, 2, experiment_params, gpu_config, l_config, known_bne)

    def _setup_logger(self, base_dir):
        return LLGAuctionLogger(self, base_dir)


    def _setup_mechanism(self):
        self.mechanism = LLGAuction(rule = self.payment_rule)


    def _setup_learning_environment(self):
        # TODO: is this the same for all settings (single, multi-unit, combinatorial???)
        self.env = AuctionEnvironment(self.mechanism,
                                      agents=self.bidders,
                                      batch_size=self.l_config.batch_size,
                                      n_players=self.n_players,
                                      strategy_to_player_closure=self._strat_to_bidder)

    def _optimal_bid(self, valuation, player_position):
        if not isinstance(valuation, torch.Tensor):
            valuation = torch.tensor(valuation)

        # all core-selecting rules are strategy proof for global player:
        if self.payment_rule in ['vcg', 'proxy', 'nearest_zero', 'nearest_bid',
                                   'nearest_vcg'] and player_position == 2:
            return valuation
        # local bidders:
        if self.payment_rule == 'vcg':
            return valuation
        if self.payment_rule in ['proxy', 'nearest_zero']:
            bid_if_positive = 1 + torch.log(valuation * (1.0 - self.gamma) + self.gamma) / (1.0 - self.gamma)
            return torch.max(torch.zeros_like(valuation), bid_if_positive)
        if self.payment_rule == 'nearest_bid':
            return (np.log(2) - torch.log(2.0 - (1. - self.gamma) * valuation)) / (1. - self.gamma)
        if self.payment_rule == 'nearest_vcg':
            bid_if_positive = 2. / (2. + self.gamma) * (
                        valuation - (3. - np.sqrt(9 - (1. - self.gamma) ** 2)) / (1. - self.gamma))
            return torch.max(torch.zeros_like(valuation), bid_if_positive)
        raise ValueError('optimal bid not implemented for other rules')

    def _setup_eval_environment(self):
        bne_strategies = [
            ClosureStrategy(partial(self._optimal_bid, player_position=i))
            for i in range(self.n_players)
        ]

        bne_env = AuctionEnvironment(
            mechanism=self.mechanism,
            agents=[self._strat_to_bidder(bne_strategies[i], player_position=i, batch_size=self.l_config.eval_batch_size)
                    for i in range(self.n_players)],
            n_players=self.n_players,
            batch_size=self.l_config.eval_batch_size,
            strategy_to_player_closure=self._strat_to_bidder
        )

        self.bne_env = bne_env

        bne_utilities_sampled = torch.tensor(
            [bne_env.get_reward(a, draw_valuations=True) for a in bne_env.agents])

        print(('Utilities in BNE (sampled):' + '\t{:.5f}' * self.n_players + '.').format(*bne_utilities_sampled))
        print("No closed form solution for BNE utilities available in this setting. Using sampled value as baseline.")
        # TODO: possibly redraw bne-env valuations over time to eliminate bias
        self.bne_utilities = bne_utilities_sampled

    def _get_logdir(self):
        name = ['LLG', self.payment_rule]
        return os.path.join(*name)

    def _training_loop(self, epoch, logger):
        # do in every iteration
        # save current params to calculate update norm
        prev_params = [torch.nn.utils.parameters_to_vector(model.parameters())
                       for model in self.models]
        # update models
        utilities = torch.tensor([
            learner.update_strategy_and_evaluate_utility()
            for learner in self.learners
        ])
        # everything after this is logging --> measure overhead
        log_params = {}
        logger.log_training_iteration(prev_params=prev_params, epoch=epoch,
                                           strat_to_bidder=self._strat_to_bidder,                                           
                                           bne_utilities=self.bne_utilities, utilities=utilities,
                                           log_params=log_params)
        if epoch % 10 == 0:
            print("epoch {}, utilities: ".format(epoch))
            for i in range(len(utilities)):
                print("{}: {:.5f}".format(i, utilities[i]))
            logger.log_ex_interim_regret(epoch=epoch, mechanism=self.mechanism, env=self.env, learners=self.learners, 
                                          u_lo=self.u_lo, u_hi=self.u_hi, regret_batch_size=self.regret_batch_size, regret_grid_size=self.regret_grid_size)

# mechanism/bidding implementation, plot
class LLLLGGExperiment(CombinatorialExperiment):
    def __init__(self, experiment_params, gpu_config: GPUController, l_config: LearningConfiguration):
        self.n_local = 4
        experiment_params['n_players'] = 6
        self.n_items = 2
        assert l_config.input_length == 2, "Learner config has to take 2 inputs!"
        super().__init__(experiment_params, gpu_config, l_config)

    def _setup_logger(self, base_dir):
        return LLGAuctionLogger(self, base_dir)

    def _setup_mechanism(self):
        self.mechanism = LLLLGGAuction(rule=self.payment_rule)

    def _setup_learning_environment(self):
        #TODO: We could handover self.mechanism in experiment and move _self_learning_environment up, since it is identical in most places
        self.mechanism = LLLLGGAuction(rule=self.mechanism_type, core_solver='NoCore', parallel=1, cuda=self.gpu_config.cuda)
        self.env = AuctionEnvironment(self.mechanism,
                                      agents=self.bidders,
                                      batch_size=self.l_config.batch_size,
                                      n_players=self.n_players,
                                      strategy_to_player_closure=self._strat_to_bidder)


    def _get_logdir(self):
        name = ['LLLLGG', self.mechanism_type, str(self.n_players) + 'p']
        self.base_dir = os.path.join(*name)  # ToDo Redundant?
        return os.path.join(*name)

    def _optimal_bid(self, valuation, player_position):
        # No bne eval known
        #TODO: Return dummy value for now
        return valuation * 9999

    def _training_loop(self, epoch, logger):
        # do in every iteration
        # save current params to calculate update norm
        prev_params = [torch.nn.utils.parameters_to_vector(model.parameters())
                       for model in self.models]
        # update models
        utilities = torch.tensor([
            learner.update_strategy_and_evaluate_utility()
            for learner in self.learners
        ])
        # everything after this is logging --> measure overhead
        # TODO: Adjust this such that we log all models params, not just the first
        log_params = {}
        logger.log_training_iteration(prev_params=prev_params, epoch=epoch,
                                           strat_to_bidder=self._strat_to_bidder,
                                           eval_batch_size=self.l_config.eval_batch_size,
                                           utilities=utilities, log_params=log_params)
        if epoch % 100 == 0:
            print("epoch {}, utilities: ".format(epoch))
            for i in range(len(utilities)):
                print("{}: {:.5f}".format(i, utilities[i]))
            logger.log_ex_interim_regret(epoch=epoch, mechanism=self.mechanism, env=self.env, learners=self.learners, 
                                          u_lo=self.u_lo, u_hi=self.u_hi, regret_batch_size=self.regret_batch_size, regret_grid_size=self.regret_grid_size)