import os
from abc import ABC

import torch

from bnelearn.bidder import Bidder, ReverseBidder
from bnelearn.experiment import Experiment, LearningConfiguration, GPUController, Logger
from bnelearn.mechanism import MultiItemUniformPriceAuction, MultiItemDiscriminatoryAuction, FPSBSplitAwardAuction, \
    Mechanism


class MultiUnitExperiment(Experiment, ABC):
    def __init__(self, n_players: int, mechanism: Mechanism, n_items: int, u_lo: float, u_hi: float, BNE1: str,
                 BNE2: str,
                 gpu_config: GPUController, logger: Logger, l_config: LearningConfiguration,
                 item_interest_limit: int = None):
        self.mechanism = mechanism
        self.n_players = n_players
        self.n_items = n_items
        self.u_lo = u_lo
        self.u_hi = u_hi
        self.BNE1 = BNE1
        self.BNE2 = BNE2
        self.item_interest_limit = item_interest_limit
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

    def _setup_name(self):
        auction_type_str = str(type(self.mechanism))
        auction_type_str = str(auction_type_str[len(auction_type_str) - auction_type_str[::-1].find('.'):-2])
        print(auction_type_str)
        log_name = auction_type_str + '_' + str(self.n_players) + 'players_' + str(self.n_items) + 'items'

        name = ['single_item', self.mechanism_type, self.valuation_prior,
                'symmetric', self.risk_profile, str(self.n_players) + 'p']

        self.logger.base_dir = os.path.join(*name)


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

    def _optimal_bid(self, valuation):
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
        pass

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
        pass

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
        pass

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
        pass

    def _training_loop(self, epoch):
        pass


# exp_no==6, two BNE types, BNE continua
class FPSBSplitAwardAuction2x2(MultiUnitExperiment):
    def __init__(self, n_players: int, gpu_config: GPUController, logger: Logger, l_config: LearningConfiguration):
        mechanism = FPSBSplitAwardAuction(cuda=gpu_config.cuda)
        super().__init__(n_players=n_players, mechanism=mechanism, n_items=2, u_lo=1.0, u_hi=1.4,
                         BNE1='PD_Sigma_BNE', BNE2='WTA_BNE', gpu_config=gpu_config, logger=logger, l_config=l_config)
        self.efficiency_parameter = 0.3
        self.exp_no = 6
        self.input_length = self.n_items - 1 if self.exp_no == 6 else self.n_items
        self.constant_marginal_values = None

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
            descending_valuations=self.exp_no != 6,
            # constant_marginal_values=self.constant_marginal_values,
            player_position=player_position,
            # efficiency_parameter=self.efficiency_parameter,
            batch_size=batch_size
        )

    def _setup_bidders(self):
        pass

    def _setup_learning_environment(self):
        pass

    def _setup_learners(self):
        pass

    def _setup_eval_environment(self):
        pass

    def _optimal_bid(self, valuation, player_position=None):
        pass

    def _training_loop(self, epoch):
        pass
