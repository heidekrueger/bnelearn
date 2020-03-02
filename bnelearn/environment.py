# -*- coding: utf-8 -*-
"""
This module contains environments - a collection of players and
possibly state histories that is used to control game playing and
implements reward allocation to agents.
"""
import sys
from abc import ABC, abstractmethod
from collections.abc import Iterable
from typing import Callable, Set
import torch

from bnelearn.bidder import Bidder, MatrixGamePlayer, Player
from bnelearn.mechanism import MatrixGame, Mechanism
from bnelearn.strategy import Strategy


class Environment(ABC):
    """Environment

    An Environment object 'manages' a repeated game, i.e. manages the current players
    and their models, collects players' actions, distributes rewards,
    runs the game itself and allows 'simulations' as in 'how would a mutated player
    do in the current setting'?
    """
    def __init__(self,
                 agents: Iterable,
                 n_players=2,
                 batch_size=1,
                 strategy_to_player_closure: Callable or None=None,
                 **kwargs #pylint: disable=unused-argument
                 ):
        assert isinstance(agents, Iterable), "iterable of agents must be supplied"

        self._strategy_to_player = strategy_to_player_closure
        self.batch_size = batch_size
        self.n_players = n_players

        # transform agents into players, if specified as Strategies:
        agents = [
            self._strategy_to_player(agent, batch_size, player_position) if isinstance(agent, Strategy) else agent
            for player_position, agent in enumerate(agents)
        ]
        self.agents: Iterable[Player] = agents
        self.__len__ = self.agents.__len__

        # test whether all provided agents implement correct batch_size
        for i, agent in enumerate(self.agents):
            if agent.batch_size != self.batch_size:
                raise ValueError("Agent {}'s batch size does not match that of the environment!".format(i))

    @abstractmethod
    def get_reward(self, agent: Player, **kwargs) -> torch.Tensor:
        """Return reward for a player playing a certain strategy"""
        pass #pylint: disable=unnecessary-pass

    def get_strategy_reward(self, strategy: Strategy, player_position: int,
                            draw_valuations=False, aggregate_batch = True,
                            **strat_to_player_kwargs) -> torch.Tensor:
        """
        Returns reward of a given strategy in given environment agent position.
        """
        if not self._strategy_to_player:
            raise NotImplementedError('This environment has no strategy_to_player closure!')
        agent = self._strategy_to_player(strategy,
                                         batch_size=self.batch_size,
                                         player_position=player_position, **strat_to_player_kwargs)
        return self.get_reward(agent, draw_valuations = draw_valuations, aggregate=aggregate_batch)

    def get_strategy_action_and_reward(self, strategy: Strategy, player_position: int,
                            draw_valuations=False, **strat_to_player_kwargs) -> torch.Tensor:
        """
        Returns reward of a given strategy in given environment agent position.
        """
        if not self._strategy_to_player:
            raise NotImplementedError('This environment has no strategy_to_player closure!')
        agent = self._strategy_to_player(strategy,
                                         batch_size=self.batch_size,
                                         player_position=player_position, **strat_to_player_kwargs)
        action = agent.get_action()
        return action, self.get_reward(agent, draw_valuations = draw_valuations, aggregate = False)


    def _generate_agent_actions(self, exclude: Set[int] or None = None):
        """
        Generator function yielding batches of bids for each environment agent
        that is not excluded.

        args:
            exclude:
                A set of player positions to exclude.
                Used e.g. to generate action profile of all but currently learning player.

        yields:
            tuple(player_position, action) for each relevant bidder
        """

        if exclude is None:
            exclude = set()

        for agent in (a for a in self.agents if a.player_position not in exclude):
            yield(agent.player_position, agent.get_action())

    def prepare_iteration(self):
        """Prepares the interim-stage of a Bayesian game,
            (e.g. in an Auction, draw bidders' valuations)
        """
        pass #pylint: disable=unnecessary-pass

    def is_empty(self):
        """True if no agents in the environment"""
        return len(self) == 0


class MatrixGameEnvironment(Environment):
    """ An environment for matrix games.

        Important features of matrix games for implementation:
        - not necessarily symmetric, i.e. each player has a fixed position
        - agents strategies do not take any input, the actions only depend
           on the game itself (no Bayesian Game)
    """

    def __init__(self,
                 game: MatrixGame,
                 agents,
                 n_players=2,
                 batch_size=1,
                 strategy_to_player_closure=None,
                 **kwargs):

        super().__init__(agents, n_players=n_players, batch_size=batch_size,
                         strategy_to_player_closure=strategy_to_player_closure)
        self.game = game

    def get_reward(self, agent, **kwargs) -> torch.tensor: #pylint: disable=arguments-differ
        """
            Simulates one batch of the environment and returns the average reward for `agent` as a scalar tensor.
        """

        if isinstance(agent, Strategy):
            agent: MatrixGamePlayer = self._strategy_to_player(
                agent,
                batch_size=self.batch_size,
                **kwargs
                )
        player_position = agent.player_position

        action_profile = torch.zeros(self.batch_size, self.game.n_players,
                                     dtype=torch.long, device=agent.device)

        action_profile[:, player_position] = agent.get_action().view(self.batch_size)

        for opponent_action in self._generate_agent_actions(exclude = set([player_position])):
            position, action = opponent_action
            action_profile[:, position] = action.view(self.batch_size)

        allocation, payments = self.game.play(action_profile.view(self.batch_size, self.n_players, -1))
        utilities =  agent.get_utility(allocation[:,player_position,:], payments[:,player_position])

        return utilities.mean()


class AuctionEnvironment(Environment):
    """
        An environment of agents to play against and evaluate strategies.

        In particular this means:
            - an iterable of sets of -i players that a strategy of a single player can be tested against
            - accept strategy as argument, then play batch_size rounds and return the reward

        Args:
        ... (TODO: document)

        strategy_to_bidder_closure: A closure (strategy, batch_size) -> Bidder to
            transform strategies into a Bidder compatible with the environment
    """

    def __init__(self, mechanism: Mechanism, agents: Iterable,
                 batch_size=100, n_players=None, strategy_to_player_closure: Callable[[Strategy], Bidder]=None,
                 eval_bid_size = None, eval_batch_size = None):

        if not n_players:
            n_players = len(agents)

        super().__init__(
            agents=agents,
            n_players=n_players,
            batch_size=batch_size,
            strategy_to_player_closure=strategy_to_player_closure
            )

        self.mechanism = mechanism
        self.eval_bid_size = eval_bid_size
        self.eval_batch_size = eval_batch_size

    def get_reward(self, agent: Bidder, draw_valuations=False, aggregate = True) -> torch.Tensor: #pylint: disable=arguments-differ
        """Returns reward of a single player against the environment.
           Reward is calculated as average utility for each of the batch_size x env_size games
        """

        if not isinstance(agent, Bidder):
            raise ValueError("Agent must be of type Bidder")

        assert agent.batch_size == self.batch_size, \
            "Agent batch_size does not match the environment!"

        player_position = agent.player_position if agent.player_position else 0

        # draw valuations
        if draw_valuations:
            agent.prepare_iteration()
            self.draw_valuations_(exclude = set([player_position]))

        # get agent_bid
        agent_bid = agent.get_action()
        action_length = agent_bid.shape[1]

        if not self.agents or len(self.agents)==1:# Env is empty --> play only with own action against 'nature'
            allocation, payments = self.mechanism.play(
                agent_bid.view(agent.batch_size, 1, action_length)
            )
            utility = agent.get_utility(allocation[:,0,:], payments[:,0])
        else: # at least 2 environment agent --> build bid_profile, then play
            # get bid profile
            bid_profile = torch.zeros(self.batch_size, self.n_players, action_length,
                                      dtype=agent_bid.dtype, device = self.mechanism.device)
            bid_profile[:, player_position, :] = agent_bid

            # Get actions for all players in the environment except the one at player_position
            # which is overwritten by the active agent instead.

            # the counter thing is an ugly af hack: if environment is dynamic, 
            # all player positions will be none. so simply start at 1 for 
            # the first opponent and count up
            # TODO: clean this up 🤷 ¯\_(ツ)_/¯
            counter = 1
            for opponent_pos, opponent_bid in self._generate_agent_actions(exclude = set([player_position])):
                # since auction mechanisms are symmetric, we'll define 'our' agent to have position 0
                if opponent_pos is None:
                    opponent_pos = counter
                bid_profile[:, opponent_pos, :] = opponent_bid
                counter = counter + 1

            allocation, payments = self.mechanism.play(bid_profile)

            # average over batch against this opponent
            utility = agent.get_utility(allocation[:,player_position,:],
                                        payments[:,player_position])

        if aggregate:
                utility = utility.mean()

        return utility


    def get_regret(self, bid_profile: torch.Tensor, agent_position: int, agent_valuation: torch.Tensor, 
                   agent_bid_actual: torch.Tensor, agent_bid_eval: torch.Tensor, compress_dtypes = False):
        #TODO: 1. Implement individual evaluation batch und bid size -> large batch for training, smaller for eval
        #TODO: 2. Implement logging for evaluations ins tensor and for printing
        #TODO: 3. Implement printing plotting of evaluation
        """
        Estimates the potential benefit of deviating from the current energy, as:
            regret(v_i) = Max_(b_i)[ E_(b_(-i))[u(v_i,b_i,b_(-i))] ]
            regret_max = Max_(v_i)[ regret(v_i) ]
            regret_expected = E_(v_i)[ regret(v_i) ]
        The current bidder is always considered with index = 0
        Input:
            bid_profile: (batch_size x n_player x n_items)
            agent_valuation: (batch_size x n_items)
            agent_bid_actual: (batch_size x n_items)
            agent_bid_eval: (bid_size x n_items)
        Output:
            regret_max
            regret_expected
            TODO: Only applicable to independent valuations. Add check. 
            TODO: Only for risk neutral bidders. Add check.
        Useful: To get the memory used by a tensor (in MB): (tensor.element_size() * tensor.nelement())/(1024*1024)
        """

        # TODO: Generalize these dimensions
        batch_size, n_player, n_items = bid_profile.shape
        # Create multidimensional bid tensor if required
        if n_items == 1:
            agent_bid_eval = agent_bid_eval.view(agent_bid_eval.shape[0], 1).to(bid_profile.device)
        elif n_items == 2:
            agent_bid_eval = torch.combinations(agent_bid_eval, with_replacement=True).to(bid_profile.device)
        elif n_items > 2:
            sys.exit("not implemented yet!")
        bid_eval_size, _ = agent_bid_eval.shape

        ## Use smaller dtypes to save memory
        if compress_dtypes:
            bid_profile = bid_profile.type(torch.float16)
            agent_valuation = agent_valuation.type(torch.float16)
            agent_bid_actual = agent_bid_actual.type(torch.float16)
            agent_bid_eval = agent_bid_eval.type(torch.float16)
        bid_profile_origin = bid_profile
        ### Evaluate alternative bids
        ## Merge alternative bids into opponnents bids (bid_no_i)
        bid_profile = self._create_bid_profile(agent_position, agent_bid_eval, bid_profile_origin)
    
        ## Calculate allocation and payments for alternative bids given opponents bids
        allocation, payments = self.mechanism.play(bid_profile)
        a_i = allocation[:,agent_position,:].view(bid_eval_size, batch_size, n_items).type(torch.bool)
        p_i = payments[:,agent_position].view(bid_eval_size, batch_size, 1).sum(2)
        
        del allocation, payments, bid_profile
        torch.cuda.empty_cache()
        # Calculate realized valuations given allocation
        try:
            v_i = agent_valuation.repeat(1,bid_eval_size * batch_size).view(batch_size, bid_eval_size, batch_size, n_items)
            v_i = torch.einsum('hijk,ijk->hijk', v_i, a_i).sum(3)
            ## Calculate utilities
            u_i_alternative = v_i - p_i.repeat(batch_size,1,1)
            # avg per bid
            u_i_alternative = torch.mean(u_i_alternative,2)
            # max per valuations
            u_i_alternative, _ = torch.max(u_i_alternative,1)
        except RuntimeError as err:
            print("Failed computing regret as batch. Trying sequential valuations computation. Decrease dimensions to fix. Error:\n {0}".format(err))
            try:
                # valuations sequential
                u_i_alternative = torch.zeros(batch_size, device = p_i.device)
                for v in range(batch_size):
                    v_i = agent_valuation[v].repeat(1,bid_eval_size * batch_size).view(bid_eval_size, batch_size, n_items)
                    #for bid in agent bid
                    v_i = torch.einsum('ijk,ijk->ijk', v_i, a_i).sum(2)
                    ## Calculate utilities
                    u_i_alternative_v = v_i - p_i
                    # avg per bid
                    u_i_alternative_v = torch.mean(u_i_alternative_v,1)
                    # max per valuations
                    u_i_alternative[v], _ = torch.max(u_i_alternative_v,0)
                    tmp = int(batch_size/100)
                    if v % tmp == 0:
                        print('{} %'.format(v*100/batch_size))
            except RuntimeError as err:
                print("Failed computing regret as batch with sequential valuations. Decrease dimensions to fix. Error:\n {0}".format(err))
                u_i_alternative = torch.ones(batch_size, device = p_i.device) * -9999999
        
        # Clean up storage
        del v_i, u_i_alternative_v
        torch.cuda.empty_cache()
        
        ### Evaluate actual bids
        ## Merge actual bids into opponnents bids (bid_no_i)
        bid_profile = self._create_bid_profile(agent_position, agent_bid_actual, bid_profile_origin)
        
        ## Calculate allocation and payments for actual bids given opponents bids
        allocation, payments = self.mechanism.play(bid_profile)
        a_i = allocation[:,agent_position,:].view(batch_size, batch_size, n_items)
        p_i = payments[:,agent_position].view(batch_size, batch_size, 1).sum(2)

        ## Calculate realized valuations given allocation
        v_i = agent_valuation.view(batch_size,1,n_items).repeat(1, batch_size, 1)
        v_i = torch.einsum('ijk,ijk->ijk', v_i, a_i).sum(2)

        ## Calculate utilities
        u_i_actual = v_i - p_i
        # avg per bid and valuation
        u_i_actual = torch.mean(u_i_actual,1)

        ## average and max regret over all valuations
        regret = u_i_alternative - u_i_actual

        # Explicitaly cleanup TODO:?


        return regret
        
    def _create_bid_profile(self, agent_position: int, player_bids: torch.tensor, original_bid_profile: torch.tensor):
        batch_size, n_player, n_items = original_bid_profile.shape
        bid_eval_size, _ = player_bids.shape
        ## Merge bid_i into opponnents bids (bid_no_i)
        # bids_(-i)
        bid_no_i_left = original_bid_profile[:, [i for i in range(n_player) if i<agent_position], :]
        bid_no_i_right = original_bid_profile[:, [i for i in range(n_player) if i>agent_position], :]
        # bids_i x batch_size
        bid_i = player_bids.repeat(1,batch_size).view(bid_eval_size*batch_size,1,n_items)
        # bid_size x bids_(-i)
        bid_no_i_left = bid_no_i_left.repeat(bid_eval_size, 1, 1)
        bid_no_i_right = bid_no_i_right.repeat(bid_eval_size, 1, 1)
        #TODO: In place combination or splitting and sequential
        return torch.cat([bid_no_i_left,bid_i,bid_no_i_right],1)

    def prepare_iteration(self):
        self.draw_valuations_()

    def draw_valuations_(self, exclude: Set[int] or None = None):
        """
            Draws new valuations for each agent in the environment except the
            excluded set.

            args:
                exclude: A set of player positions to exclude.
                    Used e.g. to generate action profile of all but currently
                    learning player.

            returns/yields:
                nothing

            side effects:
                updates agent valuation states
        """

        if exclude is None:
            exclude = set()

        for agent in (a for a in self.agents if a.player_position not in exclude):
            agent.batch_size = self.batch_size
            if isinstance(agent, Bidder):
                agent.draw_valuations_()
