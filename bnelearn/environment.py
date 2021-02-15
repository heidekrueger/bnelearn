# -*- coding: utf-8 -*-
"""
This module contains environments - a collection of players and
possibly state histories that is used to control game playing and
implements reward allocation to agents.
"""

from abc import ABC, abstractmethod
from typing import Callable, Set, List, Iterable

import torch

from bnelearn.bidder import Bidder, MatrixGamePlayer, Player
from bnelearn.mechanism import MatrixGame, Mechanism
from bnelearn.strategy import Strategy
from bnelearn.correlation_device import CorrelationDevice, IndependentValuationDevice
from bnelearn.state_device import state_device

class Environment(ABC):
    """Environment

    An Environment object 'manages' a repeated game, i.e. manages the current players
    and their models, collects players' actions, distributes rewards,
    runs the game itself and allows 'simulations' as in 'how would a mutated player
    do in the current setting'?
    """
    def __init__(self,
                 agents: Iterable,
                 n_players = 2,
                 strategy_to_player_closure: Callable or None = None,
                 **kwargs #pylint: disable=unused-argument
                 ):
        assert isinstance(agents, Iterable), "iterable of agents must be supplied"

        self._strategy_to_player = strategy_to_player_closure
        self.n_players = n_players

        # transform agents into players, if specified as Strategies:
        agents = [
            self._strategy_to_player(agent, player_position) if isinstance(agent, Strategy) else agent
            for player_position, agent in enumerate(agents)
        ]
        self.agents: Iterable[Player] = agents
        self.__len__ = self.agents.__len__

    def is_empty(self):
        """True if no agents in the environment"""
        return len(self) == 0


class MatrixGameEnvironment(Environment):
    """ An environment for matrix games.

        Important features of matrix games for implementation:
        - not necessarily symmetric, i.e. each player has a fixed position
        - agents strategies do not take any input, the actions only depend
           on the game itself (no Bayesian Game)

    not adapted to the new refactoring !!!
    """

    def __init__(self,
                 game: MatrixGame,
                 agents,
                 n_players=2,
                 batch_size=1,
                 strategy_to_player_closure=None,
                 **kwargs):
        self.batch_size = batch_size

        super().__init__(agents, n_players=n_players,
                         strategy_to_player_closure=strategy_to_player_closure)
        self.game = game

    def get_reward(self, agent, **kwargs) -> torch.tensor: #pylint: disable=arguments-differ
        """Simulates one batch of the environment and returns the average reward for `agent` as a scalar tensor.
        """
        pass

        # if isinstance(agent, Strategy):
        #     agent: MatrixGamePlayer = self._strategy_to_player(
        #         agent,
        #         batch_size=self.batch_size,
        #         **kwargs
        #         )
        # player_position = agent.player_position

        # action_profile = torch.zeros(self.batch_size, self.game.n_players,
        #                              dtype=torch.long, device=agent.device)

        # action_profile[:, player_position] = agent.get_action().view(self.batch_size)

        # for opponent_action in self._generate_agent_actions(exclude = set([player_position])):
        #     position, action = opponent_action
        #     action_profile[:, position] = action.view(self.batch_size)

        # allocation, payments = self.game.play(action_profile.view(self.batch_size, self.n_players, -1))
        # utilities =  agent.get_utility(allocation[:,player_position,:], payments[:,player_position])

        # return utilities.mean()
    
    def get_strategy_reward(self, strategy: Strategy, player_position: int,**strat_to_player_kwargs) -> torch.Tensor:
        """
        Returns reward of a given strategy in given environment agent position.

        Args:
            strategy: the strategy to be evaluated
            player_position: the player position at which the agent will be evaluated
            strat_to_player_kwargs: further arguments needed for agent creation

        """
        pass
        # if not self._strategy_to_player:
        #     raise NotImplementedError('This environment has no strategy_to_player closure!')
        # agent = self._strategy_to_player(strategy,player_position=player_position, **strat_to_player_kwargs)
        # # TODO: this should rally be in AuctionEnv subclass
        # env_agent = self.agents[player_position]
        # return self.get_reward(agent, draw_valuations=draw_valuations, aggregate=aggregate_batch)

       
    def get_strategy_action_and_reward(self, strategy: Strategy, player_position: int,
                                       draw_valuations=False, **strat_to_player_kwargs) -> torch.Tensor:
        """
        Returns reward of a given strategy in given environment agent position.
        """
        pass

        # if not self._strategy_to_player:
        #     raise NotImplementedError('This environment has no strategy_to_player closure!')
        # agent = self._strategy_to_player(strategy,player_position=player_position, **strat_to_player_kwargs)

        # # NOTE: Order matters! if draw_valuations, then action must be calculated AFTER reward
        # reward = self.get_reward(agent, aggregate = False)
        # action = agent.get_action()

        # return action, reward



class AuctionEnvironment(Environment):
    """
    An environment of agents to play against and evaluate strategies.

    In particular this means:
        - an iterable of sets of -i players that a strategy of a single player can be tested against
        - accept strategy as argument, then play batch_size rounds and return the reward

    Args:
        ... (TODO: document)
        correlation_structure

        strategy_to_bidder_closure: A closure (strategy) -> Bidder to
            transform strategies into a Bidder compatible with the environment
    """

    def __init__(
            self,
            mechanism: Mechanism,
            agents: Iterable[Bidder],
            n_players = None,
            strategy_to_player_closure: Callable[[Strategy], Bidder] = None,
            correlation_groups: List[List[int]] = None,
            correlation_devices: List[CorrelationDevice] = None,
            rule : str= "pseudorandom",
            antithetic : bool = False,
            inplace_sampling : bool = False,
            scramble : bool = True
        ):

        if not n_players:
            n_players = len(agents)
        self.rule = rule
        self.antithetic = antithetic
        self.inplace_sampling = inplace_sampling
        self.scramble = scramble

        super().__init__(
            agents = agents,
            n_players = n_players,
            strategy_to_player_closure = strategy_to_player_closure
        )

        self.mechanism = mechanism
        self.n_items = self.agents[0].n_items
        self.device = self.agents[0].device

        if not correlation_groups:
            self.correlation_groups = [list(range(n_players))] # all agents in one independent group
            self.correlation_devices = [IndependentValuationDevice(n_items= self.n_items,correlation_group=self.correlation_groups[0], rule = self.rule,antithetic = self.antithetic, inplace_sampling=self.inplace_sampling, device = self.device, scramble = self.scramble)]
        else:
            assert len(correlation_groups) == len(correlation_devices)
            self.correlation_groups = correlation_groups
            self.correlation_devices = correlation_devices

        assert sorted([a for g in self.correlation_groups for a in g]) == list(range(n_players)), \
            "Each agent should be in exactly one correlation group!"
        self.state_device = state_device(self.correlation_devices)

    def get_bid_profile(self, agents : Iterable[Bidder],valuations): 
        """ gets bid_profile from valuations

        Args:
            valuations (torch.Tensor)
            agents (Iterable[Bidder])

        Returns:
            [torch.Tensor]
        """
        bids = torch.zeros_like(valuations, device= valuations.device)
        for i in range(len(self.agents)):
            bids[:,i,:] = agents[i].get_action(valuations[:,i,:])
        return bids
    def get_reward_and_action(
            self,
            agent: Bidder,
            batch_size:int
        ):
        """
        returns utility, utility.mean() and bid_profile of player
        """

        if not isinstance(agent, Bidder):
            raise ValueError("Agent must be of type Bidder")
        player_position = agent.player_position

        output = self.state_device.draw_state(agents =self.agents,batch_size= batch_size)
        unknown_valuations = output["_unkown_valuation"]
        valuations = output["valuations"]
        if unknown_valuations is None: 
            unknown_valuations = valuations

        bid_profile = self.get_bid_profile(self.agents,valuations)
        allocation, payments = self.mechanism.play(bid_profile)
        utility = agent.get_utility(allocation[:,player_position,:],
                                    payments[:,player_position], unknown_valuations[:,player_position,:])


        return utility, utility.mean(), bid_profile[:,player_position,:]

    def get_strategy_reward(
            self,
            batch_size:int,
            strategy:Strategy,
            **strat_to_player_kwargs):
        """
        get reward for agent at position player_position with the specified strategy 
        """
        if not self._strategy_to_player:
            raise NotImplementedError('This environment has no strategy_to_player closure!')
        agent = self._strategy_to_player(strategy, **strat_to_player_kwargs)
        # TODO: this should rally be in AuctionEnv subclass
        copy_agents = self.agents
        copy_agents[strat_to_player_kwargs["player_position"]] = agent

        output = self.state_device.draw_state(agents =copy_agents,batch_size= batch_size)
        unknown_valuations = output["_unkown_valuation"]
        valuations = output["valuations"]
        if unknown_valuations is None : 
            unknown_valuations = valuations

        bid_profile = self.get_bid_profile(copy_agents,valuations)
        allocation, payments = self.mechanism.play(bid_profile)
        utility = agent.get_utility(allocation[:,strat_to_player_kwargs["player_position"],:],
                                    payments[:,strat_to_player_kwargs["player_position"]], unknown_valuations[:,strat_to_player_kwargs["player_position"],:])
        return utility.mean()

    def draw_conditionals(self, player_position: int, conditional_observation: torch.Tensor, batch_size: int = None):
        """
        Draws valuations/observations from all agents conditioned on the observation `cond`
        of the agent at `player_position` from the correlation_devices.
        """
        return self.state_device.draw_conditional(self.agents,player_position, conditional_observation, batch_size)