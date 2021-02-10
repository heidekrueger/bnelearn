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
                 batch_size = 1,
                 strategy_to_player_closure: Callable or None = None,
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
                            draw_valuations=False, aggregate_batch=True,
                            regularize=0, **strat_to_player_kwargs) -> torch.Tensor:
        """
        Returns reward of a given strategy in given environment agent position.

        Args:
            strategy: the strategy to be evaluated
            player_position: the player position at which the agent will be evaluated
            draw_valuation: whether to redraw valuations (default false)
            aggregate_batch: whether to aggregate rewards into a single scalar (True),
                or return batch_size many rewards (one for each sample). Default True
            strat_to_player_kwargs: further arguments needed for agent creation

        """
        if not self._strategy_to_player:
            raise NotImplementedError('This environment has no strategy_to_player closure!')
        # TODO: wouln't be a clone be easier? This always creates a whole
        # bidder object and draws valuations. Alternative: clone bidder and then
        # overwrite its stratrgy
        agent = self._strategy_to_player(strategy=strategy, batch_size=self.batch_size,
                                         player_position=player_position, **strat_to_player_kwargs)
        # TODO: this should rally be in AuctionEnv subclass
        env_agent = self.agents[player_position]
        if hasattr(env_agent, 'valuations'):
            agent.valuations = env_agent.valuations
        if hasattr(env_agent, '_unkown_valuation'):
            agent._unkown_valuation = env_agent._unkown_valuation
        return self.get_reward(agent, draw_valuations=draw_valuations,
                               aggregate=aggregate_batch, regularize=regularize)

    def get_strategy_action_and_reward(self, strategy: Strategy, player_position: int,
                                       draw_valuations=False, **strat_to_player_kwargs) -> torch.Tensor:
        """
        Returns reward of a given strategy in given environment agent position.
        """

        if not self._strategy_to_player:
            raise NotImplementedError('This environment has no strategy_to_player closure!')
        agent = self._strategy_to_player(strategy, batch_size=self.batch_size,
                                         player_position=player_position, **strat_to_player_kwargs)

        # NOTE: Order matters! if draw_valuations, then action must be calculated AFTER reward
        reward = self.get_reward(agent, draw_valuations = draw_valuations, aggregate = False)
        action = agent.get_action()

        return action, reward

    def _generate_agent_actions(
            self,
            exclude: Set[int] or None = None
        ):
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
            yield (agent.player_position, agent.get_action())

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
        """Simulates one batch of the environment and returns the average reward for `agent` as a scalar tensor.
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
        utilities =  agent.get_utility(allocation[:,player_position,:], payments[:,player_position], action_profile)

        return utilities.mean()


class AuctionEnvironment(Environment):
    """
    An environment of agents to play against and evaluate strategies.

    In particular this means:
        - an iterable of sets of -i players that a strategy of a single player can be tested against
        - accept strategy as argument, then play batch_size rounds and return the reward

    Args:
        ... (TODO: document)
        correlation_structure

        strategy_to_bidder_closure: A closure (strategy, batch_size) -> Bidder to
            transform strategies into a Bidder compatible with the environment
    """

    def __init__(
            self,
            mechanism: Mechanism,
            agents: Iterable[Bidder],
            batch_size = 100,
            n_players = None,
            strategy_to_player_closure: Callable[[Strategy], Bidder] = None,
            correlation_groups: List[List[int]] = None,
            correlation_devices: List[CorrelationDevice] = None
        ):

        if not n_players:
            n_players = len(agents)

        super().__init__(
            agents = agents,
            n_players = n_players,
            batch_size = batch_size,
            strategy_to_player_closure = strategy_to_player_closure
        )

        self.mechanism = mechanism

        if not correlation_groups:
            self.correlation_groups = [list(range(n_players))] # all agents in one independent group
            self.correlation_devices = [IndependentValuationDevice()]
        else:
            assert len(correlation_groups) == len(correlation_devices)
            self.correlation_groups = correlation_groups
            self.correlation_devices = correlation_devices

        assert sorted([a for g in self.correlation_groups for a in g]) == list(range(n_players)), \
            "Each agent should be in exactly one correlation group!"

    def get_reward(
            self,
            agent: Bidder,
            draw_valuations: bool = False,
            aggregate: bool = True,
            regularize: float = 0.0
        ) -> torch.Tensor: #pylint: disable=arguments-differ
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
            self.draw_valuations_()

        # get agent_bid
        agent_bid = agent.get_action()
        bid_magnitude = agent_bid.mean()
        action_length = agent_bid.shape[1]

        if not self.agents or len(self.agents)==1:# Env is empty --> play only with own action against 'nature'
            allocation, payments = self.mechanism.play(
                agent_bid.view(agent.batch_size, 1, action_length)
            )
            utility = agent.get_utility(allocation[:,0,:], payments[:,0], agent_bid.view(agent.batch_size, 1, action_length))
        else: # at least 2 environment agent --> build bid_profile, then play
            # get bid profile
            bid_profile = torch.zeros(self.batch_size, self.n_players, action_length,
                                      dtype=agent_bid.dtype, device=self.mechanism.device)
            bid_profile[:, player_position, :] = agent_bid

            # Get actions for all players in the environment except the one at player_position
            # which is overwritten by the active agent instead.

            # the counter thing is an ugly af hack: if environment is dynamic,
            # all player positions will be none. so simply start at 1 for
            # the first opponent and count up

            # ugly af hack: if environment is dynamic, all player positions will be
            # none. simply start at 1 for the first opponent and count up
            # TODO: clean this up 🤷 ¯\_(ツ)_/¯
            counter = 1
            for opponent_pos, opponent_bid in self._generate_agent_actions(exclude=set([player_position])):
                # since auction mechanisms are symmetric, we'll define 'our' agent to have position 0
                if opponent_pos is None:
                    opponent_pos = counter
                bid_profile[:, opponent_pos, :] = opponent_bid
                counter = counter + 1

            allocation, payments = self.mechanism.play(bid_profile)

            # average over batch against this opponent
            utility = agent.get_utility(allocation[:,player_position,:],
                                        payments[:,player_position], bid_profile)

            # regularize
            utility -= regularize * bid_magnitude

        if aggregate:
            utility = utility.mean()

        return utility

    def get_allocation(
            self,
            agent,
            draw_valuations: bool = False,
            aggregate: bool = True,
        ) -> torch.Tensor:
        """Returns allocation of a single player against the environment.
        """

        if not isinstance(agent, Bidder):
            raise ValueError("Agent must be of type Bidder")

        assert agent.batch_size == self.batch_size, \
            "Agent batch_size does not match the environment!"

        player_position = agent.player_position if agent.player_position else 0

        # draw valuations
        if draw_valuations:
            self.draw_valuations_()

        # get agent_bid
        agent_bid = agent.get_action()
        action_length = agent_bid.shape[1]

        if not self.agents or len(self.agents)==1:# Env is empty --> play only with own action against 'nature'
            allocation, _ = self.mechanism.play(
                agent_bid.view(agent.batch_size, 1, action_length)
            )

        else: # at least 2 environment agent --> build bid_profile, then play
            # get bid profile
            bid_profile = torch.zeros(self.batch_size, self.n_players, action_length,
                                      dtype=agent_bid.dtype, device=self.mechanism.device)
            bid_profile[:, player_position, :] = agent_bid

            for opponent_pos, opponent_bid in self._generate_agent_actions(exclude=set([player_position])):
                bid_profile[:, opponent_pos, :] = opponent_bid

            allocation, _ = self.mechanism.play(bid_profile)

        allocation = allocation[:, player_position, :]

        if aggregate:
            # Returns flat tensor with int entries `i` for a allocation of `i`th item
            allocation = torch.einsum(
                'bi,i->bi', allocation,
                torch.arange(1, action_length + 1, device=allocation.device)
            ).view(1, -1)
            allocation = allocation[allocation > 0].to(torch.int8)

        return allocation

    def prepare_iteration(self):
        self.draw_valuations_()

    def draw_valuations_(self, exclude: Set[int] or None = None):
        """
        Draws new valuations for each agent in the environment except the
        excluded set.

        args:
            exclude: (deprecated - setting this variable will return an error)
                A set of player positions to exclude.
                Used e.g. to generate action profile of all but currently
                learning player.

        returns/yields:
            nothing

        side effects:
            updates agent valuation states
        """

        # TODO: remove exclude block if it turns out to be used nowhere.
        if exclude is None:
            exclude = set()

        if exclude:
            raise ValueError('With the introduction of Correlation Devices, excluding agents is no logner supported!')

        # for agent in (a for a in self.agents if a.player_position not in exclude):
        #     agent.batch_size = self.batch_size
        #     if isinstance(agent, Bidder):
        #         agent.draw_valuations_()

        # For each group of correlated agents, draw their correlated valuations
        for group, device in zip(self.correlation_groups, self.correlation_devices):
            common_component, weights = device.get_component_and_weights()
            for i in group:
                self.agents[i].draw_valuations_(common_component, weights)

    def draw_conditionals(
            self,
            player_position: int,
            conditional_observation: torch.Tensor,
            batch_size: int = None
        ) -> dict:
        """
        Draws valuations/observations from all agents conditioned on the observation `cond`
        of the agent at `player_position` from the correlation_devices.

        Returns
            conditionals_dict (dict) with `player_position` as keys and the
                corresponding conditoned valuation `tensors` as dict-values.
        """
        batch_size_0 = conditional_observation.shape[0]
        batch_size_1 = batch_size if batch_size is not None else batch_size_0

        group_idx = [player_position in group for group in self.correlation_groups].index(True)
        cond_device = self.correlation_devices[group_idx]
        conditionals_dict = dict()

        for group, device in zip(self.correlation_groups, self.correlation_devices):

            # draw conditional valuations from all agents in same correlation
            if cond_device == device:
                conditionals_dict.update(
                    device.draw_conditionals(
                        agents = [a for a in self.agents if a.player_position in group],
                        player_position = player_position,
                        conditional_observation = conditional_observation,
                        batch_size = batch_size_1
                    )
                )

            # draw independent valuations from all agents in other correlations
            else:
                common_component, weights = device.get_component_and_weights()
                for player_position in group:
                    agent = [a for a in self.agents if a.player_position == player_position][0]
                    conditionals_dict[player_position] = agent.draw_valuations_(common_component, weights) \
                            [:batch_size_1, :].repeat(batch_size_0, 1)

        return conditionals_dict
