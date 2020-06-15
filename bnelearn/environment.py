# -*- coding: utf-8 -*-
"""
This module contains environments - a collection of players and
possibly state histories that is used to control game playing and
implements reward allocation to agents.
"""

from abc import ABC, abstractmethod
from collections.abc import Iterable
from typing import Callable, Set
from copy import deepcopy
from typing import Tuple, Set, Type, Callable
import torch
import numpy as np
from itertools import product
import math

from bnelearn.bidder import Bidder, MatrixGamePlayer, Player
from bnelearn.mechanism import MatrixGame, Mechanism
from bnelearn.strategy import Strategy
from bnelearn.mechanism import FPSBSplitAwardAuction

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
                            draw_valuations=False, aggregate_batch = True,
                            **strat_to_player_kwargs) -> torch.Tensor:
        """
        Returns reward of a given strategy in given environment agent position.
        """
        if not self._strategy_to_player:
            raise NotImplementedError('This environment has no strategy_to_player closure!')
        agent = self._strategy_to_player(strategy, batch_size=self.batch_size,
                                         player_position=player_position, **strat_to_player_kwargs)
        return self.get_reward(agent, draw_valuations=draw_valuations, aggregate=aggregate_batch)

    def get_strategy_action_and_reward(self, strategy: Strategy, player_position: int,
                                       draw_valuations=False, **strat_to_player_kwargs) -> torch.Tensor:
        """
        Returns reward of a given strategy in given environment agent position.
        """
        if not self._strategy_to_player:
            raise NotImplementedError('This environment has no strategy_to_player closure!')
        agent = self._strategy_to_player(strategy, batch_size=self.batch_size,
                                         player_position=player_position, **strat_to_player_kwargs)
        action = agent.get_action()
        return action, self.get_reward(agent, draw_valuations = draw_valuations, aggregate = False)

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

    def __init__(
            self,
            mechanism: Mechanism,
            agents: Iterable,
            batch_size = 100,
            n_players = None,
            strategy_to_player_closure: Callable[[Strategy], Bidder] = None
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

    def get_reward(
            self,
            agent: Bidder,
            draw_valuations = False,
            aggregate = True
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
                                        payments[:,player_position])

        if aggregate:
            utility = utility.mean()

        return utility

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

    def get_welfare_max(self):
        """ Returns welfare-maximising allocation
        """
        n_bundles = self.agents[0].n_items
        n_items = int(math.log(n_bundles + 1, 2))
        n_players = self.n_players

        # collect all bundles
        bundles = []
        for r in range(n_items):
            for c in torch.combinations(torch.arange(n_items), r+1).tolist():
                bundles.append(c)
        item_prod = torch.tensor(list(product(list(range(n_players)), repeat=n_items)))

        # allocate to highest valuation
        def get_bundle_idx(player_allo):
            """
            Based on a boolean tensor of length n_items returns the idx of correspunding bundle
            """
            won_items = torch.arange(n_items)[player_allo].tolist()
            bundle_idx = bundles.index(won_items)
            return bundle_idx

        def get_allo(item_allo):
            """
            Based on an item allocations returns the correspunding bundle allocations
            """
            a = torch.zeros((n_players, 2**n_items - 1), dtype=int)
            for player_position in range(n_players):
                player_allo = item_allo == player_position
                if player_allo.any():
                    bundle_idx = get_bundle_idx(player_allo)
                    a[player_position, bundle_idx] = 1
            return a

        possible_allocations = torch.zeros(
            (n_players**n_items, n_players, n_bundles),
            dtype=int, device=self.mechanism.device
        )
        for i, p in enumerate(item_prod):
            possible_allocations[i, :, :] = get_allo(p)

        # brute force: all combinations for each batch
        allocations = possible_allocations.repeat(self.batch_size, 1, 1, 1).view(
            self.batch_size, n_players**n_items, n_players, n_bundles
        ) # shape (batch_size, possible_combinations, n_players, n_bundles)

        # calculate welfare
        welfares = torch.zeros_like(allocations[:, :, :, 0], dtype=torch.float32)
        for i, agent in enumerate(self.agents):
            agent_valuations = agent.valuations.repeat(1, 1, n_players**n_items).view(
                self.batch_size, n_players**n_items, n_bundles)
            welfares[:, :, i] = agent.get_welfare(
                allocations[:, :, i, :],
                agent_valuations
            )
        max_welfares, _ = welfares.sum(axis=2).max(axis=1)

        return max_welfares

    def get_PoA(self, allocations):
        """
        Returns (Bayesian) Price of Anarchy
        """

        max_welfares = self.get_welfare_max().mean()

        actual_welfares = torch.zeros(
            self.batch_size, self.n_players,
            device=self.mechanism.device
        )
        for i, a in enumerate(self.agents):
            actual_welfares[:, i] = a.get_welfare(allocations[:, i, :])
        actual_welfares = actual_welfares.sum(axis=1).mean()

        return torch.div(max_welfares, actual_welfares)
