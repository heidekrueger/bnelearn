# -*- coding: utf-8 -*-
"""
This module contains environments - a collection of players and
possibly state histories that is used to control game playing and
implements reward allocation to agents.
"""

from abc import ABC, abstractmethod
from typing import Callable, Set, Iterable, Tuple

import torch

from bnelearn.bidder import Bidder, MatrixGamePlayer, Player
from bnelearn.mechanism import MatrixGame, Mechanism
from bnelearn.strategy import Strategy, NeuralNetStrategy
from bnelearn.sampler import ValuationObservationSampler

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
        self.agents: Iterable[Player] = [
            self._strategy_to_player(agent, batch_size, player_position) if isinstance(agent, Strategy) else agent
            for player_position, agent in enumerate(agents)
        ]
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
                            redraw_valuations=False, aggregate_batch=True,
                            regularize: float=0, deterministic: bool=False,
                            **strat_to_player_kwargs) -> torch.Tensor:
        """
        Returns reward of a given strategy in given environment agent position.

        Args:
            strategy: the strategy to be evaluated
            player_position: the player position at which the agent will be evaluated
            redraw_valuation: whether to redraw valuations (default false)
            aggregate_batch: whether to aggregate rewards into a single scalar (True),
                or return batch_size many rewards (one for each sample). Default True
            strat_to_player_kwargs: further arguments needed for agent creation
            regularize: paramter that penalizes high action values (e.g. if we
                get the same utility with different actions, we prefer the lower
                one). Default value of zero corresponds to no regularization.
        """
        if not self._strategy_to_player:
            raise NotImplementedError('This environment has no strategy_to_player closure!')

        agent = self._strategy_to_player(strategy=strategy, batch_size=self.batch_size,
                                         player_position=player_position, **strat_to_player_kwargs)
        # TODO: this should rally be in AuctionEnv subclass
        return self.get_reward(agent, redraw_valuations=redraw_valuations,
                               aggregate=aggregate_batch, regularize=regularize,
                               deterministic=deterministic)

    def get_strategy_action_and_reward(self, strategy: Strategy, player_position: int,
                                       redraw_valuations=False, **strat_to_player_kwargs) -> torch.Tensor:
        """
        Returns reward of a given strategy in given environment agent position.
        """

        if not self._strategy_to_player:
            raise NotImplementedError('This environment has no strategy_to_player closure!')
        agent = self._strategy_to_player(strategy, batch_size=self.batch_size,
                                         player_position=player_position, **strat_to_player_kwargs)

        # NOTE: Order matters! if redraw_valuations, then action must be calculated AFTER reward
        reward = self.get_reward(agent, redraw_valuations = redraw_valuations, aggregate = False)
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
        utilities = agent.get_utility(allocation[:,player_position,:], payments[:,player_position])

        return utilities.mean()


class AuctionEnvironment(Environment):
    """
    An environment of agents to play against and evaluate strategies.

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
            valuation_observation_sampler: ValuationObservationSampler,
            batch_size = 100,
            n_players = None,
            strategy_to_player_closure: Callable[[Strategy], Bidder] = None,
            redraw_every_iteration: bool = False
        ):

        assert isinstance(valuation_observation_sampler, ValuationObservationSampler)

        if not n_players:
            n_players = len(agents)

        super().__init__(
            agents = agents,
            n_players = n_players,
            batch_size = batch_size,
            strategy_to_player_closure = strategy_to_player_closure
        )

        self.mechanism = mechanism
        self.sampler = valuation_observation_sampler

        self._redraw_every_iteration = redraw_every_iteration
        # draw initial observations and iterations
        self._observations: torch.Tensor = None
        self._valuations: torch.Tensor = None
        self.draw_valuations()

    def _generate_agent_actions(self, exclude: Set[int] or None = None):
        """
        Generator function yielding batches of bids for each environment agent
        that is not excluded. Overwrites because in auction_environment, this needs
        access to observations

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

            # Set agent to eval mode -> ignore its `log_prob`s for REINFORCE
            if isinstance(agent.strategy, NeuralNetStrategy):
                agent.strategy.train(False)

            yield (agent.player_position,
                   agent.get_action(self._observations[..., agent.player_position, :]))

    def get_reward(
            self,
            agent: Bidder,
            redraw_valuations: bool = False,
            aggregate: bool = True,
            regularize: float = 0.0,
            return_allocation: bool = False,
            deterministic: bool = False,
        ) -> torch.Tensor or Tuple[torch.Tensor, torch.Tensor]: #pylint: disable=arguments-differ
        """Returns reward of a single player against the environment, and optionally additionally the allocation of that player.
           Reward is calculated as average utility for each of the batch_size x env_size games
        """

        if not isinstance(agent, Bidder):
            raise ValueError("Agent must be of type Bidder")

        assert agent.batch_size == self.batch_size, \
            "Agent batch_size does not match the environment!"

        player_position = agent.player_position if agent.player_position else 0

        # draw valuations if desired
        if redraw_valuations:
            self.draw_valuations()

        agent_observation = self._observations[:, player_position, :] 
        agent_valuation = self._valuations[:, player_position, :]

        # get agent_bid
        agent_bid = agent.get_action(agent_observation, deterministic=deterministic)
        action_length = agent_bid.shape[-1]

        if not self.agents or len(self.agents)==1:# Env is empty --> play only with own action against 'nature'
            allocations, payments = self.mechanism.play(
                agent_bid.view(agent.batch_size, 1, action_length)
            )
        else: # at least 2 environment agent --> build bid_profile, then play
            # get bid profile
            bid_profile = torch.empty(self.batch_size, self.n_players, action_length,
                                      dtype=agent_bid.dtype, device=self.mechanism.device)
            bid_profile[..., player_position, :] = agent_bid

            # Get actions for all players in the environment except the one at player_position
            # which is overwritten by the active agent instead.

            # ugly af hack: if environment is dynamic, all player positions will be
            # none. simply start at 1 for the first opponent and count up
            # TODO: clean this up 🤷 ¯\_(ツ)_/¯
            counter = 1
            for opponent_pos, opponent_bid in self._generate_agent_actions(exclude=set([player_position])):
                # since auction mechanisms are symmetric, we'll define 'our' agent to have position 0
                if opponent_pos is None:
                    opponent_pos = counter
                bid_profile[:, opponent_pos, :] = opponent_bid.detach()
                counter = counter + 1

            allocations, payments = self.mechanism.play(bid_profile)

        agent_allocation = allocations[..., player_position, :]
        agent_payment = payments[..., player_position]

        # average over batch against this opponent
        agent_utility = agent.get_utility(agent_allocation, agent_payment, agent_valuation)

        # regularize
        agent_utility -= regularize * agent_bid.mean()

        if aggregate:
            agent_utility = agent_utility.mean()

            if return_allocation:
                # Returns flat tensor with int entries `i` for an allocation of `i`th item
                agent_allocation = torch.einsum(
                    'bi,i->bi', agent_allocation,
                    torch.arange(1, action_length + 1, device=agent_allocation.device)
                ).view(1, -1)
                agent_allocation = agent_allocation[agent_allocation > 0].to(torch.int8)

        return agent_utility if not return_allocation else (agent_utility, agent_allocation)

    def get_allocation(
            self,
            agent,
            redraw_valuations: bool = False,
            aggregate: bool = True,
        ) -> torch.Tensor:
        """Returns allocation of a single player against the environment.
        """
        return self.get_reward(
            agent, redraw_valuations, aggregate, return_allocation=True
            )[1]

    def get_revenue(self, redraw_valuations: bool = False) -> float:
        """Returns the average seller revenue over a batch.

        Args:
            redraw_valuations (bool): whether or not to redraw the valuations of
                the agents.

        Returns:
            revenue (float): average of seller revenue over a batch of games.

        """
        if redraw_valuations:
            self.draw_valuations()

        action_length = self.agents[0].bid_size

        bid_profile = torch.zeros(self.batch_size, self.n_players, action_length,
                                  device=self.mechanism.device)
        for pos, bid in self._generate_agent_actions():  # pylint: disable=protected-access
            bid_profile[:, pos, :] = bid
        _, payments = self.mechanism.play(bid_profile)

        return payments.sum(axis=1).float().mean()

    def get_efficiency(self, redraw_valuations: bool = False) -> float:
        """Average percentage that the actual welfare reaches of the maximal
        possible welfare over a batch.

        Args:
            redraw_valuations (:bool:) whether or not to redraw the valuations of
                the agents.

        Returns:
            efficiency (:float:) Percentage that the actual welfare reaches of
                the maximale possible welfare. Averaged over batch.

        """
        batch_size = min(self.sampler.default_batch_size, 2 ** 13)

        if redraw_valuations:
            self.draw_valuations()

        # pylint: disable=protected-access
        valuations = self._valuations[:batch_size, :, :]

        action_length = self.agents[0].bid_size

        # Calculate actual welfare under the current strategies
        bid_profile = torch.zeros(batch_size, self.n_players, action_length,
                                  device=self.mechanism.device)
        for pos, bid in self._generate_agent_actions():  # pylint: disable=protected-access
            bid_profile[:, pos, :] = bid[:batch_size, ...]
        actual_allocations, _ = self.mechanism.play(bid_profile)
        actual_welfare = torch.zeros(batch_size, device=self.mechanism.device)
        for a in self.agents:
            actual_welfare += a.get_welfare(
                actual_allocations[:batch_size, a.player_position],
                valuations[..., a.player_position, :]
            )

        # Calculate counterfactual welfare under truthful strategies
        maximum_allocations, _ = self.mechanism.play(valuations)
        maximum_welfare = torch.zeros_like(actual_welfare)
        for a in self.agents:
            maximum_welfare += a.get_welfare(
                maximum_allocations[:batch_size, a.player_position],
                valuations[..., a.player_position, :]
            )

        efficiency = (actual_welfare / maximum_welfare).mean().float()
        return efficiency

    def prepare_iteration(self):
        if self._redraw_every_iteration:
            self.draw_valuations()

    def draw_valuations(self):
        """
        Draws a new valuation and observation profile

        returns/yields:
            nothing

        side effects:
            updates agent's valuations and observation states
        """

        self._valuations, self._observations = \
            self.sampler.draw_profiles(batch_sizes=self.batch_size)

    def draw_conditionals(
            self,
            conditioned_player: int,
            conditioned_observation: torch.Tensor,
            inner_batch_size: int = None
        ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Draws a conditional valuation / observation profile based on a (vector of)
        fixed observations for one player.

        Total batch size will be conditioned_observation.shape[0] x inner_batch_size
        """

        cv, co = self.sampler.draw_conditional_profiles(
            conditioned_player, conditioned_observation,
            inner_batch_size
        )

        return cv, co
