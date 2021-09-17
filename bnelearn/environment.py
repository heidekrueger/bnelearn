# -*- coding: utf-8 -*-
"""
This module contains environments - a collection of players and
possibly state histories that is used to control game playing and
implements reward allocation to agents.
"""

from abc import ABC, abstractmethod
from typing import Callable, Set, Iterable, List, Tuple, Dict
from copy import copy
import inspect

import torch

from bnelearn.bidder import Bidder, MatrixGamePlayer, Player
from bnelearn.mechanism import MatrixGame, Mechanism, DoubleAuctionMechanism, kDoubleAuction
from bnelearn.strategy import Strategy, TruthfulStrategy
from bnelearn.sampler import ValuationObservationSampler
from bnelearn.util.metrics import ex_interim_utility, ex_interim_util_loss
from bnelearn.util.tensor_util import all_combinations


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
                            regularize: float=0,
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
                get the same utility with different actions, we prefer the loweer
                one). Default value of zero corresponds to no regularization.

        """
        if not self._strategy_to_player:
            raise NotImplementedError('This environment has no strategy_to_player closure!')

        agent = self._strategy_to_player(strategy=strategy, batch_size=self.batch_size,
                                         player_position=player_position, **strat_to_player_kwargs)
        # TODO: this should rally be in AuctionEnv subclass
        return self.get_reward(agent, redraw_valuations=redraw_valuations,
                               aggregate=aggregate_batch, regularize=regularize)

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
            yield (agent.player_position,
                   agent.get_action(self._observations[:, agent.player_position, :]))

    def get_reward(
            self,
            agent: Bidder,
            redraw_valuations: bool = False,
            aggregate: bool = True,
            regularize: float = 0.0,
            return_allocation: bool = False
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
        agent_bid = agent.get_action(agent_observation)
        action_length = agent_bid.shape[1]

        if not self.agents or len(self.agents)==1:# Env is empty --> play only with own action against 'nature'
            allocations, payments = self.mechanism.play(
                agent_bid.view(agent.batch_size, 1, action_length)
            )
        else: # at least 2 environment agent --> build bid_profile, then play
            # get bid profile
            bid_profile = torch.empty(self.batch_size, self.n_players, action_length,
                                      dtype=agent_bid.dtype, device=self.mechanism.device)
            bid_profile[:, player_position, :] = agent_bid

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
                bid_profile[:, opponent_pos, :] = opponent_bid
                counter = counter + 1

            allocations, payments = self.mechanism.play(bid_profile)

        agent_allocation = allocations[:, player_position, :]
        agent_payment = payments[:,player_position]

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


    # Properties of the market ################################################

    def get_revenue(self, redraw_valuations: bool=False) -> float:
        """Returns the average seller revenue over a batch.

        Args:
            redraw_valuations (:bool:) whether or not to redraw the valuations
                of the agents.

        Returns:
            revenue (:float:) average of seller revenue over a batch of games.
        """
        if redraw_valuations:
            self.draw_valuations()

        action_length = self.agents[0].bid_size

        bid_profile = torch.zeros(self.batch_size, self.n_players, action_length,
                                  device=self.mechanism.device)
        for pos, bid in self._generate_agent_actions():
            bid_profile[:, pos, :] = bid
        _, payments = self.mechanism.play(bid_profile)

        return payments.sum(axis=1).float().mean()

    def get_efficiency(self, redraw_valuations: bool=False,
                       batch_size: int=2**8) -> float:
        """Average percentage that the actual welfare reaches of the maximal
        possible welfare over a batch.

        Args:
            redraw_valuations (:bool:) whether or not to redraw the valuations
                of the agents.
            batch_size (:int:) maximal batch size for efficiency calculation.

        Returns:
            efficiency (:float:) percentage that the actual welfare reaches of
                the maximale possible welfare. Averaged over batch.
        """
        batch_size = min(self.batch_size, batch_size)

        if redraw_valuations:
            self.draw_valuations()

        # Efficiency of two-sided markets (double-acutions)
        if DoubleAuctionMechanism in inspect.getmro(type(self.mechanism)):
            """Def 1 (M&S). A bilateral bargaining mechanism is ex post
            efficient iff the buyer gets the object whenever her valuation
            is higher, and the seller keeps the object whenever her
            valuation is higher.

            Def 2. A bilateral bargaining mechanism is ex post efficient
            iff the probability of trade is 1 if the buyer reports a higher
            value than the seller and 0 otherwise.

            Note:
                Should be static and independent of strategies.
            """
            # let each valuation of one agent compete against all vluationns
            # of the opponents
            truthful_bid_profile = all_combinations(self._observations[:batch_size, ...])
            # (batch_size**n_agents, n_agents, 1)

            # TODO: Do we really need all combinations when we flatten them anyway?
            #       (And don't take a mean over the current agent)
            actual_allocations, _ = self.mechanism.play(truthful_bid_profile)

            # This mechanism's allocations can be used as baseline b/c they're efficient
            efficient_allocations, _ = \
                kDoubleAuction(self.mechanism.n_buyers, self.mechanism.n_sellers, 0.5) \
                .play(truthful_bid_profile)

            # ``allocated to buyer'' == ``buyer reported higher value''
            allocated_to_highest_reported_value = \
                (actual_allocations == efficient_allocations).all(dim=2).all(dim=1)

            efficiency = allocated_to_highest_reported_value.float().mean()

        # Efficiency of single-sided markets
        else:

            # pylint: disable=protected-access
            valuations = self._valuations[:batch_size, :, :]

            action_length = self.agents[0].bid_size

            # Calculate actual welfare under the current strategies
            bid_profile = torch.zeros(batch_size, self.n_players, action_length,
                                      device=self.mechanism.device)
            for pos, bid in self._generate_agent_actions():
                bid_profile[:, pos, :] = bid[:batch_size, ...]
            actual_allocations, _ = self.mechanism.play(bid_profile)

            actual_welfare = torch.zeros(batch_size, device=self.mechanism.device)
            for a in self.agents:
                actual_welfare += a.get_welfare(
                    actual_allocations[:batch_size, a.player_position],
                    valuations[..., a.player_position, :]
                )

            # Calculate counterfactual welfare under truthful strategies
            # Note: This assumes max. welfare is reached under truthful strategies
            maximum_allocations, _ = self.mechanism.play(valuations)
            maximum_welfare = torch.zeros_like(actual_welfare)
            for a in self.agents:
                maximum_welfare += a.get_welfare(
                    maximum_allocations[:batch_size, a.player_position],
                    valuations[..., a.player_position, :]
                )

            efficiency = actual_welfare / maximum_welfare
            efficiency[maximum_welfare == 0] = 1  # full eff. when no welfare gain was possible
            efficiency = efficiency.mean()

        return efficiency

    def get_budget_balance(self, redraw_valuations: bool=False,
                           batch_size: int=2**8) -> List[float]:
        """Calculate deviation from a budget balnced market.

        Definition: A market is budget balanced when payments from buyers and
        sellers add up to zero. This may not hold for two-sided markets.

        Measurement: Measure minimal and maximal difference from zero when
        adding up all payments, should be zero under BB.

        Args:
            redraw_valuations (:bool:) whether or not to redraw the valuations
                of the agents.
            batch_size (:int:) maximal batch size for BB calculation.

        Returns:
            budget_balance (:list:) maximal budget superplus and maximal budget
                defictit over considered batch.

        Note:
            Should be static and independent of strategies.
        """

        # Single sided auctions are budget balanced by definition
        if not DoubleAuctionMechanism in inspect.getmro(type(self.mechanism)):
            return 0, 0

        batch_size = min(batch_size, self.batch_size)

        if redraw_valuations:
            self.draw_valuations()

        # Calculate all combinations of bids
        bid_profile = all_combinations(self._observations[:batch_size, ...])
        _, payments = self.mechanism.play(bid_profile)

        # Differentiate sellers' and buyers' payments
        payments_buyers = payments[..., :self.mechanism.n_buyers].sum(axis=-1)
        payments_sellers = payments[..., self.mechanism.n_buyers:].sum(axis=-1)
        budget = (payments_buyers - payments_sellers)

        return [(-budget.min()).relu(), budget.max().relu()]

    def get_individual_rationality(self, redraw_valuations: bool=False,
                                   batch_size: int=2**8) -> List[float]:
        """Calculate individual rationality.

        Definition: IR requires that each agent has non-negative expected
        utility after they know their own valuation, but before they learn the
        other's valuation (interim).

        Measurement: Check that minimal utility over prior is non-negative.

        Args:
            redraw_valuations (:bool:) whether or not to redraw the valuations
                of the agents.
            batch_size (:int:) maximal batch size for IR calculation.

        Returns:
            individual_rationality (:list:) of minimal utility over prior for
                each agent.

        Note:
            Should be static and independent of strategies.
        """
        batch_size = min(batch_size, self.batch_size)

        if redraw_valuations:
            self.draw_valuations()

        actual_strategies = [copy(a.strategy) for a in self.agents]
        for a in self.agents:  # all agents truthfull
            a.strategy = TruthfulStrategy()

        individual_rationality = [
            - ex_interim_utility(
                self, player_position=a.player_position,
                agent_observations=self._observations
                    [:batch_size, a.player_position, :],
                agent_actions=a.get_action(self._observations
                    [:batch_size, a.player_position, :]),
                opponent_batch_size=batch_size,
                device=self.mechanism.device
            ).min()
            for a in self.agents
        ]

        for i, a in enumerate(self.agents):  # restore strategies
            a.strategy = actual_strategies[i]

        return individual_rationality

    def get_incentive_compatibility(self, redraw_valuations: bool=False,
                                    batch_size: int=2**8, grid_size: int=2**8) -> List[float]:
        r"""Measure if mechansim is incentive compatible.

        Definition of incentive compatiblity:
        * Informal: A mechanism is (Bayesian) incentive compatible if honest
        reporting forms an BNE. E.g., in an IC mechanism, each individual can
        maximize her expected utility by reporting truthful, given that the
        other is expected to report truthful.
        * Formal: Iff for all $v$ and $v'$: $u(v, b=v) \geq u(v, b=v')$.

        Measurement: Set all agents to truthful and measure the utility loss on
        a grid over the current player's valuations. Positive values mean that
        we found actions different from truthful that lead to higher utility.

        Ususally, IC can be assumed b/c for any BNE of any bargaining, there
        is an equivalent IC direct mechanism that yields the same outocmes
        (revelation principle).

        Args:
            redraw_valuations (:bool:) whether or not to redraw the valuations
                of the agents.
            batch_size (:int:) maximal batch size for IR calculation.
            grid_size (:int:) number of alternative actions to consider.

        Returns:
            incentive_compatibility (:list:) of average utility loss over batch
                when reporting truthfully for each agent.

        Note:
            Should be static and independent of strategies.
        """
        batch_size = min(batch_size, self.batch_size)

        if redraw_valuations:
            self.draw_valuations()

        actual_strategies = [copy(a.strategy) for a in self.agents]
        for a in self.agents:  # all agents truthfull
            a.strategy = TruthfulStrategy()

        with torch.no_grad():  # don't need any gradient information here
            utility_loss, _ = zip(*[
                ex_interim_util_loss(
                    env=self, player_position=a.player_position,
                    agent_observations=self._observations[
                        :batch_size, a.player_position, :],
                    grid_size=grid_size
                )
                for a in self.agents
            ])

        ex_ante_util_loss = [l.mean() for l in utility_loss]

        for i, a in enumerate(self.agents):  # restore strategies
            a.strategy = actual_strategies[i]

        return ex_ante_util_loss


    # Properties of the strategies in this market #############################

    def get_da_strategy_metrics(self, redraw_valuations: bool=False) -> Tuple[float]:
        """Calculate metrics that evaluate the current strategies and
        ultimately enable a comparison between different equilibria. The
        definitions of these metrics are from the paper by Leininger et al.
        1989. See the doc-strings of the individual methods for more
        information.

        Args:
            redraw_valuations (:bool:) whether or not to redraw the valuations
                of the agents.
            batch_size (:int:) maximal batch size for calculation.

        Returns:
            calculated_metrics(:Tuple[float]:) the tuple of all calculated
            metrics in the order:
                1. expected_utility_buyers
                2. expected_utility_sellers
                3. gains_from_trade
                4. efficiency_from_trade
            When not risk-neutral, one can also measure:
                5. expected_profit_buyers
                6. expected_profit_sellers
                7. profits_from_trade
                8. normalized_profits_from_trade
        """
        assert DoubleAuctionMechanism in inspect.getmro(type(self.mechanism)), \
            'These metrics are calculated specifically for double auctions.'

        if redraw_valuations:
            self.draw_valuations()

        bid_profile = self._get_bid_profile_by_current_strategies()
        valuations = self._observations

        actual_allocations, actual_payments = self.mechanism.play(bid_profile)
        truthful_allocations, truthful_payments = self.mechanism.play(valuations)

        expected_utility_metrics = self._get_strategy_utility_metrics(
            valuations, actual_allocations, actual_payments,
            truthful_allocations, truthful_payments
            )

        if self._all_agents_risk_neutral():  # -> profit = utility
            expected_profits_metrics = expected_utility_metrics
        else:
            expected_profits_metrics = self._get_strategy_utility_metrics(
                valuations, actual_allocations, actual_payments,
                truthful_allocations, truthful_payments, force_risk_neutral=True
                )

        return expected_utility_metrics + expected_profits_metrics

    def _get_strategy_utility_metrics(self, valuations, allocations, payments,
                                      truthful_allocations, truthful_payments,
                                      force_risk_neutral: bool=False):
        """Calculates the strategy metrics for the utilities of the agents.
        With force_risk_neutral=True, this corresponds to the profits metrics
        otherwise this represents the utility metrics.

        Args:
            valuations (Tensor of dim (*agent_batch_sizes x n_players x n_actions))
            allocations (Tensor of dim (*agent_batch_sizes x n_players x n_actions))
            payments (Tensor of dim (*agent_batch_sizes x n_players x n_actions))
            truthful_allocations (Tensor of dim (*agent_batch_sizes x n_players x n_actions))
            truthful_payments (Tensor of dim (*agent_batch_sizes x n_players x n_actions))
            force_risk_neutral (:bool:): Whether we set the agents to be risk_neutral

        Returns:
            expected_utility_buyers: mean ex_ante_utilities of buyers
            expected_utility_sellers: mean ex_ante_utilities of sellers
            # TODO: Check the definition for more than one buyer and seller!
            gains_from_trade: sum of ex_ante_utilities of buyers and sellers
            efficiency_from_trade: gains_from_trade / sum_of_ex_ante_utilities_with_truthful_strategy
        """
        expected_utility_buyers, expected_utility_sellers = self._get_expected_utilities(
            allocations, payments, valuations, force_risk_neutral=force_risk_neutral)
        exp_truthful_utilities = self._get_expected_utilities(
            truthful_allocations, truthful_payments, valuations, force_risk_neutral=force_risk_neutral)
        gains_from_trade = expected_utility_buyers + expected_utility_sellers
        efficiency_from_trade = gains_from_trade / sum(exp_truthful_utilities)
        return (
            expected_utility_buyers,
            expected_utility_sellers,
            gains_from_trade,
            efficiency_from_trade
        )

    def _all_agents_risk_neutral(self) -> bool:
        return all(agent.risk == 1.0 for agent in self.agents)

    def _get_bid_profile_by_current_strategies(self):
        action_length = self.agents[0].bid_size

        bid_profile = torch.zeros(self.batch_size, self.n_players, action_length,
                                  device=self.mechanism.device)
        for pos, bid in self._generate_agent_actions():
            bid_profile[:, pos, :] = bid
        return bid_profile

    def _get_expected_utilities(self, allocations, payments, valuations,
                                force_risk_neutral: bool=False) -> Tuple[float]:
        """Estimates the ex-ante utilities for the given instances via Monte-Carlo-Estimation.
        If force_risk_neutral is set to True, the risk-neutral utility functions are used. This
        corresponds then to the expected_profits.

        Returns:
            ex_ante_utilities = int_(prior_buyer x prior_seller)
                ex_post_utility(strategies_buyer(val_buyer), strategies_seller(val_seller))
        """
        if force_risk_neutral:  # Make agents risk neutral
            risk_parameters = [agent.risk for agent in self.agents]
            for agent in self.agents:
                agent.risk = 1.0

        utilities = torch.zeros(payments.shape, device=self.mechanism.device)
        for k, agent in enumerate(self.agents):
            utilities[:, k] = agent.get_utility(
                allocations[:, k,...], payments[:, k], valuations[:, k, ...]
                )

        if force_risk_neutral:  # Restore risk levels
            for agent, stored_risk in zip(self.agents, risk_parameters):
                agent.risk = stored_risk

        utility_buyers = utilities[:, :self.mechanism.n_buyers].mean()
        utility_seller = utilities[:, self.mechanism.n_buyers:].mean()
        return float(utility_buyers.detach().cpu()), float(utility_seller.detach().cpu())

    def get_pareto_efficiency_of_current_strategy(self, redraw_valuations: bool=False) -> float:
        """Calculates the Pareto-Efficiency for the current strategies.
        Gives the percentage of realized trade for the given strategic bids
        compared to the amount of trade if both agents would report truthfully.

        Returns:
            pareto_efficiency = (num_realized_trade_with_curr_strategy /
                (num_realized_trade_with_truthful_strategy).

        Note:
            Unlike the two-sided markets version of `self.get_efficiency`,
            this method calculates efficiency based on the current strategy
            profile.
        """
        if redraw_valuations:
            self.draw_valuations()
        if DoubleAuctionMechanism in inspect.getmro(type(self.mechanism)):
            bid_profile = self._get_bid_profile_by_current_strategies()
            actual_allocations, _ = self.mechanism.play(bid_profile)

            efficient_allocations, _ = \
                kDoubleAuction(self.mechanism.n_buyers, self.mechanism.n_sellers, 0.5) \
                .play(self._observations)

            # ``allocated to buyer'' == ``buyer reported higher value''
            allocated_to_highest_reported_value = \
                (actual_allocations == efficient_allocations).all(dim=2).all(dim=1)

            return float(allocated_to_highest_reported_value.float().mean().detach().cpu())
