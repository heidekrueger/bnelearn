# -*- coding: utf-8 -*-
import random
from abc import ABC, abstractmethod
from collections import deque
from collections.abc import Iterable
from typing import Callable, List, Set, Tuple

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

    @abstractmethod
    def get_reward(self, agent: Player or Strategy, **kwargs):
        """Return reward for a player playing a certain strategy"""
        pass #pylint: disable=unnecessary-pass

    def _generate_agent_actions(self, exclude: Set[int] or None = None, **kwargs):
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

    def get_player_from_strategy(self, strategy: Strategy, **kwargs):
        """ Transform a strategy into a player that plays that strategy """
        if self._strategy_to_player:
            player: Player = self._strategy_to_player(strategy, self.batch_size, **kwargs)
            #bidder.player_position = player_position
            return player

        raise NotImplementedError("No strategy_to_player closure provided!")

    def size(self):
        """Returns the number of agents/opponent setups in the environment."""
        return len(self.agents)

    def is_empty(self):
        """True if no agents in the environment"""
        return len(self.agents) == 0


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

    def _calc_exp_util(self, m, probs, n_players, i, iteration = 0):
        """
        Calculates the expected utility for a player (i)

        Args
        ----------
        m:  payoff matrix of player of interest (i)
        probs: current beliefs
        n_players: number of players
        i: player of interest
        iteration: must initialized with 0 and counts up to number of players
        ----------

        Returns
        ----------
        exp_util: expected utility of player i (tensor of dimension (1 x n_actions[i])
        ----------
        """
        iteration += 1
        if iteration >= n_players:
            return m
        else:
            exp_util = probs[(i+iteration)%n_players] \
                       .matmul(self._calc_exp_util(m, probs, n_players, i, iteration)) \
                       .squeeze()
            return exp_util

    def solve_with_gerding_k_smooth_fictitious_play(self, dev, initial_beliefs: torch.Tensor=None,
                                                    w_b=1, iterations=100
                                                   ) -> Tuple[List[torch.Tensor],List[torch.Tensor],List[torch.Tensor]]:
        """
        Based on paper: Gerding et al. (2008). However this is based on auctions)

        1. Players have initial guesses about other players probabilities for certain actions
        2. All players calculate their corresponding expected utility for taking an action
            given the guesses about the other players actions
        3. All players play k-exponential fictitious play response (\\sigma_i(b) in Gerding et al. (2008))
        4. Update beliefs about other players actions considering P(a|v) and \\sigma(a|v)

        Mixed NE for BattleOfTheSexes:
        initial_beliefs = torch.tensor([[0.6, 0.4],[0.4, 0.6]], dtype = torch.float, device = dev)

        For testing PaulTestGame
        initial_beliefs = torch.tensor([[0.1,0.9],[0.2,0.8],[0.3,0.7]], dtype = torch.float, device = dev)
        -------------
        Args
        w_b: bid specific weights. Currently just one.
        -------------

        Returns
        -------------
        sigma:
            history of probabilities of playing a certain action
            (list of length n_players [tensor of dimension (n_iterations, n_actions[current player])])
        values:
            history of expected valuations of playing a certain action
            (list of length n_players [tensor of dimension (n_iterations, n_actions[current player])])
        strat:
            latest probabilities of playing a certain action
            (list of length n_players [tensor of dimension (n_actions[current player])])
        -------------
        """
        # Parameters
        tau = torch.tensor(1.0)
        n_players = self.game.outcomes.shape[len(self.game.outcomes.shape)-1]
        n_actions = [self.game.outcomes.shape[i] for i in range(len(self.game.outcomes.shape)-1)]
        probs = [torch.zeros(n_actions[i], dtype = torch.float, device = dev) for i in range(n_players)]

        # 1. assumptions of player i's actions
        if initial_beliefs is None:
            initial_beliefs = [torch.zeros(n_actions[i], dtype = torch.float, device = dev) for i in range(n_players)]
            for i in range(n_players):
                initial_beliefs[i] = torch.rand(n_actions[i], dtype = torch.float, device = dev)
                initial_beliefs[i] = initial_beliefs[i]/initial_beliefs[i].sum(0)

        probs = initial_beliefs.copy()

        sigma =[    torch.zeros(iterations, n_actions[i], dtype = torch.float, device = dev)
                    for i in range(n_players)
               ] # actions
        strat = [torch.zeros(n_actions[i], dtype = torch.float, device = dev) for i in range(n_players)]
        values = [torch.zeros(iterations, n_actions[i], dtype = torch.float, device = dev) for i in range(n_players)]
        for n in range(iterations):
            for i in range(n_players):
                # 2. Compute expected values based on global probabilities
                player_p_matrix = self.game.outcomes.select(n_players,i) \
                                  .permute(*[(1+i+j)%n_players for j in range(n_players)])
                values[i][n,:] = self._calc_exp_util(player_p_matrix, probs, n_players, i)
                sigma[i][n,:] = (w_b * torch.exp((1/tau) * (values[i][n,:]))) \
                              / (w_b * torch.exp((1/tau) * (values[i][n,:]))).sum()

            # 4. Update global probabilities
            for i in range(0,n_players):
                probs[i] = 1/((n+1) + 1) * ((n+1) * probs[i] + sigma[i][n,:])

            tau = 1/torch.log(torch.tensor(n+3.))

        for i in range(n_players):
            strat[i] = sigma[i][iterations-1,:]

        return sigma, values, strat

    def solve_with_gerding_smooth_fictitious_play(self, dev, initial_beliefs: torch.Tensor=None,
                                                  w_b=1, iterations=100
                                                 ) -> Tuple[List[torch.Tensor],List[torch.Tensor],List[torch.Tensor]]:
        """
        Inspired by paper: Gerding et al. (2008) but with different updating procedure

        1. Players have initial guesses about other players probabilities for certain actions
        2. All players calculate their corresponding expected utility for taking an action
            given the guesses about the other players actions
        3. Update own actions and beliefs about other players actions according to \\sigma_i(b) in Gerding et al. (2008)

        Mixed NE for BattleOfTheSexes:
        initial_beliefs = torch.tensor([[0.6, 0.4],[0.4, 0.6]], dtype = torch.float, device = dev)

        For testing PaulTestGame
        initial_beliefs = torch.tensor([[0.1,0.9],[0.2,0.8],[0.3,0.7]], dtype = torch.float, device = dev)

        -------------
        Args
        w_b: bid specific weights. Currently just one.
        -------------

        Returns
        -------------
        sigma:
            history of probabilities of playing a certain action
            (list of length n_players [tensor of dimension (n_iterations, n_actions[current player])])
        values:
            history of expected valuations of playing a certain action
            (list of length n_players [tensor of dimension (n_iterations, n_actions[current player])])
        strat:
            latest probabilities of playing a certain action
            (list of length n_players [tensor of dimension (n_actions[current player])])
        -------------
        """
        # Parameters
        tau = torch.tensor(1.0)
        n_players = self.game.outcomes.shape[len(self.game.outcomes.shape)-1]
        n_actions = [self.game.outcomes.shape[i] for i in range(len(self.game.outcomes.shape)-1)]
        probs = [torch.zeros(n_actions[i], dtype = torch.float, device = dev) for i in range(n_players)]

        # 1. assumptions of player i's actions
        if initial_beliefs is None:
            initial_beliefs = [torch.zeros(n_actions[i], dtype = torch.float, device = dev) for i in range(n_players)]
            for i in range(n_players):
                initial_beliefs[i] = torch.rand(n_actions[i], dtype = torch.float, device = dev)
                initial_beliefs[i] = initial_beliefs[i]/initial_beliefs[i].sum(0)

        probs = initial_beliefs.copy()

        sigma = [torch.zeros(iterations, n_actions[i], dtype = torch.float, device = dev) for i in range(n_players)]
        strat = [torch.zeros(n_actions[i], dtype = torch.float, device = dev) for i in range(n_players)]
        values = [torch.zeros(iterations, n_actions[i], dtype = torch.float, device = dev) for i in range(n_players)]
        for n in range(iterations):
            # 2. Choose actions and compute values based on global probabilities
            for i in range(n_players):
                player_p_matrix = self.game.outcomes \
                                  .select(n_players,i) \
                                  .permute(*[(1+i+j)%n_players for j in range(n_players)])
                values[i][n,:] = self._calc_exp_util(player_p_matrix, probs, n_players, i)

            # 3. Update global probabilities
            for i in range(0,n_players):
                probs[i] = (w_b * torch.exp((1/tau) * (values[i][:(n+1),:].sum(0) / (n+1)))) \
                         / (w_b * torch.exp((1/tau) * (values[i][:(n+1),:].sum(0) / (n+1)))).sum()
                sigma[i][n,:] = probs[i]

            tau = 1/torch.log(torch.tensor(n+3.))

        for i in range(n_players):
            strat[i] = sigma[i][iterations-1,:]
        return sigma, values, strat

    def solve_with_smooth_fictitious_play_known_fct(self, dev, initial_beliefs: torch.Tensor=None,
                                                    tau=1, iterations=100
                                                   ) -> Tuple[List[torch.Tensor],List[torch.Tensor],List[torch.Tensor]]:
        """
        Returns
        -------------
        sigma:
            history of probabilities (0 or 1) of playing a certain action
            (list of length n_players [tensor of dimension (n_iterations, n_actions[current player])])
        values:
            history of valuations of playing a certain action as best response (BR)
            (list of length n_players [tensor of dimension (n_iterations, n_actions[current player])])
        strat:
            probability of playing a certain action based on past experiences
            (list of length n_players [tensor of dimension (n_actions[current player])])
        -------------
        Based on description in: Fudenberg, 1999 - The Theory of Learning, Chapter 2.2
        Players choose actions simultaneously

        0. Initialize weight function k(i)_0 ##For now with only ones
        1. Player (i) computes probability of opponents (-i) playing a strategy (s(-i)) at time (t) as:
            gamma(i)_t(s(-i)) = k(i)_t(s(-i))/(sum_(s(-i) in S(-i)) k(i)_t(s(-i))))
        2. Player (i) plays a rule (p(i)_t(gamma(i)_t), i.e. action) as best response
        """
        # Parameters
        torch.manual_seed(0)
        random.seed(a=0)
        n_players = self.game.outcomes.shape[len(self.game.outcomes.shape)-1]
        n_actions = [self.game.outcomes.shape[i] for i in range(len(self.game.outcomes.shape)-1)]
        weights = [torch.zeros(iterations, n_actions[i], dtype = torch.float, device = dev) for i in range(n_players)]
        probs = [torch.zeros(n_actions[i], dtype = torch.float, device = dev) for i in range(n_players)]
        values = [torch.zeros(iterations, dtype = torch.float, device = dev) for i in range(n_players)]
        # 0. Initialize weight function k(i)_0
        if initial_beliefs is None:
            initial_beliefs = [torch.rand(n_actions[i], dtype = torch.float, device = dev) for i in range(n_players)]
        # normalize
        for i in range(n_players):
            for a in range(n_actions[i]):
                weights[i][0,a] = initial_beliefs[i][a]/initial_beliefs[i].sum()

        for n in range(iterations-1):
            # 1. Player (i) computes probability of opponents (-i) playing a strategy (s(-i)) at time (t)
            for i in range(n_players):
                probs[i][:] = weights[i][n,:]
            for i in range(n_players):
                player_p_matrix = self.game.outcomes \
                                  .select(n_players,i) \
                                  .permute(*[(1+i+j)%n_players for j in range(n_players)])
                # 2. Player (i) computes expected utils according to beliefs. Then applies softmax (exponential smooth)
                probs_tmp = (1/tau * self._calc_exp_util(player_p_matrix, probs, n_players, i)).softmax(0)
                weights[i][n+1,:] = 1/(n+1) * (n * weights[i][n,:] + probs_tmp)
            # updating tau -> 0 Todo: Review update fct. here...
            tau = 1/torch.log(torch.tensor(n+3.))
            # TODO: Get payoff of realization
            for i in range(n_players):
                values[i][n] = 0
        return weights, values, probs

    def solve_with_fudenberg_fictitious_play(self, dev, initial_beliefs: torch.Tensor=None,
                                             smooth = False, tau=1, iterations=100
                                            ) -> Tuple[List[torch.Tensor],List[torch.Tensor],List[torch.Tensor]]:
        """
        Based on description in: Fudenberg, 1999 - The Theory of Learning, Chapter 2.2

        Players choose actions simultaneously

        0. Initialize weight function k(i)_0 ##For now with only ones
        1. Player (i) computes probability of opponents (-i) playing a strategy (s(-i)) at time (t) as:
            gamma(i)_t(s(-i)) = k(i)_t(s(-i))/(sum_(s(-i) in S(-i)) k(i)_t(s(-i))))
        2. Player (i) plays a rule (p(i)_t(gamma(i)_t), i.e. action) as best response


        Returns
        -------------
        sigma:
            history of probabilities (0 or 1) of playing a certain action
            (list of length n_players [tensor of dimension (n_iterations, n_actions[current player])])
        values:
            history of valuations of playing a certain action as best response (BR)
            (list of length n_players [tensor of dimension (n_iterations, n_actions[current player])])
        strat:
            probability of playing a certain action based on past experiences
            (list of length n_players [tensor of dimension (n_actions[current player])])
        -------------
        """

        # Parameters
        torch.manual_seed(0)
        random.seed(a=0)
        n_players = self.game.outcomes.shape[len(self.game.outcomes.shape)-1]
        n_actions = [self.game.outcomes.shape[i] for i in range(len(self.game.outcomes.shape)-1)]
        weights = [torch.zeros(iterations, n_actions[i], dtype = torch.float, device = dev) for i in range(n_players)]

        probs = [torch.zeros(n_actions[i], dtype = torch.float, device = dev) for i in range(n_players)]
        values = [torch.zeros(iterations, dtype = torch.float, device = dev) for i in range(n_players)]

        # 0. Initialize weight function k(i)_0
        if initial_beliefs is None:
            initial_beliefs = [torch.rand(n_actions[i], dtype = torch.float, device = dev) for i in range(n_players)]
        for i in range(n_players):
            weights[i][0,:] = initial_beliefs[i].clone()

        for n in range(iterations-1):
            action = [None for i in range(n_players)]
            # 1. Player (i) computes probability of opponents (-i) playing a strategy (s(-i)) at time (t)
            for i in range(n_players):
                for a in range(n_actions[i]):
                    probs[i][a] = weights[i][n,a].sum()/weights[i][n,:].sum()
            for i in range(n_players):
                player_p_matrix = self.game.outcomes \
                                  .select(n_players,i) \
                                  .permute(*[(1+i+j)%n_players for j in range(n_players)])

                if smooth:
                    # 2. Player (i) computes expected utils according to beliefs. Then applies softmax (expon. smooth)
                    probs_tmp = (1/tau * self._calc_exp_util(player_p_matrix, probs, n_players, i)).softmax(0)
                    action[i] = random.choices(population=[i for i in range(len(probs_tmp))], weights=probs_tmp, k=1)[0]
                else:
                    # 2. Player (i) plays a rule (p(i)_t(gamma(i)_t), i.e. action) as best response
                    action[i] = self._calc_exp_util(player_p_matrix, probs, n_players, i).max(dim = 0, keepdim=False)[1]
                weights[i][n+1,:] = weights[i][n,:]
                weights[i][n+1,action[i]] = weights[i][n,action[i]] + 1
            if smooth:
                # updating tau -> 0 Todo: Review update fct. here...
                tau = 1/torch.log(torch.tensor(n+3.))
            # Get payoff of realization
            for i in range(n_players):
                values[i][n] = self.game.outcomes[tuple(action)][i]

        return weights, values, probs


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

    def __init__(self, mechanism: Mechanism, agents: Iterable, max_env_size=None,
                 batch_size=100, n_players=2, strategy_to_bidder_closure: Callable[[Strategy], Bidder]=None):

        super().__init__(
            agents=agents,
            n_players=n_players,
            batch_size=batch_size,
            strategy_to_player_closure=strategy_to_bidder_closure
            )
        self.max_env_size = max_env_size

        # turn agents into deque TODO: might want to change this.
        self.agents = deque(self.agents, max_env_size)
        self.mechanism = mechanism

    def get_reward(self, agent: Bidder, draw_valuations=False, **kwargs): #pylint: disable=arguments-differ
        """Returns reward of a single player against the environment.
           Reward is calculated as average utility for each of the batch_size x env_size games
        """

        if not isinstance(agent, Player):
            raise ValueError()
        # get a batch of bids for each player
        agent.batch_size = self.batch_size

        player_position = agent.player_position if agent.player_position else 0

        # draw valuations
        if draw_valuations:
            agent.prepare_iteration()
            self.draw_valuations_(exclude = set([player_position]))

        # get agent_bid
        agent_bid = agent.get_action()
        n_items = agent_bid.shape[1]

        if not self.agents:# Env is empty --> play only with own action against 'nature'
            allocation, payments = self.mechanism.play(
                agent_bid.view(self.batch_size, self.n_players, n_items)
            )
            utility = agent.get_utility(allocation[:,0,:], payments[:,0]).mean()
        else: # at least 1 environment agent --> build bid_profile, then play
            # get bid profile
            # TODO: check where those params should be taken from ultimately... game? agent? self?
            bid_profile = torch.zeros(self.batch_size, self.n_players, n_items,
                                      dtype=agent_bid.dtype, device = self.mechanism.device)
            bid_profile[:, player_position, :] = agent_bid

            # ugly af hack: if environment is dynamic, all player positions will be
            # none. simply start at 1 for the first opponent and count up
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
            utility = agent.get_utility(
                allocation[:,player_position,:],
                payments[:,player_position]
                ).mean()

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

    def push_agent(self, agent: Bidder or Strategy, **kwargs):
        """
            Add an agent to the environment, possibly pushing out the oldest one)
        """
        if isinstance(agent, Strategy):
            agent: Bidder = self.get_player_from_strategy(agent, **kwargs)

        self.agents.append(agent)
