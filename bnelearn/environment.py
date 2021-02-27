# -*- coding: utf-8 -*-
"""
This module contains environments - a collection of players and
possibly state histories that is used to control game playing and
implements reward allocation to agents.
"""

from abc import ABC, abstractmethod
from typing import Callable, Set, List, Iterable

import numpy as np
import torch
import chaospy
import vegas
from zunis.integration import Integrator
from emukit.examples.vanilla_bayesian_quadrature_with_rbf.vanilla_bq_loop_with_rbf import create_vanilla_bq_loop_with_rbf_kernel

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
        ... 
        strategy_to_bidder_closure: A closure (strategy) -> Bidder to
            transform strategies into a Bidder compatible with the environment
        rule : str among [pseudorandon, halton, sobol, latin_hypercube]
        inplace_sampling: use torch.Tensor.uniform_ whenever possible instead of torch.distributions.Uniform(0,1).rsample(torch.Size)
        scramble : True : randomized sobol sequence, False : deterministic sobol sequence
        
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
        self.domain = self.get_domain()
        
        self.integral_bounds = self.get_integral_bounds()
        self.dtype= self.agents[0].get_valuation_grid(1).dtype


    def get_domain(self):
        """
        returns integration domain for the utility calculations 
        for example for LLG returns np.array([[0,0,0],[1,1,2]])
        """
        
        integral_bounds = []
        for agent in self.agents: 
            for j in range(agent.n_items):
                test = False
                if isinstance(agent._grid_lb,torch.Tensor) :
                    try:                      
                        low = agent._grid_lb.item()
                    except Exception:
                        if len(agent._grid_lb) > 1 :
                            low = agent._grid_lb.numpy()
                            test = True
                else : 
                    low = agent._grid_lb

                if isinstance(agent._grid_ub,torch.Tensor) :
                    try : 
                        high = agent._grid_ub.item()
                    except Exception: 
                        if len(agent._grid_ub) > 1 :
                            high = agent._grid_ub.numpy()
                else : 
                    high = agent._grid_ub
                if test : 
                    for j in range(len(np.vstack([low,high]).T)):
                        integral_bounds.append(np.vstack([low,high]).T[j].tolist())
                else : 
                    integral_bounds.append([low,high])

        return np.array(integral_bounds).T

    def get_integral_bounds(self):
        """
        returns integration bound for the utility calculations 
        for example for LLG returns np.array([[0,1],[0,1],[0,2]])
        """
        integral_bounds = []
        for agent in self.agents: 
            for _ in range(agent.n_items):
                test = False
                if isinstance(agent._grid_lb,torch.Tensor) :
                    try:                      
                        low = agent._grid_lb.item()
                    except Exception:
                        if len(agent._grid_lb) > 1 :
                            low = agent._grid_lb.numpy()
                            test = True
                else : 
                    low = agent._grid_lb

                if isinstance(agent._grid_ub,torch.Tensor) :
                    try : 
                        high = agent._grid_ub.item()
                    except Exception: 
                        if len(agent._grid_ub) > 1 :
                            high = agent._grid_ub.numpy()
                else: 
                    high = agent._grid_ub
                if test: 
                    for j in range(len(np.vstack([low,high]).T)):
                        integral_bounds.append(np.vstack([low,high]).T[j].tolist())
                else: 
                    integral_bounds.append([low,high])
        return integral_bounds

    def get_bid_profile(self, agents : Iterable[Bidder],valuations): 
        """ gets bid_profile from valuations

        Args:
            valuations (torch.Tensor)
            agents (Iterable[Bidder])

        Returns:
            [torch.Tensor] : bid profile
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
        if unknown_valuations is None: 
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

class AuctionEnvironment_Gaussian_Quad(AuctionEnvironment):
    """
    An environment of agents to play against and evaluate strategies.

    In particular this means:
        - an iterable of sets of -i players that a strategy of a single player can be tested against
        - accept strategy as argument, then play batch_size rounds and return the reward

    Args:
        ... 
        strategy_to_bidder_closure: A closure (strategy) -> Bidder to
            transform strategies into a Bidder compatible with the environment
        rule : not used
        inplace_sampling: not used
        scramble : True : not used
        degree : int : max degree of the orthogonal polynomials used for the integration, Default 60 for LLG
        The utilities are calculated using the Gaussian Quadrature Method. Only to be used in low dimensional 
        settings.
        Gaussian Quadrature methods allow exact estimation of the integral of a polynomial of degree 2N-1 
        using only N points. The integrand is approximated using a family of orthogonal polynomials.
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
            scramble : bool = True,
            degree : int = 60
        ):
        self.degree = degree
        super(AuctionEnvironment_Gaussian_Quad,self).__init__(mechanism = mechanism,
            agents=agents,
            n_players = n_players,
            strategy_to_player_closure =  strategy_to_player_closure,
            correlation_groups = correlation_groups,
            correlation_devices = correlation_devices,
            rule = rule,
            antithetic = antithetic,
            inplace_sampling = inplace_sampling,
            scramble = scramble)

        zeros, weights = chaospy.quad_gauss_legendre(self.degree, domain=self.domain)
        self.zeros = zeros
        self.weights = weights

    def get_strategy_reward(
        self,
        batch_size:int,
        strategy:Strategy,
        **strat_to_player_kwargs):

        if not self._strategy_to_player:
            raise NotImplementedError('This environment has no strategy_to_player closure!')
        agent = self._strategy_to_player(strategy, **strat_to_player_kwargs)
        player_position = strat_to_player_kwargs["player_position"]
        copy_agents = self.agents
        copy_agents[strat_to_player_kwargs["player_position"]] = agent

        def f(X):
            """
            help function to calculate the expected value
            """
            batch= len(self.weights)
            valuations = torch.tensor(np.array(X), dtype = self.dtype, device = self.device).reshape(batch,self.n_players,self.agents[0].n_items)
            bids = self.get_bid_profile(copy_agents,valuations)
            allocation, payments = self.mechanism.play(bids)
            utility = self.agents[player_position].get_utility(allocation[:,player_position,:], payments[:,player_position],valuations[:,player_position,:])
            return utility.detach()
        result = (torch.from_numpy(self.weights).to(self.device))*(f(self.zeros.T).flatten())

        return result.sum() / self.domain[1].prod()


class AuctionEnvironment_Bayesian_Quad(AuctionEnvironment):
    """
    An environment of agents to play against and evaluate strategies.

    In particular this means:
        - an iterable of sets of -i players that a strategy of a single player can be tested against
        - accept strategy as argument, then play batch_size rounds and return the reward

    Args:
        ... 
        strategy_to_bidder_closure: A closure (strategy) -> Bidder to
            transform strategies into a Bidder compatible with the environment
        rule : not used
        inplace_sampling: not used
        scramble : True : not used
        stopping condition: number of observations (in addition to batch_size) to condition on to get the posterior distribution of the 
        expected utility for a given strategy, observations are chosen to decrease the variance of the integral
        should be a small int because the method is slow
        The Bayesian Quadrature method treats the numerical integration problem as a Bayesian Inference problem. 
        A Gaussian Process Prior is placed on the integrand u. The integrand is evaluated on a small number of samples O of the integration
        domain. After conditioning on the observations, we get a posterior distribution over the integrand 
        that is used to derive a posterior distribution of the integral
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
            scramble : bool = True,
            stopping_condition : int = 2
        ):
        self.stopping_condition = stopping_condition
        super(AuctionEnvironment_Bayesian_Quad,self).__init__(mechanism = mechanism,
            agents=agents,
            n_players = n_players,
            strategy_to_player_closure =  strategy_to_player_closure,
            correlation_groups = correlation_groups,
            correlation_devices = correlation_devices,
            rule = rule,
            antithetic = antithetic,
            inplace_sampling = inplace_sampling,
            scramble = scramble)


    def get_strategy_reward(
        self,
        batch_size:int,
        strategy:Strategy,
        **strat_to_player_kwargs):

        if not self._strategy_to_player:
            raise NotImplementedError('This environment has no strategy_to_player closure!')
        agent = self._strategy_to_player(strategy, **strat_to_player_kwargs)
        player_position = strat_to_player_kwargs["player_position"]
        copy_agents = self.agents
        copy_agents[strat_to_player_kwargs["player_position"]] = agent
        X_init = self.state_device.draw_state(copy_agents, batch_size=batch_size)["valuations"].reshape(batch_size,self.n_players*self.agents[0].n_items).detach().cpu().numpy()

        def f(X):
            """
            help function to calculate the integral, emukit_loop.run_loop only accept a function in this format, output should be numpy array
            """
            if X.shape == (self.n_players * self.agents[0].n_items,):
                batch = 1
            else : 
                batch = len(X)

            valuations = torch.tensor(np.array(X), dtype = self.dtype, device = self.device).reshape(batch,self.n_players,self.agents[0].n_items)
            bids = self.get_bid_profile(copy_agents,valuations)
            allocation, payments = self.mechanism.play(bids)
            utility = self.agents[player_position].get_utility(allocation[:,player_position,:], payments[:,player_position],valuations[:,player_position,:])
            return utility.detach().cpu().numpy().reshape(batch,1)

        emukit_loop = create_vanilla_bq_loop_with_rbf_kernel(X=X_init, Y=f(X_init),integral_bounds=self.integral_bounds,measure=None)
        emukit_loop.run_loop(user_function=f, stopping_condition=self.stopping_condition)
        integral_mean, _= emukit_loop.model.integrate()
        integral_mean = integral_mean / self.domain[1].prod()
        return torch.tensor(integral_mean,dtype = self.dtype, device = self.device)

class AuctionEnvironment_NIS(AuctionEnvironment):
    """
    An environment of agents to play against and evaluate strategies.

    In particular this means:
        - an iterable of sets of -i players that a strategy of a single player can be tested against
        - accept strategy as argument, then play batch_size rounds and return the reward

    Args:
        ... 
        strategy_to_bidder_closure: A closure (strategy) -> Bidder to
            transform strategies into a Bidder compatible with the environment
        rule : not used
        inplace_sampling: not used
        scramble : True : not used
        n_iter : int, general number of iterations
        The Importance sampling technique is a classical variance reduction technique for MC methods 
        that aims at directing the sampling  effort at the most important regions of the integration domain. 
        The NIS algorithm uses normalizing flows to automatically find a suitable sampling distribution.
    
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
            scramble : bool = True,
            n_iter : int = 2
                    ):
        self.n_iter = n_iter

        super(AuctionEnvironment_NIS,self).__init__(mechanism = mechanism,
            agents=agents,
            n_players = n_players,
            strategy_to_player_closure =  strategy_to_player_closure,
            correlation_groups = correlation_groups,
            correlation_devices = correlation_devices,
            rule = rule,
            antithetic = antithetic,
            inplace_sampling = inplace_sampling,
            scramble = scramble)
        self.n_items = self.agents[0].n_items
        self.factor = torch.from_numpy(self.domain[1]).reshape(self.n_players,self.n_items).to(self.device)
    def get_strategy_reward(
        self,
        batch_size:int,
        strategy:Strategy,
        **strat_to_player_kwargs):

        if not self._strategy_to_player:
            raise NotImplementedError('This environment has no strategy_to_player closure!')
        agent = self._strategy_to_player(strategy, **strat_to_player_kwargs)
        player_position = strat_to_player_kwargs["player_position"]
        # TODO: this should rally be in AuctionEnv subclass
        copy_agents = self.agents
        copy_agents[strat_to_player_kwargs["player_position"]] = agent

        def f(X):
            """
            help function used for the integration
            """

            valuations = self.factor * X.reshape(batch_size,self.n_players,self.agents[0].n_items)
            valuations  = valuations.type(torch.FloatTensor).to(self.device)
            
            bids = self.get_bid_profile(copy_agents, valuations)
            allocation, payments = self.mechanism.play(bids)
            utility = self.agents[player_position].get_utility(allocation[:,player_position,:], payments[:,player_position],valuations[:,player_position,:])
            return utility.detach()

        dim = self.n_players * self.n_items

        integrator = Integrator(d=dim,f=f,device=self.device, n_iter= self.n_iter , n_points= batch_size)
        result, uncertainty, _ = integrator.integrate()

        return torch.tensor(result,dtype = self.dtype, device=self.device)


class AuctionEnvironment_Classical_Vegas(AuctionEnvironment):
    """
    An environment of agents to play against and evaluate strategies.

    In particular this means:
        - an iterable of sets of -i players that a strategy of a single player can be tested against
        - accept strategy as argument, then play batch_size rounds and return the reward

    Args:
        ... 
        strategy_to_bidder_closure: A closure (strategy) -> Bidder to
            transform strategies into a Bidder compatible with the environment
        rule : not used
        inplace_sampling: not used
        scramble : True : not used
        n_int : int, general number of iterations
        The Vegas algorithm is an adaptive Monte Carlo algorithm widely used in High Energy physics.
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
            scramble : bool = False,
            n_int : int = 10):
        self.n_int = n_int

        super(AuctionEnvironment_Classical_Vegas,self).__init__(mechanism = mechanism,
            agents=agents,
            n_players = n_players,
            strategy_to_player_closure =  strategy_to_player_closure,
            correlation_groups = correlation_groups,
            correlation_devices = correlation_devices,
            rule = rule,
            antithetic = antithetic,
            inplace_sampling = inplace_sampling,
            scramble = scramble)
        self.n_items = self.agents[0].n_items
        self.dim = self.n_players * self.agents[0].n_items
        self.integ = vegas.Integrator(self.integral_bounds)



    def get_strategy_reward(
        self,
        batch_size:int,
        strategy:Strategy,
        **strat_to_player_kwargs):

        if not self._strategy_to_player:
            raise NotImplementedError('This environment has no strategy_to_player closure!')
        agent = self._strategy_to_player(strategy, **strat_to_player_kwargs)
        player_position = strat_to_player_kwargs["player_position"]
        copy_agents = self.agents
        copy_agents[strat_to_player_kwargs["player_position"]] = agent

        @vegas.batchintegrand
        def f(X):
            """
            help function to calculate the expected utility
            """
            if X.shape == (self.n_players * self.agents[0].n_items,):
                batch = 1
            else : 
                batch = len(X)

            valuations = torch.tensor(np.array(X), dtype = self.dtype, device = self.device).reshape(batch,self.n_players,self.agents[0].n_items)
            bids = self.get_bid_profile(copy_agents,valuations)
            allocation, payments = self.mechanism.play(bids)
            utility = self.agents[player_position].get_utility(allocation[:,player_position,:], payments[:,player_position],valuations[:,player_position,:])
            return utility.detach().cpu().numpy().astype(np.double)
        result = self.integ(f, nitn=self.n_int, neval=batch_size) / self.domain[1].prod()
        return torch.tensor(result.val,dtype = self.dtype, device=self.device)