from collections import Iterable, deque
import torch

from bnelearn.bidder import Bidder
from bnelearn.mechanism import Mechanism
from bnelearn.strategy import Strategy

dynamic = object()


class Environment():
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

    def __init__(self, mechanism: Mechanism, environment_agents: Iterable, max_env_size=None,
                 batch_size=100, n_players=2, strategy_to_bidder_closure=None):
        assert isinstance(environment_agents, Iterable), "iterable of environment_agents must be supplied"

        self.agents = deque(environment_agents, max_env_size)
        self.max_env_size = max_env_size
        self.batch_size = batch_size
        self.mechanism = mechanism
        self.n_players = n_players
        self._strategy_to_bidder_closure = strategy_to_bidder_closure


    def get_reward(self, agent: Bidder or Strategy, print_percentage=False):
        """Returns reward of a single player against the environment.
           Reward is calculated as average utility for each of the batch_size x env_size games
        """

        if isinstance(agent, Strategy):
            agent: Bidder = self._bidder_from_strategy(agent)
        
        # get a batch of bids for each player
        agent.batch_size = self.batch_size
        agent.draw_valuations_()
        agent_bid = agent.get_action()

        if print_percentage:
            print((agent_bid / agent.valuations).mean().item())

        utility: torch.Tensor = torch.tensor(0.0, device=agent.device)

        for opponent_bid in self._generate_opponent_bids():
            allocation, payments = self.mechanism.run(
                torch.cat((agent_bid, opponent_bid), 1).view(self.batch_size, self.n_players, 1)
            )
            u = agent.get_utility(allocation[:,0,:], payments[:,0]).mean()
            utility.add_(u)
        
        # average over all players in the environment
        utility.div_(self.size())
        
        return utility
    
    def push_agent(self, agent: Bidder or Strategy):
        """
            Add an agent to the environment, possibly pushing out the oldest one)
        """
        if isinstance(agent, Strategy):
            agent: Bidder = self._bidder_from_strategy(agent)
        
        self.agents.append(agent)


    def _bidder_from_strategy(self, strategy: Strategy):
        """ Transform a strategy into a player that plays that strategy """
        if self._strategy_to_bidder_closure:
            return self._strategy_to_bidder_closure(strategy, self.batch_size) 
        
        raise NotImplementedError()

    def _generate_opponent_bids(self):
        """ Generator function yielding batches of bids for each player in environment agents"""
        for opponent in self.agents:
            opponent.batch_size = self.batch_size
            opponent.draw_valuations_()
            yield opponent.get_action()           

    def size(self):
        """Returns the number of agents/opponent setups in the environment.""" 
        return len(self.agents)
    
    def is_empty(self):
        """True if no agents in the environment"""
        return len(self.agents) == 0