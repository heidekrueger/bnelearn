"""Implements logic to draw possibly correlated valuations between bidders.
   In particular, the models in Ausubel & Baranov 2019
"""
from abc import ABC, abstractmethod
import math
from typing import List, Dict
import torch
from torch.distributions import Distribution
from bnelearn.bidder import Bidder


def superseeded(func):
    def wrapper(*args, **kwargs):
        print("Calling a superseded function!")
        func(*args, **kwargs)
    return wrapper

class CorrelationDevice(ABC):
    """
    Implements logic to draw from joint prior distributions that are not
    independent in each bidder.

    Most of the work is done in the `draw_conditional` method. There, one agent
    is passed with its `player_position`. Her observation then is the condition
    for all further samples. Thus, any random samples are from a distribution
    conditioned on that observation.

    The method `get_weights` is sometimes used when the correlation is defined
    as a weighted sum.
    """

    def __init__(self, common_component_dist: Distribution or None,
                 batch_size: int, n_items: int, correlation_model: str,
                 correlation: float):

        assert 0.0 <= correlation <= 1.0, "Invalid correlation!"
        self.corr = correlation
        self.dist = common_component_dist
        self.batch_size = batch_size
        self.n_items = n_items
        self.correlation_model = correlation_model

    def draw_common_component(self):
        if self.dist is None:
            return None

        return self.dist.sample([self.batch_size, self.n_items])

    @abstractmethod
    def draw_conditionals(
            self, agents: List[Bidder],
            player_position: int,
            conditional_observation: torch.Tensor,
            batch_size: int=None
        ) -> Dict[int, torch.Tensor]:
        """
        Draw conditional observations of all opponents given the agent's
        observation `cond` at position `player_position` and draw conditional
        own type (if type != observation).

        Args
        ----
            agents: List[Bidder], list of bidders whose valuations are to be
                drwan.
            player_position: int, player position of agent.
            conditional_observation: torch.Tensor[shape=(batch_size_cond,
                n_items)], valuation of bidder on which the other valuations
                are to be conditioned on.
            batch_size: int, batch size for conditional types if different from
                batch size of `conditional_observation`.

        Returns
        -------
            dict {player_position[int]: cond_valuation[torch.Tensor[shape=(
                batch_size_cond * batch_size)]]}.
        """
        raise NotImplementedError

    @abstractmethod
    def get_weights(self):
        pass

    def get_component_and_weights(self):
        return self.draw_common_component(), self.get_weights()


class IndependentValuationDevice(CorrelationDevice):
    """Dummy `CorrelationDevice` for no correlation between agents."""
    def __init__(self):
        super().__init__(None, None, None, 'independent_valuations', 0.0)

    def get_weights(self):
        return torch.tensor(0.)

    def draw_conditionals(self, agents: List[Bidder], player_position: int,
                          conditional_observation: torch.Tensor, batch_size:
                          int=None):
        batch_size_0 = conditional_observation.shape[0]
        batch_size_1 = batch_size if batch_size is not None else batch_size_0
        return {
            agent.player_position: agent.draw_valuations_(
                common_component=self.draw_common_component(),
                weights=self.get_weights()
            )[:batch_size_1, :].repeat(batch_size_0, 1)
            for agent in agents
        }


class BernoulliWeightsCorrelationDevice(CorrelationDevice):
    """
    Implements correlation between two or more bidders, where their valuations
    depend additively on an individual component z_i and a common component s.
    In this scheme, a Bernoulli (0 or 1) weight determines that either v_i =
    z_i or v_i = s, with weights/probabilities being set such that correlation
    gamma is achieved between bidders.
    """
    def __init__(self, common_component_dist: Distribution,
                 batch_size: int, n_items, correlation: float):
        super().__init__(common_component_dist, batch_size, n_items,
                         "Bernoulli_weights_model", correlation)

    @superseeded
    def get_weights(self):
        """
        Choose individual component with prob (1-gamma), common component with
        prob gamma.
        """
        return torch.bernoulli(
            torch.tensor(self.corr).repeat(self.batch_size, 1) # different weight for each batch
        ).repeat(1, self.n_items)                              # same weight for each item in batch

    def draw_conditionals(
            self,
            agents: List[Bidder],
            player_position: int,
            conditional_observation: torch.Tensor,
            batch_size: int=None
        ) -> Dict[int, torch.Tensor]:
        """
        Conditional draws supported for arbitrary number of agents.
        """

        # TODO: possibly unify LLG correlation devices in parent
        batch_size_0 = conditional_observation.shape[0]
        batch_size_1 = batch_size if batch_size is not None else batch_size_0
        conditionals_dict = dict()

        # own valuation is given: repeat for new sample dimension
        conditionals_dict[player_position] = conditional_observation\
            .repeat(1,batch_size_1) \
            .view(batch_size_0 * batch_size_1, 1)

        # draw conditional observation of other local bidders
        for agent in agents:
            if agent.player_position != player_position:
                u = torch.empty(
                    (batch_size_1, 1),
                    device=conditional_observation.device
                ).uniform_(0, 1)
                conditionals_dict[agent.player_position] = \
                    self.local_cond_sample(conditional_observation) \
                        (u[:, 0]).view(batch_size_0 * batch_size_1, 1)

        return conditionals_dict

    def local_cond_sample(self, conditional_observation):
        """
        Draw samples of the opposing local bidder conditional on the local
        bidders' valuation: For a batch of obseravtions of one of the local
        bidders in LLG `conditional_observation`, this returns the iCDF of the
        other local bidder.
        """
        cond_batch = conditional_observation.view(-1, 1).shape[0]

        def icdf(x: torch.Tensor) -> torch.Tensor:
            sample_batch = x.view(-1, 1).shape[0]
            xx = x.repeat(1, cond_batch).view(cond_batch, sample_batch)
            ccond = conditional_observation.repeat(1, sample_batch) \
                .view(cond_batch, sample_batch)
            equal_local_values = torch.bernoulli(
                self.corr * torch.ones_like(x)) \
                .repeat(1, cond_batch).view(cond_batch, sample_batch)
            result = torch.logical_not(equal_local_values) * xx \
                + equal_local_values * ccond * torch.ones_like(xx)
            return result

        return icdf


class ConstantWeightsCorrelationDevice(CorrelationDevice):
    """
    Draw valuations according to the constant weights model in Ausubel &
    Baranov.

    Bidders valuations depend additively on an individual component z_i and a
    common component s. In this scheme, a weight w (across the entire batch!)
    is chosen such that
       v_i = (1-w)z_i + ws
    such that the correlation between v_i becomes gamma.
    """
    @superseeded
    def __init__(self, common_component_dist: Distribution,
                 batch_size: int, n_items: int, correlation: float):
        self.correlation = correlation
        self.weight = 0.5 if correlation == 0.5 \
            else (correlation - math.sqrt(correlation*(1-correlation))) / \
                (2*correlation - 1)
        super().__init__(common_component_dist, batch_size, n_items,
                         "constant_weights_model", correlation)

    @superseeded
    def get_weights(self):
        return self.weight

    @superseeded
    def draw_conditionals(
            self,
            agents: List[Bidder],
            player_position: int,
            conditional_observation: torch.Tensor,
            batch_size: int=None
        ) -> Dict[int, torch.Tensor]:
        """
        Conditional draws supported for arbitrary number of agents.
        """

        batch_size_0 = conditional_observation.shape[0]
        batch_size_1 = batch_size if batch_size is not None else batch_size_0
        conditionals_dict = dict()

        # own valuation is given: repeat for new sample dimension
        conditionals_dict[player_position] = conditional_observation \
            .repeat(1, batch_size_1)\
            .view(batch_size_0 * batch_size_1, 1)

        # draw conditional observation of other local bidders
        for agent in agents:
            if agent.player_position != player_position:
                conditionals_dict[agent.player_position] = \
                    self.draw_conditional_v2(
                        conditional_observation, batch_size_1) \
                        .view(batch_size_0 * batch_size_1, 1)

        return conditionals_dict

    @superseeded
    def draw_z1_given_v1(self, v1: torch.Tensor, batch_size):
        """
        Sample own private component of local bidder conditioned on its
        observation.

        Returns tensor of shape (v1.shape[0], batch_size).
        """
        batch_size_0 = v1.shape[0]
        batch_size_1 = batch_size
        gamma = self.correlation

        # degenerate case: z1 doesn't matter, but we still need the interface
        if gamma == 1.0:
            return torch.empty((batch_size_1, 1), device=v1.device) \
                .uniform_(0, 1) \
                .repeat(batch_size_0, 1) \
                .view(batch_size_0, batch_size_1)

        w = self.weight
        l_bounds = torch.max(torch.zeros_like(v1), (v1 - w)/(1 - w)) \
            .repeat(1, batch_size_1) \
            .view(batch_size_0, batch_size_1)
        u_bounds = torch.min(torch.ones_like(v1), v1/(1 - w)) \
            .repeat(1, batch_size_1) \
            .view(batch_size_0, batch_size_1)
        uniform = torch.empty((batch_size_1, 1), device=v1.device) \
            .uniform_(0, 1) \
            .repeat(batch_size_0, 1) \
            .view(batch_size_0, batch_size_1)
        return (u_bounds - l_bounds) * uniform + l_bounds

    @superseeded
    def draw_conditional_v2(self, v1: torch.Tensor, batch_size: int):
        """
        Sample local opponents observation conditioned on the other local's
        observation.

        Returns tensor of shape (v1.shape[0], batch_size).
        """
        # z2 is conditionally independent from v1
        z2 = torch.empty(batch_size, device=v1.device).uniform_(0, 1)

        z1_given_v1 = self.draw_z1_given_v1(v1, batch_size)
        v2 = (1 - self.weight)*(z2 - z1_given_v1) + v1
        return v2


class MineralRightsCorrelationDevice(CorrelationDevice):
    """
    Draw valuations according to the single item mineral rights model in
    Krishna.
    """
    def __init__(self, common_component_dist: Distribution,
                 batch_size: int, n_items: int, correlation: float):
        super().__init__(common_component_dist, batch_size, n_items,
                         "mineral_rights_model", correlation)

    def get_weights(self):
        return torch.tensor(.5) # must be strictly between 0, 1 to trigger right case

    def draw_conditionals(
            self,
            agents: List[Bidder],
            player_position: int,
            conditional_observation: torch.Tensor,
            batch_size: int=None
        ) -> Dict[int, torch.Tensor]:
        # This method should work for an arbirtray number of agents

        opponent_positions = [a.player_position for a in agents if
                              a.player_position != player_position]
        batch_size_0 = conditional_observation.shape[0]
        batch_size_1 = batch_size if batch_size is not None else batch_size_0
        conditionals_dict = dict()

        # Draw valuation given agent's observation
        v = self.draw_v_given_o(conditional_observation, batch_size_1) \
            .view(batch_size_0 * batch_size_1, 1)
        conditionals_dict[player_position] = v

        # Draw opponents' observations
        for opponent_position in opponent_positions:
            conditionals_dict.update(
                {opponent_position: torch.empty(
                    batch_size_1, device=conditional_observation.device) \
                    .uniform_(0, 1).repeat(1, batch_size_0) \
                    .view(batch_size_0 * batch_size_1, 1) * 2 * v}
            )

        return conditionals_dict

    @staticmethod
    def draw_v_given_o(o: torch.Tensor, batch_size: int):
        """
        Draw the common valuation given an observation o. This is done via
        calling its inverse CDF at a uniformly random sample.
        """
        c = -4 / (o**2 - 4)
        cond_batch_size = o.shape[0]

        o = o.clone().repeat(1, batch_size).view(cond_batch_size, batch_size)
        c = c.repeat(1, batch_size).view(cond_batch_size, batch_size)

        uniform = torch.empty((1, batch_size), device=o.device) \
            .uniform_(0, 1) \
            .repeat(1, cond_batch_size).view(cond_batch_size, batch_size)

        return o / torch.sqrt(-c * o**2 + 4*c + uniform*o**2 - 4*uniform)


class AffiliatedObservationsDevice(CorrelationDevice):
    """
    Draw valuations according to the single item affiliated observations model
    in Krishna.
    """
    def __init__(self, common_component_dist: Distribution, batch_size: int,
                 n_common_components: int, correlation: float):
        super().__init__(common_component_dist, batch_size,
                         n_common_components, "affiliated_observations_model",
                         correlation)

    def get_weights(self):
        return torch.tensor(.5) # must be strictly between 0, 1 to trigger right case

    def draw_conditionals(
            self,
            agents: List[Bidder],
            player_position: int,
            conditional_observation: torch.Tensor,
            batch_size: int=None
        ) -> Dict[int, torch.Tensor]:
        # This method should work for an arbirtray number of agents

        opponent_positions = [a.player_position for a in agents if
                              a.player_position != player_position]
        batch_size_0 = conditional_observation.shape[0]
        batch_size_1 = batch_size if batch_size is not None else batch_size_0
        conditionals_dict = dict()

        # Draw common component
        common_component = self.draw_common_given_o1(conditional_observation,
            batch_size_1) \
            .view(batch_size_0 * batch_size_1, 1)

        # Draw opponents' observations
        for opponent_position in opponent_positions:
            conditionals_dict.update(
                {opponent_position: torch.empty(
                    (1, batch_size_1), device=conditional_observation.device) \
                    .uniform_(0, 1).repeat(batch_size_0, 1) \
                    .view(batch_size_0 * batch_size_1, 1) \
                    + common_component}
            )

        # Common valuation given as mean of observations
        n_players = len(opponent_positions) + 1
        obs_sum = torch.zeros((n_players, batch_size_0 * batch_size_1),
                              device=conditional_observation.device)
        obs_sum[player_position, :] = conditional_observation \
            .repeat(1, batch_size_1) \
            .view(batch_size_0 * batch_size_1)
        for opponent_position in opponent_positions:
            obs_sum[opponent_position, :] = \
                conditionals_dict[opponent_position] \
                .view(batch_size_0 * batch_size_1)

        conditionals_dict[player_position] = obs_sum.mean(axis=0) \
                .view(batch_size_0 * batch_size_1, 1)

        return conditionals_dict

    @staticmethod
    def draw_common_given_o1(o1: torch.Tensor, batch_size: int):
        """
        Sample common signal conditioned on one agent's observation. Returns
        tensor of shape (v1.shape[0], batch_size).
        """
        batch_size_0 = o1.shape[0]
        batch_size_1 = batch_size

        l_bounds = torch.max(torch.zeros_like(o1), o1 - 1) \
            .repeat(1, batch_size_1).view(batch_size_0, batch_size_1)
        u_bounds = torch.min(torch.ones_like(o1), o1) \
            .repeat(1, batch_size_1).view(batch_size_0, batch_size_1)
        uniform = torch.empty((batch_size_1, 1), device=o1.device)\
            .uniform_(0, 1) \
            .repeat(batch_size_0, 1).view(batch_size_0, batch_size_1)

        return (u_bounds - l_bounds) * uniform + l_bounds


class MultiUnitDevice(CorrelationDevice):
    """
    Draw valuations according to the single item affiliated observations model
    in Krishna.
    """
    def __init__(self, common_component_dist: Distribution, batch_size: int,
                 n_common_components: int, correlation: float):
        super().__init__(common_component_dist, batch_size,
                         n_common_components, "multi_unit_model",
                         correlation)

    def get_weights(self):
        return torch.tensor(self.corr)

    def draw_conditionals(
            self, agents: List[Bidder],
            player_position: int,
            conditional_observation: torch.Tensor,
            batch_size: int=None
        ) -> Dict[int, torch.Tensor]:
        raise NotImplementedError
