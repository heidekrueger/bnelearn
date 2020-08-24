"""Implements logic to draw possibly correlated valuations between bidders.
   In particular, the models in Ausubel & Baranov 2019
"""
from abc import ABC, abstractmethod
import math
import numpy as np
from typing import List
import torch
from torch.distributions import Distribution
from bnelearn.bidder import Bidder

class CorrelationDevice(ABC):
    """
    Implements logic to draw from joint prior distributions that are not
    independent in each bidder.
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
    def draw_conditionals(self, agents: List[Bidder], player_position: int,
                          cond: torch.Tensor, batch_size: int=None):
        """Draw conditional types of all agents given one agent's observation `cond`"""
        raise NotImplementedError

    @abstractmethod
    def get_weights(self):
        pass

    def get_component_and_weights(self):
        return self.draw_common_component(), self.get_weights()


class IndependentValuationDevice(CorrelationDevice):
    def __init__(self):
        super().__init__(None, None, None, 'independent_valuations', 0.0)

    def get_weights(self):
        return torch.tensor(0.)

    def draw_conditionals(self, agents: List[Bidder], player_position: int,
                          cond: torch.Tensor, batch_size: int=None):
        batch_size_0 = cond.shape[0]
        batch_size_1 = batch_size if batch_size is not None else batch_size_0
        return {
            agent.player_position: agent.draw_valuations_(
                common_component=self.draw_common_component(), weights=self.get_weights()
            )[:batch_size_1, :].repeat(batch_size_0, 1)
            for agent in agents
        }


class BernoulliWeightsCorrelationDevice(CorrelationDevice):
    def __init__(self, common_component_dist: Distribution,
                 batch_size: int, n_items, correlation: float):
        super().__init__(common_component_dist, batch_size, n_items, "Bernoulli_weights_model", correlation)

    def get_weights(self):
        """choose individual component with prob (1-gamma), common component with prob gamma"""
        return torch.bernoulli(
            torch.tensor(self.corr).repeat(self.batch_size, 1) # different weight for each batch
        ).repeat(1, self.n_items)                              # same weight for each item in batch

    def draw_conditionals(self, agents: List[Bidder], player_position: int,
                          cond: torch.Tensor, batch_size: int=None):
        """Draw conditional types of all agents given one agent's observation `cond`"""
        # TODO Nils @Stefan: check math here
        opponent_positions = [a.player_position for a in agents if a.player_position != player_position]
        batch_size_0 = cond.shape[0]
        batch_size_1 = batch_size if batch_size is not None else batch_size_0
        conditionals_dict = dict()

        if player_position == 2: # agent is global bidder
            # no information about local bidders' valuations: can be drawn independently from global bidder
            for opponent_position in opponent_positions:
                conditionals_dict[opponent_position] = agents[opponent_position].draw_valuations_(
                    common_component=self.draw_common_component(), weights=self.get_weights()
                )[:batch_size_1, :].repeat(batch_size_0, 1).view(batch_size_0 * batch_size_1, 1)

        else: # agent is local bidder
            # opposing local bidder has to be conditioned
            local_opponent = min(opponent_positions)
            conditionals_dict[local_opponent] = self.local_cond_sample(cond)(
                torch.zeros(batch_size_1, device=cond.device).uniform_(0, 1)
            ).view(batch_size_0 * batch_size_1, 1)

            # opposing global bidder can be drawn independently
            global_oppopnent = 2
            conditionals_dict[global_oppopnent] = agents[global_oppopnent].draw_valuations_(
                common_component=self.draw_common_component(), weights=self.get_weights()
            )[:batch_size_1, :].repeat(batch_size_0, 1).view(batch_size_0 * batch_size_1, 1)

        return conditionals_dict

    def local_cond_sample(self, cond):
        """Draw samples of the opposing local bidder conditional one the local bidders' valuation."""
        cond_batch = cond.view(-1, 1).shape[0]

        def icdf(x: torch.Tensor):
            sample_batch = x.view(-1, 1).shape[0]
            xx = x.repeat(1, cond_batch).view(cond_batch, sample_batch)
            ccond = cond.repeat(1, sample_batch).view(cond_batch, sample_batch)
            switch = (torch.zeros_like(x).uniform_(0, 1) > self.corr) \
                .repeat(1, cond_batch).view(cond_batch, sample_batch)
            result = switch * xx + torch.logical_not(switch) * ccond * torch.ones_like(xx)
            return result

        return icdf


class ConstantWeightsCorrelationDevice(CorrelationDevice):
    """Draw valuations according to the constant weights model in Ausubel & Baranov"""
    def __init__(self, common_component_dist: Distribution,
                 batch_size: int, n_items: int, correlation: float):
        self.correlation = correlation
        self.weight = 0.5 if correlation == 0.5 \
            else (correlation - math.sqrt(correlation*(1-correlation))) / (2*correlation - 1)
        super().__init__(common_component_dist, batch_size, n_items, "constant_weights_model", correlation)

    def get_weights(self):
        return self.weight

    def draw_conditionals(self, agents: List[Bidder], player_position: int,
                          cond: torch.Tensor, batch_size: int=None):
        """Draw conditional types of all agents given one agent's observation `cond`"""
        # TODO Nils @Stefan: check math here
        opponent_positions = [a.player_position for a in agents if a.player_position != player_position]
        batch_size_0 = cond.shape[0]
        batch_size_1 = batch_size if batch_size is not None else batch_size_0
        conditionals_dict = dict()

        if player_position == 2: # agent is global bidder
            # no information about local bidders' valuations: can be drawn independently from global bidder
            for opponent_position in opponent_positions:
                conditionals_dict[opponent_position] = agents[opponent_position].draw_valuations_(
                    common_component=self.draw_common_component(), weights=self.get_weights()
                )[:batch_size_1, :].repeat(batch_size_0, 1).view(batch_size_0 * batch_size_1, 1)

        else: # agent is local bidder
            # opposing local bidder has to be conditioned
            local_opponent = min(opponent_positions)
            uniform = torch.zeros((batch_size_1, 1), device=cond.device).uniform_(0, 1)
            conditionals_dict[local_opponent] = self.icdf_v2_cond_v1(cond)(uniform) \
                .view(batch_size_0 * batch_size_1, 1)

            # opposing global bidder can be drawn independently
            global_oppopnent = 2
            conditionals_dict[global_oppopnent] = agents[global_oppopnent].draw_valuations_(
                common_component=self.draw_common_component(), weights=self.get_weights()
            )[:batch_size_1, :].repeat(batch_size_0, 1).view(batch_size_0 * batch_size_1, 1)

        return conditionals_dict

    #@staticmethod
    # def pdf_v2_cond_v1(self, gamma, v1):
    #     """Conditional PDF of observation 2 given observation 1"""
    #     switch = gamma < 0.5
    #     [mini, maxi] = sorted((gamma, 1 - gamma))

    #     if v1 < mini:
    #         def pdf(v2):
    #             result = torch.zeros_like(v2)

    #             increase_mask = v2 < v1
    #             result[increase_mask] = v2[increase_mask]

    #             constant_mask = torch.logical_and(v2 >= v1, v2 < 1-gamma)
    #             result[constant_mask] = v1

    #             decrease_mask = torch.logical_and(v2 >= 1-gamma, v2 < (1-gamma) + v1)
    #             result[decrease_mask] = 1 - gamma + v1 - v2[decrease_mask]

    #             h = v1
    #             l = 1-gamma - v1 + h
    #             result /= h*l

    #             return result

    #     elif v1 > maxi:
    #         def pdf(v2):
    #             result = torch.zeros_like(v2)

    #             increase_mask = torch.logical_and(v2 > (gamma-1) + v1, v2 < gamma)
    #             result[increase_mask] = v2[increase_mask] - (gamma-1) - v1

    #             constant_mask = torch.logical_and(v2 >= gamma, v2 < v1)
    #             result[constant_mask] = 1 - v1

    #             decrease_mask = v2 >= v1
    #             result[decrease_mask] = 1 - v2[decrease_mask]

    #             h = 1 - v1
    #             l = v1 - gamma + h
    #             result /= h*l

    #             return result

    #     else:
    #         def pdf(v2):
    #             result = torch.zeros_like(v2)

    #             if switch:
    #                 increase_mask = v2 < mini
    #                 result[increase_mask] = v2[increase_mask]

    #                 constant_mask = torch.logical_and(v2 >= mini, v2 < maxi)
    #                 result[constant_mask] = 1 - maxi

    #                 decrease_mask = v2 >= maxi
    #                 result[decrease_mask] = 1 - v2[decrease_mask]

    #                 h = 1 - maxi
    #                 l = maxi - mini + h
    #                 result /= h*l

    #             else:
    #                 increase_mask = torch.logical_and(v2 >= (gamma-1) + v1, v2 < v1)
    #                 result[increase_mask] = v2[increase_mask] - v1 + (1-gamma)

    #                 decrease_mask = torch.logical_and(v2 >= v1, v2 < (1-gamma) + v1)
    #                 result[decrease_mask] = v1 + (1-gamma) - v2[decrease_mask]

    #                 result /= (1-gamma)**2

    #             return result

    #     return pdf

    #@staticmethod
    # def cdf_v2_cond_v1(self, gamma, v1):
    #     """Conditional CDF of observation 2 given observation 1"""
    #     switch = gamma < 0.5
    #     [mini, maxi] = sorted((gamma, 1 - gamma))

    #     if v1 < mini:
    #         def cdf(v2):
    #             result = torch.zeros_like(v2)

    #             increase_mask = v2 < v1
    #             result[increase_mask] = 0.5 * v2[increase_mask]**2

    #             constant_mask = torch.logical_and(v2 >= v1, v2 < 1-gamma)
    #             result[constant_mask] = v1*v2[constant_mask] - 0.5 * v1**2

    #             decrease_mask = torch.logical_and(v2 >= 1-gamma, v2 < (1-gamma) + v1)
    #             result[decrease_mask] =  (1-gamma + v1)*v2[decrease_mask] \
    #                 - .5*v2[decrease_mask]**2 + v1*(1-gamma) - .5*v1**2 \
    #                 + .5*(1-gamma)**2 - (1-gamma+v1)*(1-gamma)

    #             h = v1
    #             l = 1-gamma - v1 + h
    #             result /= h*l

    #             final_mask = v2 >= (1-gamma) + v1
    #             result[final_mask] = 1
    #             return result

    #     elif v1 > maxi:
    #         def cdf(v2):
    #             result = torch.zeros_like(v2)

    #             increase_mask = torch.logical_and(v2 > (gamma-1) + v1, v2 < gamma)
    #             result[increase_mask] = 0.5 * (v2[increase_mask] - gamma+1 - v1)**2

    #             constant_mask = torch.logical_and(v2 >= gamma, v2 < v1)
    #             result[constant_mask] = (1 - v1)*(v2[constant_mask] - gamma) + .5*(1-v1)**2

    #             decrease_mask = v2 >= v1
    #             result[decrease_mask] = (v2[decrease_mask]) - .5*(v2[decrease_mask])**2 \
    #                 + (1 - v1)*(v1 - gamma) + .5*(1-v1)**2 - v1 + .5*v1**2

    #             h = 1 - v1
    #             l = v1 - gamma + h
    #             result /= h*l
    #             return result

    #     else:
    #         def cdf(v2):
    #             result = torch.zeros_like(v2)

    #             if switch:
    #                 increase_mask = v2 < mini
    #                 result[increase_mask] = .5*v2[increase_mask]**2

    #                 constant_mask = torch.logical_and(v2 >= mini, v2 < maxi)
    #                 result[constant_mask] = (1 - maxi)*v2[constant_mask] - (1 - maxi)*mini \
    #                     + .5*mini**2

    #                 decrease_mask = v2 >= maxi
    #                 result[decrease_mask] = v2[decrease_mask] - .5*v2[decrease_mask]**2 \
    #                     - maxi + .5*maxi**2 + (1 - maxi)*maxi - (1 - maxi)*mini + .5*mini**2

    #                 h = 1 - maxi
    #                 l = maxi - mini + h
    #                 result /= h*l
    #             else:
    #                 increase_mask = torch.logical_and(v2 >= (gamma-1) + v1, v2 < v1)
    #                 result[increase_mask] = 0.5*(v2[increase_mask])**2 \
    #                     + (-v1 + (1-gamma))*(v2[increase_mask]) \
    #                     - 0.5*((gamma-1) + v1)**2 - (-v1 + (1-gamma))*((gamma-1) + v1)

    #                 decrease_mask = torch.logical_and(v2 >= v1, v2 < (1-gamma) + v1)
    #                 result[decrease_mask] = (v1 + (1-gamma))*v2[decrease_mask] \
    #                     - .5*v2[decrease_mask]**2 - (v1 + (1-gamma))*v1 + v1**2 \
    #                     + (-v1 + (1-gamma))*(1-gamma) - .5*(v1 - (1-gamma))**2

    #                 result /= (1-gamma)**2

    #                 final_mask = v2 >= (1-gamma) + v1
    #                 result[final_mask] = 1
    #             return result

    #     return cdf

    def icdf_v2_cond_v1(self, v1):
        """Conditional inverse CDF of observation 2 given observation 1"""
        gamma = self.correlation
        switch = gamma < 0.5
        [mini, maxi] = sorted((gamma, 1 - gamma))
        cond_batch = v1.shape[0]

        def icdf(x):
            sample_batch = x.view(-1, 1).shape[0]
            xx = x.repeat(1, cond_batch).view(cond_batch, sample_batch)
            vv1 = v1.repeat(1, sample_batch).view(cond_batch, sample_batch)
            result = torch.zeros_like(xx)

            # case 1/3 for cond
            cond_mask_0 = vv1 < mini

            h = vv1
            l = 1-gamma - vv1 + h
            c = h*l

            increase_mask = xx < (.5/c) * vv1**2
            increase_mask = torch.logical_and(cond_mask_0, increase_mask)
            result[increase_mask] = torch.sqrt(2*c[increase_mask] * xx[increase_mask])

            constant_mask = torch.logical_and(xx >= .5*vv1**2 / c,
                                              xx < (vv1*(1-gamma) - .5*vv1**2)/c)
            constant_mask = torch.logical_and(cond_mask_0, constant_mask)
            result[constant_mask] = (c[constant_mask]*xx[constant_mask] + .5*vv1[constant_mask]**2) \
                / vv1[constant_mask]

            decrease_mask = xx >= (vv1*(1-gamma) - .5*vv1**2)/c
            decrease_mask = torch.logical_and(cond_mask_0, decrease_mask)
            c_1 = 1-gamma + vv1
            c_2 = -.5
            c_3 = vv1*(1-gamma) - .5*vv1**2 + .5*(1-gamma)**2 - (1-gamma+vv1)*(1-gamma)
            result[decrease_mask] = (torch.sqrt(4*c[decrease_mask]*c_2 \
                * xx[decrease_mask] + torch.pow(c_1[decrease_mask], 2) \
                - 4*c_2*c_3[decrease_mask]) - c_1[decrease_mask])/(2*c_2)


            # case 2/3 for cond
            cond_mask_1 = vv1 > maxi
            h = 1 - vv1
            l = vv1 - gamma + h
            c = h*l

            increase_mask = xx < (0.5/c) * (1 - vv1)**2
            increase_mask = torch.logical_and(cond_mask_1, increase_mask)
            result[increase_mask] = torch.sqrt(xx[increase_mask]/(.5/c[increase_mask])) \
                - (-gamma+1 - vv1[increase_mask])

            constant_mask = torch.logical_and(xx >= (0.5/c) * (1 - vv1)**2,
                                              xx < (1 - vv1)*(vv1 - gamma + .5*(1 - vv1))/c)
            constant_mask = torch.logical_and(cond_mask_1, constant_mask)
            result[constant_mask] = (c[constant_mask]/(1 - vv1[constant_mask]))*xx[constant_mask] \
                - .5*(1 - vv1[constant_mask]) + gamma

            decrease_mask = xx >= (1 - vv1)*(vv1 - gamma + .5*(1 - vv1))/c
            decrease_mask = torch.logical_and(cond_mask_1, decrease_mask)
            result[decrease_mask] = -torch.sqrt(2 *(-c[decrease_mask]*xx[decrease_mask] \
                + (gamma-1)*(vv1[decrease_mask] - 1))) + 1


            # case 2/3 for cond
            cond_mask_2 = torch.logical_not(torch.logical_or(cond_mask_0, cond_mask_1))
            if switch:
                h = 1 - maxi
                l = maxi - mini + h
                c = h*l

                increase_mask = xx < (.5/c)*mini**2
                increase_mask = torch.logical_and(cond_mask_2, increase_mask)
                result[increase_mask] = torch.sqrt(2*c * xx[increase_mask])

                constant_mask = torch.logical_and(xx >= (.5/c)*mini**2,
                                                xx < ((1-maxi)*(maxi-mini) + .5*mini**2)/c)
                constant_mask = torch.logical_and(cond_mask_2, constant_mask)
                result[constant_mask] = (-2*c*xx[constant_mask] + mini**2 \
                    + 2*(maxi-1)*mini)/(2*(maxi - 1))

                decrease_mask = xx >= ((1-maxi)*(maxi-mini) + .5*mini**2)/c
                decrease_mask = torch.logical_and(cond_mask_2, decrease_mask)
                result[decrease_mask] = 1 - torch.sqrt(-2*c*xx[decrease_mask] \
                            + mini**2 + 2*(maxi-1)*mini - maxi**2 + 1) 
            else:
                c = (1-gamma)**2
                increase_mask = xx < .5
                increase_mask = torch.logical_and(cond_mask_2, increase_mask)
                result[increase_mask] = torch.sqrt(2*c*xx[increase_mask]) + gamma + vv1[increase_mask] - 1

                decrease_mask = xx >= .5
                decrease_mask = torch.logical_and(cond_mask_2, decrease_mask)
                result[decrease_mask] = -torch.sqrt(2*(-c*xx[decrease_mask] + gamma**2 \
                    - 2*gamma + 1)) + vv1[decrease_mask] + 1 - gamma

            return result

        return icdf


class MineralRightsCorrelationDevice(CorrelationDevice):
    """Draw valuations according to the constant weights model in Ausubel & Baranov"""
    def __init__(self, common_component_dist: Distribution,
                 batch_size: int, n_items: int, correlation: float):
        super().__init__(common_component_dist, batch_size, n_items, "mineral_rights_model", correlation)

    def get_weights(self):
        return torch.tensor(.5) # must be strictly between 0, 1 to trigger right case

    def draw_conditionals(self, agents: List[Bidder], player_position: int,
                          cond: torch.Tensor, batch_size: int=None):
        """
        Draw conditional types of all agents given one agent's observation `cond`.

        Args
        ----
            agents: List[Bidder], list of bidders whose valuations are to be drwan.
            player_position: int, player position of agent.
            cond: torch.Tensor, valuation of bidder on which the other valuations are
                to be conditioned on.
            batch_size: int, batch size if different from batch size of `cond`.

        returns
        -------
            dict {player_position[int]: cond_valuation[torch.Tensor]}.
        """
        opponent_positions = [a.player_position for a in agents if a.player_position != player_position]
        batch_size_0 = cond.shape[0]
        batch_size_1 = batch_size if batch_size is not None else batch_size_0
        u = torch.zeros((batch_size_1, 3), device=cond.device).uniform_(0, 1)

        # (batch_size, batch_size): 1st dim for cond, 2nd for sample for each cond
        x0 = self.cond_marginal_icdf(cond)(u[:, 0])
        x1 = self.cond2_icdf(cond.repeat_interleave(batch_size_1, 0).view(batch_size_0, batch_size_1), x0)(u[:, 1])

        # Need to clip for numeric problems at edges
        x0 = torch.clamp(x0, 0, 2)
        x1 = torch.clamp(x1, 0, 2)

        # Set NaNs to 2
        x0[x0 != x0] = 2
        x1[x1 != x1] = 2

        # Use symmetry to decrease bias
        perm = torch.randperm(batch_size_1)
        cut = int(batch_size_1 / 2)
        temp = x0[:, :cut].clone()
        x0[:, :cut] = x1[:, :cut]
        x1[:, :cut] = temp
        x0 = x0[:, perm]
        x1 = x1[:, perm]

        # Sample agent's own type conditioned on signals
        z = torch.max(cond.repeat_interleave(batch_size_1, 0).view(batch_size_0, batch_size_1), torch.max(x0, x1))
        agent_type = self.type_cond_on_signal_icdf(z)(u[:, 2])

        return {
            player_position: agent_type.view(batch_size_0 * batch_size_1, 1),
            opponent_positions[0]: x0.view(batch_size_0 * batch_size_1, 1),
            opponent_positions[1]: x1.view(batch_size_0 * batch_size_1, 1)
        }

    @staticmethod
    def type_cond_on_signal_icdf(o):
        """iCDF of type distribution given observation o"""
        eps = 1e-4
        oo = torch.clamp(o, 0, 2 - eps)
        c = (- 4 / (torch.pow(oo, 2) - 4)).clone()
        cond_batch = c.shape[0]
        def icdf(v):
            vv = v.repeat(cond_batch, 1)
            return oo / torch.sqrt(-c * torch.pow(oo, 2) + 4*c + vv * torch.pow(oo, 2) - 4*vv)
        return icdf

    @staticmethod
    def density(x):
        """Mineral rights analytic density. Given in Krishna."""
        x = x.view(-1, 3)
        zz = torch.max(x, axis=1)[0]
        result = (4 - zz**2) / (16 * zz**2)
        result[torch.any(x < 0, 1)] = 0
        result[torch.any(x > 2, 1)] = 0
        return result

    # @staticmethod
    # def marginal_pdf(x):
    #     """Marginal density. Calulated via product density of two uniform distributions."""
    #     return -torch.log(x/2) / 2

    #     # distribution we want to sample from!
    #     def cond1_pdf(cond):
    #         """PDF of two given one"""
    #         factor = 1 / MineralRightsCorrelationDevice.marginal_pdf(cond)
    #         def pdf(x):
    #             x = np.atleast_2d(x)
    #             return factor * MineralRightsCorrelationDevice.density(
    #                 torch.cat([cond * torch.ones((x.shape[0], 1)), x], 1)
    #             )
    #         return pdf

    # @staticmethod
    # def cond_marginal_pdf(cond):
    #     """PDF of one, marginal one and given one"""
    #     def pdf(x):
    #         result = torch.zeros_like(x)
    #         maximum = x.clone()
    #         maximum[x < cond] = cond
    #         result = (maximum - 2) / (2*maximum * torch.log(cond/2))
    #         return result
    #     return pdf

    # @staticmethod
    # def cond_marginal_cdf(cond):
    #     """CDF of one, marginal one and given one"""
    #     z = cond
    #     f = (cond - 2) / (2*cond*torch.log(cond/2))
    #     c_1 = f * z
    #     c_2 = 2 * torch.log(cond/2)
    #     c_3 = (cond - 2*np.log(cond)) / (2*torch.log(cond/2))
    #     def cdf(x):
    #         x = x.view(-1, 1)
    #         result = torch.zeros_like(x)
    #         result[x < z] = f * x[x < z]
    #         result[x >= z] = c_1 + (x[x >= z] - 2*torch.log(x[x >= z])) / c_2 - c_3
    #         result[x < 0] = 0 # use clipping
    #         result[x >= 2] = 1
    #         return result
    #     return cdf

    @staticmethod
    def cond_marginal_icdf(cond):
        """iCDF of one, marginal one and given one"""
        z = cond.view(-1, 1)
        cond_batch = z.shape[0]

        # constants
        f = (z - 2) / (2*z * torch.log(z/2))
        f_inv = 1. / f
        c_1 = f * z
        c_2 = 2 * torch.log(z/2)
        c_3 = (z - 2*torch.log(z)) / (2*torch.log(z/2))

        def cdf(x):
            xx = x.repeat(1, cond_batch).view(cond_batch, -1)
            sample_batch = xx.shape[1]

            zz = z.repeat(1, sample_batch)
            ff = f.repeat(1, sample_batch)
            ff_inv = f_inv.repeat(1, sample_batch)
            cc_1 = c_1.repeat(1, sample_batch)
            cc_2 = c_2.repeat(1, sample_batch)
            cc_3 = c_3.repeat(1, sample_batch)

            result = torch.zeros((cond_batch, sample_batch), device=x.device)
            mask = xx < ff * zz
            result[mask] = ff_inv[mask] * xx[mask]
            result[~mask] = -2 * MineralRightsCorrelationDevice.lambertw_approx(
                - 1 / (2*torch.sqrt(torch.exp(cc_2[~mask]*(xx[~mask] - cc_1[~mask] + cc_3[~mask]))))
            )
            return result
        return cdf

    # @staticmethod
    # def cond2_pdf(cond1, cond2):
    #     """PDF when conditioning on two of three agents"""
    #     z = torch.max(cond1, cond2)
    #     factor = (4*z) / (2 - z) # s.t. it integrates to 1
    #     def pdf(x):
    #         x = x.view(-1, 1)
    #         return factor * MineralRightsCorrelationDevice.density(
    #             torch.cat([cond1 * torch.ones_like(x), cond2 * torch.ones_like(x), x], 1)
    #         )
    #     return pdf

    # @staticmethod
    # def cond2_cdf(cond1, cond2):
    #     """CDF when conditioning on two of three agents"""
    #     z = torch.max(cond1, cond2)
    #     factor_1 = (4*z) / (2 - z)
    #     factor_2 = (4-z**2) / (16*z**2)
    #     def cdf(x):
    #         result = np.zeros_like(x)
    #         f1 = factor_1 * torch.ones_like(x)
    #         f2 = factor_2 * torch.ones_like(x)
    #         zz = z * np.ones_like(x)

    #         result[x < z] = f1[x < z] * f2[x < z] * x[x < z]
    #         result[x >= z] = f1[x >= z] * (f2[x >= z] * zz[x >= z] - (x[x >= z]**2 + 4)/(16*x[x >= z]) \
    #             + (zz[x >= z]**2 + 4)/(16*zz[x >= z]))
    #         result[x < 0] = 0 # use clipping
    #         result[x >= 2] = 1
    #         return result
    #     return cdf

    @staticmethod
    def cond2_icdf(cond1, cond2):
        """iCDF when conditioning on two of three agents"""
        z = torch.max(cond1, cond2)
        cond_batch = z.shape[0]

        factor_1 = (4*z) / (2 - z)
        factor_2 = (4 - torch.pow(z, 2)) / (16*torch.pow(z, 2))

        def icdf(x):
            xx = x.repeat(cond_batch, 1).view_as(z)

            result = torch.zeros_like(z)
            sect1 = xx < z * factor_1 * factor_2
            result[sect1] = xx[sect1] / factor_1[sect1] / factor_2[sect1]

            sect2 = torch.logical_not(sect1)
            f1 = factor_1[sect2]
            f2 = factor_2[sect2]
            zz = z[sect2]
            result[sect2] = -(torch.sqrt(-32*f1*xx[sect2]*zz*(16*f2*zz**2 + zz**2 + 4) + f1**2*(256*f2**2*zz**4 \
                + 32*f2*(zz**2 + 4)*zz**2 + (zz**2 - 4)**2) + 256*xx[sect2]**2*zz**2) - f1*(16*f2*zz**2 + zz**2 \
                + 4) + 16*xx[sect2]*zz) / (2*f1*zz)

            return result
        return icdf

    @staticmethod
    def lambertw_approx(z, iters=4):
        """
        Approximation of Lambert W function via Halley’s method for
        positive values and via Winitzki approx. for negative values.
        """
        a = torch.zeros_like(z)
        eps = 0
        mask = z > eps

        for i in range(iters):
            a[mask] = a[mask] - (a[mask]*torch.exp(a[mask]) - z[mask]) / \
                (torch.exp(a[mask])*(a[mask] + 1)-((a[mask] + 2)*(a[mask]*torch.exp(a[mask]) - z[mask]))/(2*a[mask]+2))

        a[~mask] = (np.exp(1)*z[~mask]) / \
            (1 + ((np.exp(1) - 1)**(-1) - 1/np.sqrt(2) + 1/torch.sqrt(2*np.exp(1)*z[~mask] + 2))**(-1))
        return a


class AffiliatedObservationsDevice(CorrelationDevice):
    """Draw valuations according to the constant weights model in Ausubel & Baranov"""
    def __init__(self, common_component_dist: Distribution,
                 batch_size: int, n_common_components: int, correlation: float):
        super().__init__(common_component_dist, batch_size, n_common_components,
                         "affiliated_observations_model", correlation)

    def get_weights(self):
        return torch.tensor(.5) # must be strictly between 0, 1 to trigger right case

    def draw_conditionals(self, agents: List[Bidder], player_position: int,
                          cond: torch.Tensor, batch_size: int=None):
        """
        Draw conditional types of all agents given one agent's observation `cond`.

        Args
        ----
            agents: List[Bidder], list of bidders whose valuations are to be drwan.
            player_position: int, player position of agent.
            cond: torch.Tensor, valuation of bidder on which the other valuations are
                to be conditioned on.
            batch_size: int, batch size if different from batch size of `cond`.

        returns
        -------
            dict {player_position[int]: cond_valuation[torch.Tensor]}.
        """
        opponent_positions = [a.player_position for a in agents if a.player_position != player_position]
        batch_size_0 = cond.shape[0]
        batch_size_1 = batch_size if batch_size is not None else batch_size_0
        u = torch.zeros((batch_size_1, 1), device=cond.device).uniform_(0, 1)

        # (batch_size, batch_size): 1st dim for cond, 2nd for sample for each cond
        opponent_observation = self.icdf_o2_cond_o1(cond)(u[:, 0])

        # Sample agent's own type conditioned on signal
        agent_type = 0.5 * (
            cond.repeat_interleave(batch_size_1, 0).view(batch_size_0, batch_size_1) \
            + opponent_observation
        )

        return {
            player_position: agent_type.view(batch_size_0 * batch_size_1, 1),
            opponent_positions[0]: opponent_observation.view(batch_size_0 * batch_size_1, 1),
        }

    # @staticmethod
    # def pdf_o2_cond_o1(o1):
    #     """Conditional PDF of observation 2 given observation 1"""
    #     def pdf(o2):
    #         result = torch.zeros_like(o2)

    #         mask_1 = o2 < o1
    #         result[mask_1] = o2[mask_1]

    #         mask_2 = torch.logical_and(o2 >= o1, o2 < 1)
    #         result[mask_2] = o1

    #         mask_3 = torch.logical_and(o2 < o1 + 1, o2 >= 1)
    #         result[mask_3] = 1 + o1 - o2[mask_3]

    #         return result

    #     if o1 < 1.0:
    #         f = 1 / o1
    #         return lambda o2: f * pdf(o2)
    #     else:
    #         o1 = 2 - o1
    #         f = 1 / o1
    #         return lambda o2: f * pdf(2 - o2)
    #     return pdf

    # @staticmethod
    # def cdf_o2_cond_o1(o1):
    #     """Conditional CDF of observation 2 given observation 1"""

    #     def cdf(o2):
    #         result = torch.zeros_like(o2)

    #         mask_0 = o2 < 0
    #         result[mask_0] = 0

    #         mask_1 = torch.logical_and(o2 >= 0, o2 < o1)
    #         result[mask_1] = (1/(2*o1)) * torch.pow(o2[mask_1], 2)

    #         mask_2 = torch.logical_and(o2 >= o1, o2 < 1)
    #         result[mask_2] = -0.5*o1 + o2[mask_2]

    #         mask_3 = torch.logical_and(o2 >= 1, o2 < 1 + o1)
    #         result[mask_3] = -0.5 * o1 + 1 + ((1+o1-(o2[mask_3]+1)/2)/o1) * (o2[mask_3] - 1)

    #         mask_4 = o2 >= 1 + o1
    #         result[mask_4] = 1

    #         return result

    #     if o1 < 1.0:
    #         return lambda o2: cdf(o2)
    #     else:
    #         o1 = 2 - o1
    #         return lambda o2: cdf(o2 - (1-o1))

    @staticmethod
    def icdf_o2_cond_o1(o1):
        """Conditional iCDF of observation 2 given observation 1"""
        o1_flat = o1.view(-1, 1)
        cond_batch = o1_flat.shape[0]

        def icdf(x):
            sample_batch = x.view(-1, 1).shape[0]
            xx = x.repeat(1, cond_batch).view(cond_batch, -1)
            oo1 = o1_flat.repeat(1, sample_batch)
            result = torch.zeros_like(xx)

            mask_0 = oo1 > 1.0
            oo1[mask_0] = 2 - oo1[mask_0]

            mask_1 = xx < 0.5*oo1
            result[mask_1] = torch.sqrt(2*oo1[mask_1] * xx[mask_1])

            mask_2 = torch.logical_and(xx >= 0.5*oo1, xx < 1 - 0.5*oo1)
            result[mask_2] = xx[mask_2] + 0.5*oo1[mask_2]

            mask_3 = xx >= 1 - 0.5*oo1
            result[mask_3] = -torch.sqrt(2*oo1[mask_3]*(-xx[mask_3] + 1)) + oo1[mask_3] + 1

            mask_4 = oo1 > 1
            result[mask_4] += oo1[mask_4] - 1

            result[mask_0] += (1 - oo1[mask_0])
            return result

        return icdf
