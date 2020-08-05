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
            )[:batch_size_0, ...].repeat(batch_size_1, 1)
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

    @abstractmethod
    def draw_conditionals(self, agents: List[Bidder], player_position: int,
                          cond: torch.Tensor, batch_size: int=None):
        """Draw conditional types of all agents given one agent's observation `cond`"""
        raise NotImplementedError

class ConstantWeightsCorrelationDevice(CorrelationDevice):
    """Draw valuations according to the constant weights model in Ausubel & Baranov"""
    def __init__(self, common_component_dist: Distribution, 
                 batch_size: int, n_items: int, correlation: float):
        self.weight = 0.5 if correlation == 0.5 \
            else (correlation - math.sqrt(correlation*(1-correlation))) / (2*correlation - 1)
        super().__init__(common_component_dist, batch_size, n_items, "constant_weights_model", correlation)

    def get_weights(self):
        return self.weight

    @abstractmethod
    def draw_conditionals(self, agents: List[Bidder], player_position: int,
                          cond: torch.Tensor, batch_size: int=None):
        """Draw conditional types of all agents given one agent's observation `cond`"""
        raise NotImplementedError

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
        raise NotImplementedError
        opponent_positions = [a.player_position for a in agents if a.player_position != player_position]
        batch_size_0 = cond.shape[0]
        batch_size_1 = batch_size if batch_size is not None else batch_size_0
        u = torch.zeros((batch_size_1, 2), device=cond.device).uniform_(0, 1)

        # (batch_size, batch_size): 1st dim for cond, 2nd for sample for each cond
        x0 = self.cond_marginal_icdf(cond)(u[:, 0])
        x1 = self.cond2_icdf(cond.repeat(1, batch_size_1).view(batch_size_0, batch_size_1), x0)(u[:, 1])

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

        # Sample agent's own type conditioned on signal
        own_type = self.type_cond_on_signal_icdf(cond)(torch.zeros_like(cond).uniform_(0, 1))

        return {
            player_position: own_type,
            opponent_positions[0]: x0.view(batch_size_0 * batch_size_1, 1),
            opponent_positions[1]: x1.view(batch_size_0 * batch_size_1, 1)
        }

    @staticmethod
    def type_cond_on_signal_icdf(z):
        c = - 4 / (z**2 - 4)
        def icdf(v):
            return z / torch.sqrt(-c * z**2 + 4*c + v * z**2 - 4*v)
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
            xx = x.repeat(cond_batch, 1)
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
            xx = x.repeat(cond_batch, 1)

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
    def lambertw_approx(z, iters=2):
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
