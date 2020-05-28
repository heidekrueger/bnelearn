"""Implements logic to draw possibly correlated valuations between bidders.
   In particular, the models in Ausubel & Baranov 2019
"""
from abc import ABC, abstractmethod
import math
import torch
from torch.distributions import Distribution

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
    def get_weights(self):
        pass

    def get_component_and_weights(self):
        return self.draw_common_component(), self.get_weights()

class IndependentValuationDevice(CorrelationDevice):
    def __init__(self):
        super().__init__(None, None, None, 'independent_valuations', 0.0)
    
    def get_weights(self):
        return torch.tensor(0.)

class BernoulliWeightsCorrelationDevice(CorrelationDevice):
    def __init__(self, common_component_dist: Distribution, 
                 batch_size: int, n_items, correlation: float):
        super().__init__(common_component_dist, batch_size, n_items, "Bernoulli_weights_model", correlation)

    def get_weights(self):
        "choose individual component with prob (1-gamma), common component with prob gamma"
        return torch.bernoulli(
            torch.tensor(self.corr).repeat(self.batch_size, 1) # different weight for each batch 
            ).repeat(1, self.n_items)                          # same weight for each item in batch

class ConstantWeightsCorrelationDevice(CorrelationDevice):
    """Draw valuations according to the constant weights model in Ausubel & Baranov"""
    def __init__(self, common_component_dist: Distribution, 
                 batch_size: int, n_items: int, correlation: float):
        self.weight = 0.5 if correlation == 0.5 \
            else (correlation - math.sqrt(correlation*(1-correlation))) / (2*correlation - 1)
        super().__init__(common_component_dist, batch_size, n_items, "constant_weights_model", correlation)

    def get_weights(self):
        return self.weight
        