import warnings
from typing import Type

import torch
from torch.optim import Optimizer
import torch.nn as nn
from dataclasses import dataclass, field

# Only used on front end
@dataclass(frozen=True)
class RunningConfiguration:
    n_runs: int = 1
    n_epochs: int = 1000
    specific_gpu: int = 1
    n_players: list = None

@dataclass#(frozen=True)
class ExperimentConfiguration:
    payment_rule: str
    n_players: int = None
    model_sharing: bool = True
    risk: float = 1.0

    # Gaussian Distribution
    valuation_mean: float = None
    valuation_std: float = None

    # Uniform Distribution
    u_lo: list = None
    u_hi: list = None

    # MultiUnit #TODO: Check which are really needed and make sense here
    n_items: int = None
    BNE1: str = None #??
    BNE2: str = None#??
    #constant_marginal_values
    #item_interest_limit
    #efficiency_parameter
    #pretrain_transform
    #input_length
    def set_n_players(self, n_players):
        self.n_players=n_players

@dataclass(frozen=True)
class LoggingConfiguration:
    plot_frequency = 100
    log_metrics: list = None
    regret_batch_size: int = None
    regret_grid_size: int = None
    regret_frequency: int = 100
    eval_batch_size: int = 2**12
    cache_eval_actions: bool = True
    #TODO: Save figure to disc stuff, boolean
    log_png = True
    log_svg = True
    file_name = #Timestamp

    disable_logging = False

    def __post_init__(self):
        metrics = self.log_metrics
        self.log_metrics = {'bne': False,
                            'rmse': False,
                            'l2': False,
                            'regret': False}
        if metrics is not None:
            for metric in metrics:
                assert metric in ['bne','l2','rmse','regret'], "Metric not known."
                self.log_metrics[metric] = True
        if self.log_metrics['regret'] and self.regret_batch_size is None:
            self.regret_batch_size = 2**8
            self.regret_grid_size: int = 2**8

        if disable_logging:
            log_png = False
            log_svg = False

@dataclass(frozen=True)
class LearningConfiguration:
    learner_hyperparams: dict = None
    optimizer_type: str or Type[Optimizer] = 'adam'
    optimizer_hyperparams: dict = None
    hidden_nodes: list = None 
    hidden_activations: list = None 
    pretrain_iters: int = 500
    batch_size: int = 2**13

    def __post_init__(self):
        self.optimizer: Type[Optimizer] = self._set_optimizer(self.optimizer_type)
        if self.learner_hyperparams is None:
            self.learner_hyperparams = {'population_size': 128,
                                        'sigma': 1.,
                                        'scale_sigma_by_model_size': True}
        if self.optimizer_hyperparams is None:
            self.optimizer_hyperparams= {'lr': 3e-3}
        if self.hidden_nodes is None:
            self.hidden_nodes = [5,5,5]
        if self.hidden_activations is None:
            self.hidden_activations = [nn.SELU(),nn.SELU(),nn.SELU()]

    @staticmethod  #TODO: How to do this with a static method?
    def _set_optimizer(optimizer: str or Type[Optimizer]) -> Type[Optimizer]:
        """Maps shortcut strings to torch.optim.Optimizer types, if required."""
        if isinstance(optimizer, type) and issubclass(optimizer, Optimizer):
            return optimizer

        if isinstance(optimizer, str):
            if optimizer in ('adam', 'Adam'):
                return torch.optim.Adam
            if optimizer in ('SGD', 'sgd', 'Sgd'):
                return torch.optim.SGD
            # TODO: add more optimizers as needed
        raise ValueError('Optimizer type could not be inferred!')