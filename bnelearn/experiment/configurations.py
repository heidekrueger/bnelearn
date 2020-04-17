from typing import Type
import time

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
    n_units: int
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
    input_length: int = None
    BNE1: str = None
    BNE2: str = None
    constant_marginal_values: bool = False
    item_interest_limit: int = None
    efficiency_parameter: float = None
    pretrain_transform: callable = None

    def __post_init__(self):
        if self.input_length is None:
            self.input_length = self.n_units

@dataclass#(frozen=True) TODO: frozen not possible with post_init
class LoggingConfiguration:
    file_name: str = time.strftime('%Y-%m-%d %a %H:%M:%S')
    plot_frequency: int = 100
    plot_points: int = 100
    plot_show_inline: bool = True
    log_metrics: list = None
    regret_batch_size: int = None
    regret_grid_size: int = None
    regret_frequency: int = 100
    eval_batch_size: int = 2**12
    cache_eval_actions: bool = True
    
    save_figure_to_disk_png: bool = True
    save_figure_to_disk_svg: bool = True
    save_figure_data_to_disk: bool = True
    save_disable_all: bool = False

    def __post_init__(self):
        metrics = self.log_metrics
        self.log_metrics = {'opt': False,
                            'rmse': False,
                            'l2': False,
                            'regret': False}
        if metrics is not None:
            for metric in metrics:
                assert metric in self.log_metrics.keys(), "Metric not known."
                self.log_metrics[metric] = True
        if self.log_metrics['regret'] and self.regret_batch_size is None:
            self.regret_batch_size: int = 2**8
            self.regret_grid_size: int = 2**8
        if self.save_disable_all:
            self.save_figure_to_disk_png = False
            self.save_figure_to_disk_svg = False
            self.save_figure_data_to_disk = False

@dataclass#(frozen=True)
class LearningConfiguration:
    learner_hyperparams: dict = None
    optimizer_type: str or Type[Optimizer] = 'adam'
    optimizer_hyperparams: dict = None
    hidden_nodes: list = None 
    hidden_activations: list = None 
    pretrain_iters: int = 500
    batch_size: int = 2**13
    input_length: int = 1 #TODO: To be removed later

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

    @staticmethod
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