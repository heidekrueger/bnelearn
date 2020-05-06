from typing import Type, List
import time
import os

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

@dataclass
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

    # LLG
    gamma: float = 0.0

    # Multi-Unit
    n_units: int = None
    pretrain_transform: callable = None
    constant_marginal_values: bool = False
    item_interest_limit: int = None

    # Split-Award
    efficiency_parameter: float = None
    input_length: int = None

    # LLLLGG
    core_solver: str = 'NoCore'

    def __post_init__(self):
        if self.input_length is None:
            self.input_length = self.n_units



@dataclass
class LoggingConfiguration:
    """Controls logging and evaluation aspects of an experiment suite.

    If logging is enabled, the experiment runs will be logged to the following
    directories:
        log_root_dir / 
            [setting-specific dir hierarchy determined by Experiment subclasses] /
                experiment_timestamp + experiment_name / 
                    run_timestamp + run_seed
    """
    enable_logging: bool = True #If false, disables ALL logging
    # root directory for logging. subdirectories will be inferred and created based on experiment name and config
    log_root_dir: str = os.path.join(os.path.expanduser('~'), 'bnelearn', 'experiments')
    # TODO Stefan: where is this used.
    experiment_name: str = None
    # Rationale behind timestamp format: should be ordered chronologically but include weekday. 
    # Using . instead of : for compatability with Windows 
    experiment_timestamp: str = time.strftime('%Y-%m-%d %a %H.%M') #removed %S here, we won't need seconds
    plot_frequency: int = 100
    plot_points: int = 100
    plot_show_inline: bool = True
    log_metrics: list = None
    regret_batch_size: int = None
    regret_grid_size: int = None
    regret_frequency: int = 100
    eval_batch_size: int = 2**22
    cache_eval_actions: bool = True
    max_epochs: int = None
    save_tb_events_to_csv_aggregate: bool = True
    save_tb_events_to_csv_detailed: bool = False
    save_tb_events_to_binary: bool = False
    save_models: bool = True

    save_figure_to_disk_png: bool = True
    save_figure_to_disk_svg: bool = True
    save_figure_data_to_disk: bool = True

    def __post_init__(self):

        self.experiment_dir = self.experiment_timestamp
        if self.experiment_name:
            self.experiment_dir += ' ' + str(self.experiment_name)

        metrics = self.log_metrics
        self.log_metrics = {'opt': False,
                            'l2': False,
                            'regret': False}
        if metrics is not None:
            for metric in metrics:
                assert metric in self.log_metrics.keys(), "Metric not known."
                self.log_metrics[metric] = True
        if self.log_metrics['regret'] and self.regret_batch_size is None:
            self.regret_batch_size: int = 2**8
            self.regret_grid_size: int = 2**8
        if not self.enable_logging:
            self.save_tb_events_to_csv_aggregate = False
            self.save_tb_events_to_csv_detailed: bool = False
            self.save_tb_events_to_binary: bool = False
            self.save_models = False
            self.save_figure_to_disk_png = False
            self.save_figure_to_disk_svg = False
            self.save_figure_data_to_disk = False

    # def update_file_name(self, name=None):
    #     if name is None:
    #         self.file_name = time.strftime('%Y-%m-%d %a %H:%M:%S')

@dataclass
class LearningConfiguration:
    learner_hyperparams: dict = None
    optimizer_type: str or Type[Optimizer] = 'adam'
    optimizer_hyperparams: dict = None
    hidden_nodes: List[int] = None
    hidden_activations: List[nn.Module] = None
    pretrain_iters: int = 500
    batch_size: int = 2**18
    input_length: int = 1 #TODO: Stefan: remove here. @Paul

    def __post_init__(self):
        self.optimizer: Type[Optimizer] = self._set_optimizer(self.optimizer_type)
        if self.learner_hyperparams is None:
            self.learner_hyperparams = {'population_size': 64,
                                        'sigma': 1.,
                                        'scale_sigma_by_model_size': True}
        if self.optimizer_hyperparams is None:
            self.optimizer_hyperparams= {'lr': 3e-3}
        if self.hidden_nodes is None:
            self.hidden_nodes = [5,5,5]
        if self.hidden_activations is None:
            self.hidden_activations = [nn.SELU() for layer in self.hidden_nodes]

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
