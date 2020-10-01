"""This module provides dataclasses that are used to hold configs."""
import dataclasses
import json
import os
import time
import warnings
from dataclasses import dataclass
from typing import List, Type, Iterable

import torch
import torch.nn as nn

from torch.optim import Optimizer


# ToDo Perhaps all the defaults should be moved to the ConfigurationManager?

class EnhancedJSONEncoder(json.JSONEncoder):
    def default(self, o):
        if dataclasses.is_dataclass(o):
            return dataclasses.asdict(o)
        return super().default(o)


# Only used on front end
@dataclass
class RunningConfig:
    n_runs: int = 1
    n_epochs: int = 1000
    #n_players: int = None
    seeds: Iterable[int] = None


@dataclass
class SettingConfig:
    n_players: int = None
    payment_rule: str = 'first_price'
    risk: float = 1.0

    # SymmetricPriorSingleItemExperiment
    common_prior: torch.distributions.Distribution = None

    # Gaussian Distribution
    valuation_mean: float = None
    valuation_std: float = None

    # Uniform Distribution
    u_lo: list = None
    u_hi: list = None

    # Correlated Valuations, independent by default
    gamma: float = 0.0
    correlation_types: str = 'independent'
    correlation_groups: List[List[int]] = None  # player_ids in each group
    correlation_coefficients: List[float] = None  # coefficients in each group

    # Multi-Unit
    n_units: int = None
    pretrain_transform: callable = None
    constant_marginal_values: bool = False
    item_interest_limit: int = None

    # Split-Award
    efficiency_parameter: float = None

    # LLLLGG
    core_solver: str = 'NoCore'
    # parallel: int = 1 in hardware config now
    


@dataclass
class LearningConfig:
    model_sharing: bool = True
    learner_hyperparams: dict = None
    optimizer_type: str or Type[Optimizer] = 'adam'
    optimizer_hyperparams: dict = None
    hidden_nodes: List[int] = None
    hidden_activations: List[nn.Module] = None
    pretrain_iters: int = 500
    batch_size: int = 2 ** 18

    def __post_init__(self):
        self.optimizer: Type[Optimizer] = self._set_optimizer(self.optimizer_type)
        if self.learner_hyperparams is None:
            self.learner_hyperparams = {'population_size': 64,
                                        'sigma': 1.,
                                        'scale_sigma_by_model_size': True}
        if self.optimizer_hyperparams is None:
            self.optimizer_hyperparams = {'lr': 1e-3}
        if self.hidden_nodes is None:
            self.hidden_nodes = [10,10]
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
            # add more optimizers as needed
        raise ValueError('Optimizer type could not be inferred!')


@dataclass
class LoggingConfig:
    """Controls logging and evaluation aspects of an experiment suite.

    If logging is enabled, the experiment runs will be logged to the following
    directories:
        log_root_dir /
            [setting-specific dir hierarchy determined by Experiment subclasses] /
                experiment_timestamp + experiment_name /
                    run_timestamp + run_seed
    """
    enable_logging: bool = True  # If false, disables ALL logging
    # root directory for logging. subdirectories will be inferred and created based on experiment name and config
    log_root_dir: str = os.path.join(os.path.expanduser('~'), 'bnelearn', 'experiments')
    experiment_name: str = None
    # Rationale behind timestamp format: should be ordered chronologically but include weekday.
    # Using . instead of : for compatability with Windows
    experiment_timestamp: str = time.strftime('%Y-%m-%d %a %H.%M')  # removed %S here, we won't need seconds
    plotting: bool = True
    plot_frequency: int = 100
    plot_points: int = 100
    plot_show_inline: bool = True
    log_metrics: dict = None
    export_step_wise_linear_bid_function_size:float = None
    log_componentwise_norm: bool = False

    # Stopping Criterion #TODO: this section should go into ExperimentConfiguration
    stopping_criterion_rel_util_loss_diff: float = None
    stopping_criterion_frequency: int = 100  # how often (each x iters) to calculate the stopping criterion metric
    stopping_criterion_duration: int = 3  # the x most recent evaluations will be used for calculating stationarity
    stopping_criterion_batch_size: int = 2 ** 10  # TODO: ideally this should be unified with general util_loss batch and grid sizes
    stopping_criterion_grid_size: int = 2 ** 9

    # Utility Loss calculation
    util_loss_batch_size: int = None
    util_loss_grid_size: int = None
    util_loss_frequency: int = 100

    # Eval vs known bne
    eval_batch_size: int = 2 ** 22
    cache_eval_actions: bool = True

    save_tb_events_to_csv_aggregate: bool = True
    save_tb_events_to_csv_detailed: bool = False
    save_tb_events_to_binary_detailed: bool = False
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
                            'util_loss': False}
        if metrics is not None:
            for metric, value in metrics.items():
                assert metric in self.log_metrics.keys(), "Metric not known."
                self.log_metrics[metric] = value
        if self.log_metrics['util_loss'] and self.util_loss_batch_size is None:
            self.util_loss_batch_size: int = 2 ** 8
            self.util_loss_grid_size: int = 2 ** 8
        if not self.enable_logging:
            self.save_tb_events_to_csv_aggregate = False
            self.save_tb_events_to_csv_detailed: bool = False
            self.save_tb_events_to_binary_detailed: bool = False
            self.save_models = False
            self.save_figure_to_disk_png = False
            self.save_figure_to_disk_svg = False
            self.save_figure_data_to_disk = False


@dataclass
class HardwareConfig:
    cuda: bool = True
    specific_gpu: int = 0
    fallback: bool = False
    max_cpu_threads: int = 1

    def __post_init__(self):
        if self.cuda and not torch.cuda.is_available():
            warnings.warn('Cuda not available. Falling back to CPU!')
            self.cuda = False
            self.fallback = True
        self.device = 'cuda' if self.cuda else 'cpu'

        if self.cuda and self.specific_gpu is not None:
            torch.cuda.set_device(self.specific_gpu)


@dataclass
class ExperimentConfig:
    experiment_class: str
    running: RunningConfig = None
    setting: SettingConfig = None
    learning: LearningConfig = None
    logging: LoggingConfig = None
    hardware: HardwareConfig = None
