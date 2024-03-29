"""
This module provides dataclasses that are used to hold configs.
The values which are set to None are either not necessary, specific only for certain kinds of experiments or
are set later depending on other values
"""
import dataclasses
import json
from dataclasses import dataclass
from typing import List, Iterable

import torch
import torch.nn as nn


class EnhancedJSONEncoder(json.JSONEncoder):
    def default(self, o):
        if dataclasses.is_dataclass(o):
            return dataclasses.asdict(o)
        return super().default(o)


# Only used on front end
@dataclass
class RunningConfig:
    n_runs: int
    n_epochs: int
    seeds: Iterable[int] = None


@dataclass
class SettingConfig:
    n_players: int
    n_items: int
    payment_rule: str
    risk: float

    # SymmetricPriorSingleItemExperiment
    common_prior: torch.distributions.Distribution = None

    # Gaussian Distribution
    valuation_mean: float = None
    valuation_std: float = None

    # Uniform Distribution
    u_lo: list = None
    u_hi: list = None

    # Correlated Valuations, independent by default
    gamma: float = None
    correlation_types: str = None
    correlation_groups: List[List[int]] = None  # player_ids in each group
    correlation_coefficients: List[float] = None  # coefficients in each group

    # Multi-Unit
    pretrain_transform: callable = None
    constant_marginal_values: bool = False
    item_interest_limit: int = None

    # Split-Award
    efficiency_parameter: float = None

    # LLLLGG
    core_solver: str = None
    # parallel: int = 1 in hardware config now

    # Contests
    tullock_impact_factor: float = None
    impact_function: str = None
    crowdsourcing_values: List = None


@dataclass
class LearningConfig:
    model_sharing: bool
    learner_type: str
    learner_hyperparams: dict
    optimizer_type: str
    optimizer_hyperparams: dict
    scheduler_type: str
    scheduler_hyperparams: dict
    hidden_nodes: List[int]
    pretrain_iters: int
    pretrain_to_bne: None or int
    batch_size: int
    smoothing_temperature: float
    redraw_every_iteration: bool
    mixed_strategy: str
    bias: bool
    hidden_activations: List[nn.Module] = None
    value_contest: bool = True



@dataclass
class LoggingConfig:
    """Controls logging and evaluation aspects of an experiment suite.

    If logging is enabled, the experiment runs will be logged to the following
    directories:

    .. code-block:: bash

        log_root_dir /
            [setting-specific dir hierarchy determined by Experiment subclasses] /
                experiment_timestamp + experiment_name /
                    run_timestamp + run_seed

    """

    enable_logging: bool  # If false, disables ALL logging
    log_root_dir: str

    # Utility Loss calculation
    util_loss_batch_size: int
    util_loss_opponent_batch_size: int
    util_loss_grid_size: int
    eval_frequency: int

    # Eval vs known bne
    eval_batch_size: int
    cache_eval_actions: bool

    plot_frequency: int
    plot_points: int
    plot_show_inline: bool
    log_metrics: dict
    best_response: bool

    save_tb_events_to_csv_aggregate: bool
    save_tb_events_to_csv_detailed: bool
    save_tb_events_to_binary_detailed: bool
    save_models: bool
    log_componentwise_norm: bool

    save_figure_to_disk_png: bool
    save_figure_to_disk_svg: bool
    save_figure_data_to_disk: bool

    export_step_wise_linear_bid_function_size = None
    experiment_dir: str = None
    experiment_name: str = None


@dataclass
class HardwareConfig:
    cuda: bool
    specific_gpu: int
    fallback: bool
    max_cpu_threads: int
    device: str = None


@dataclass
class ExperimentConfig:
    experiment_class: str
    running: RunningConfig
    setting: SettingConfig
    learning: LearningConfig
    logging: LoggingConfig
    hardware: HardwareConfig
