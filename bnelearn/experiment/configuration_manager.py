import json
import os
from typing import List, Type, Iterable

from dataclasses import replace

import torch
import torch.nn as nn
from torch.optim import Optimizer

from bnelearn.experiment.combinatorial_experiment import (LLGExperiment,
                                                          LLLLGGExperiment)
from bnelearn.experiment.configurations import (SettingConfig,
                                                LearningConfig,
                                                LoggingConfig,
                                                RunningConfig, ExperimentConfig, HardwareConfig, EnhancedJSONEncoder)
from bnelearn.experiment.multi_unit_experiment import (MultiUnitExperiment,
                                                       SplitAwardExperiment)
from bnelearn.experiment.single_item_experiment import (
    GaussianSymmetricPriorSingleItemExperiment,
    TwoPlayerAsymmetricUniformPriorSingleItemExperiment,
    UniformSymmetricPriorSingleItemExperiment,
    MineralRightsExperiment)

# the lists that are defaults will never be mutated, so we're ok with using them here.
# pylint: disable = dangerous-default-value

# ToDO Some default parameters are still set inside the inheritors of Experiment, should some of the loging of which
#  parameters go with which class be encapsulated here?
from bnelearn.util import logging


class ConfigurationManager:
    """
    Allows to init any type of experiment with some default values and get an ExperimentConfiguration object
    after selectively changing the attributes
    """

    def _init_single_item_uniform_symmetric(self):
        self.learning.model_sharing = True
        self.setting.u_lo = 0
        self.setting.u_hi = 1

    def _init_single_item_gaussian_symmetric(self):
        self.learning.model_sharing = True
        self.setting.valuation_mean = 15
        self.setting.valuation_std = 5

    def _init_single_item_asymmetric_uniform_overlapping(self):
        self.running.n_runs = 1
        self.running.n_epochs = 500
        self.learning.model_sharing = False
        self.setting.u_lo = [5, 5]
        self.setting.u_hi = [15, 25]

    def _init_single_item_asymmetric_uniform_disjunct(self):
        self.learning.model_sharing = False
        self.setting.u_lo = [0, 6]
        self.setting.u_hi = [5, 7]

    def _init_mineral_rights(self):
        self.running.n_runs = 1
        self.running.n_epochs = 2000
        self.setting.n_players = 3
        self.logging.log_metrics = {'opt': True,
                                    'l2': True,
                                    'util_loss': False}
        self.setting.correlation_groups = [[0, 1, 2]]
        self.setting.correlation_types = 'corr_type'
        self.setting.correlation_coefficients = [1.0]
        self.setting.u_lo = 0
        self.setting.u_hi = 1

    def _init_llg(self):
        self.running.n_runs = 1
        self.running.n_epochs = 1000
        self.learning.model_sharing = True
        self.setting.u_lo = [0, 0, 0]
        self.setting.u_hi = [1, 1, 2]
        self.setting.n_players = 3
        self.setting.payment_rule = 'nearest_zero'

        self.setting.correlation_groups = [[0, 1], [2]]
        self.setting.correlation_types = 'independent'

    def with_correlation(self, gamma, correlation_type = 'Bernoulli_weights'):

        self.setting.gamma = gamma
        self.setting.correlation_types = correlation_type if gamma > 0.0 else 'independent'

        return self

    def _init_llllgg(self):
        self.running.n_runs = 1
        self.running.n_epochs = 20000
        self.logging.util_loss_batch_size = 2 ** 12
        self.learning.model_sharing = True
        self.setting.u_lo = [0, 0, 0, 0, 0, 0]
        self.setting.u_hi = [1, 1, 1, 1, 2, 2]
        self.setting.core_solver = 'NoCore'
        self.setting.parallel = 1
        self.setting.n_players = 6
        self.logging.util_loss_frequency = 1000  # Or 100?
        self.logging.log_metrics = {'opt': False,
                                    'l2': False,
                                    'util_loss': True}

    def _init_multiunit(self):
        self.setting.payment_rule = 'vcg'
        self.setting.n_units = 2
        self.learning.model_sharing = True
        self.setting.u_lo = [0, 0]
        self.setting.u_hi = [1, 1]
        self.setting.risk = 1.0
        self.setting.constant_marginal_values = False
        self.logging.plot_points = 1000

    def _init_splitaward(self):
        self.running.n_runs = 1
        self.setting.n_units = 2
        self.learning.model_sharing = True
        self.setting.u_lo = [1, 1]
        self.setting.u_hi = [1.4, 1.4]
        self.setting.constant_marginal_values = False
        self.setting.efficiency_parameter = 0.3
        self.logging.log_componentwise_norm = True

    experiment_types = {
        'single_item_uniform_symmetric':
            (_init_single_item_uniform_symmetric, UniformSymmetricPriorSingleItemExperiment),
        'single_item_gaussian_symmetric':
            (_init_single_item_gaussian_symmetric, GaussianSymmetricPriorSingleItemExperiment),
        'single_item_asymmetric_uniform_overlapping':
            (_init_single_item_asymmetric_uniform_overlapping, TwoPlayerAsymmetricUniformPriorSingleItemExperiment),
        'single_item_asymmetric_uniform_disjunct':
            (_init_single_item_asymmetric_uniform_disjunct, TwoPlayerAsymmetricUniformPriorSingleItemExperiment),
        'mineral_rights': (_init_mineral_rights, MineralRightsExperiment),
        'llg':
            (_init_llg, LLGExperiment),
        'llllgg':
            (_init_llllgg, LLLLGGExperiment),
        'multiunit':
            (_init_multiunit, MultiUnitExperiment),
        'splitaward':
            (_init_splitaward, SplitAwardExperiment)
    }

    def __init__(self, experiment_type: str):
        self.experiment_type = experiment_type

        # Common defaults
        self.running = RunningConfig(n_runs=1, n_epochs=100)
        self.setting = SettingConfig(payment_rule='first_price', risk=1.0, n_players=2)
        self.learning = LearningConfig(optimizer_type='adam',
                                       pretrain_iters=500,
                                       batch_size=2 ** 18,
                                       model_sharing=True)
        self.hardware = HardwareConfig(specific_gpu=0, cuda=True)
        self.logging = LoggingConfig(util_loss_batch_size=2 ** 4,
                                     util_loss_grid_size=2 ** 4,
                                     log_metrics={'opt': True,
                                                  'l2': True,
                                                  'util_loss': True},
                                     enable_logging=True,
                                     eval_batch_size=2 ** 22,
                                     save_tb_events_to_csv_detailed=False,
                                     save_tb_events_to_binary_detailed=False,
                                     stopping_criterion_rel_util_loss_diff=None)


        # Defaults specific to an experiment type
        if self.experiment_type not in ConfigurationManager.experiment_types:
            raise Exception('The experiment type does not exist')
        else:
            ConfigurationManager.experiment_types[self.experiment_type][0](self)

    # pylint: disable=too-many-arguments, unused-argument
    def get_config(self, n_runs: int = None, n_epochs: int = None, n_players: int = None, seeds: Iterable[int] = None,
                   payment_rule: str = None, risk: float = None,
                   common_prior: torch.distributions.Distribution = None, valuation_mean: float = None,
                   valuation_std: float = None, u_lo: list = None, u_hi: list = None,
                   correlation_types: str = None, correlation_groups: List[List[int]] = None,
                   correlation_coefficients: List[float] = None, n_units: int = None,
                   pretrain_transform: callable = None, constant_marginal_values: bool = None,
                   item_interest_limit: int = None, efficiency_parameter: float = None,
                   core_solver: str = None, gamma: float = None, model_sharing: bool = None,
                   learner_hyperparams: dict = None, optimizer_type: str or Type[Optimizer] = None,
                   optimizer_hyperparams: dict = None, hidden_nodes: List[int] = None,
                   hidden_activations: List[nn.Module] = None, pretrain_iters: int = None, batch_size: int = None,
                   enable_logging: bool = None, log_root_dir: str = None, experiment_name: str = None,
                   experiment_timestamp: str = None, plot_frequency: int = None, plot_points: int = None,
                   plot_show_inline: bool = None, log_metrics: dict = None, log_componentwise_norm: bool = None,
                   export_step_wise_linear_bid_function_size: float = None,
                   stopping_criterion_rel_util_loss_diff: float = None, stopping_criterion_frequency: int = None,
                   stopping_criterion_duration: int = None, stopping_criterion_batch_size: int = None,
                   stopping_criterion_grid_size: int = None, util_loss_batch_size: int = None,
                   util_loss_grid_size: int = None, util_loss_frequency: int = None, eval_batch_size: int = None,
                   cache_eval_action: bool = None, save_tb_events_to_csv_aggregate: bool = None,
                   save_tb_events_to_csv_detailed: bool = None, save_tb_events_to_binary_detailed: bool = None,
                   save_models: bool = None, save_figure_to_disk_png: bool = None,
                   save_figure_to_disk_svg: bool = None, save_figure_data_to_disk: bool = None,
                   cuda: bool = None, specific_gpu: int = None, fallback: bool = None, max_cpu_threads: int = None):
        """
        Allows to selectively override any parameter which was set to default on init
        :return: experiment configuration for the Experiment init, experiment class to run it dynamically
        """

        # If a specific parameter was passed, the corresponding config would be assigned
        # TODO: Stefan: What if we explicitly want to change a value to None?
        for arg, v in {key: value for key, value in locals().items() if key != 'self' and value is not None}.items():
            if hasattr(self.running, arg):
                self.running = replace(self.running, **{arg:v})
            elif hasattr(self.setting, arg):
                self.setting = replace(self.setting, **{arg:v})
            elif hasattr(self.logging, arg):
                self.logging = replace(self.logging, **{arg:v})
            elif hasattr(self.learning, arg):
                self.learning = replace(self.learning, **{arg:v})
            elif hasattr(self.hardware, arg):
                self.hardware = replace(self.hardware, **{arg:v})

        experiment_config = ExperimentConfig(experiment_class=self.experiment_type,
                                             running=self.running,
                                             setting=self.setting,
                                             learning=self.learning,
                                             logging=self.logging,
                                             hardware=self.hardware
                                             )

        return experiment_config, ConfigurationManager.experiment_types[self.experiment_type][1]

    @staticmethod
    def get_class_by_experiment_type(experiment_type):
        return ConfigurationManager.experiment_types[experiment_type][1]

    @staticmethod
    def compare_two_experiment_configs(conf1: ExperimentConfig, conf2: ExperimentConfig) -> bool:
        """
        Checks whether two given configurations are identical
        """
        if str(conf1.setting.common_prior) != str(conf2.setting.common_prior) \
                or str(conf1.learning.hidden_activations) != str(conf2.learning.hidden_activations):
            return False

        temp_cp = conf1.setting.common_prior
        temp_ha = conf1.learning.hidden_activations

        conf1.setting.common_prior = str(conf1.setting.common_prior)
        conf1.learning.hidden_activations = str(conf1.learning.hidden_activations)
        conf2.setting.common_prior = str(conf2.setting.common_prior)
        conf2.learning.hidden_activations = str(conf2.learning.hidden_activations)

        c1 = json.dumps(conf1, cls=EnhancedJSONEncoder)
        c2 = json.dumps(conf2, cls=EnhancedJSONEncoder)

        # To prevent compromising the objects
        conf1.setting.common_prior = temp_cp
        conf1.learning.hidden_activations = temp_ha
        conf2.setting.common_prior = temp_cp
        conf2.learning.hidden_activations = temp_ha

        return c1 == c2

    # It is here and not in logging because logging can't depend on Experiments (while Experiments depend on logging)
    @staticmethod
    def experiment_config_could_be_serialized_properly(exp_config: ExperimentConfig) -> bool:
        """
        Tests whether the given config could be serialized and deserialized properly
        """
        dir_path = os.path.join(os.getcwd(), 'temp')
        file_path = os.path.join(dir_path, 'temp_config.json')
        if not os.path.exists(dir_path):
            try:
                os.mkdir(dir_path)
            except OSError:
                print("Creation of the directory %s failed" % dir_path)

        logging.log_experiment_configurations(experiment_log_dir=dir_path, experiment_configuration=exp_config)
        exp_retrieved_config = logging.get_experiment_config_from_configurations_log(experiment_log_dir=dir_path)
        ConfigurationManager.get_class_by_experiment_type(exp_retrieved_config.experiment_class)(exp_retrieved_config)

        if os.path.exists(file_path):
            os.remove(file_path)
            os.removedirs(dir_path)

        return ConfigurationManager.compare_two_experiment_configs(exp_config, exp_retrieved_config)
