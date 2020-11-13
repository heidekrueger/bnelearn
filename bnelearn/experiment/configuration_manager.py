import json
import os
import time
import warnings
from typing import List, Type, Iterable

import torch
import torch.nn as nn
from torch.optim import Optimizer
from bnelearn.util import logging as logging_utils

from bnelearn.experiment.configurations import (SettingConfig,
                                                LearningConfig,
                                                LoggingConfig,
                                                RunningConfig, ExperimentConfig, HardwareConfig,
                                                EnhancedJSONEncoder)

from bnelearn.experiment.combinatorial_experiment import (LLGExperiment,
                                                          LLLLGGExperiment)
from bnelearn.experiment.multi_unit_experiment import (MultiUnitExperiment, SplitAwardExperiment)

from bnelearn.experiment.single_item_experiment import (GaussianSymmetricPriorSingleItemExperiment,
                                                        TwoPlayerAsymmetricUniformPriorSingleItemExperiment,
                                                        UniformSymmetricPriorSingleItemExperiment,
                                                        MineralRightsExperiment,
                                                        AffiliatedObservationsExperiment)



# the lists that are defaults will never be mutated, so we're ok with using them here.
# pylint: disable = dangerous-default-value

# This module explicitly takes care of unifying lots of variables, it's okay to use many locals here.
# pylint: disable=too-many-instance-attributes



class ConfigurationManager:
    """
    The class provides a 'front-end' for the whole package. It allows for creation of a full and
    consistent ExperimentConfiguration, as defined by the ExperimentConfig dataclass.
    It manages all the defaults, including those specific for each experiment type, auto-inits the parameters that
    are not supposed to be initialized manually, and allows to selectively change any part of the configuration,
    while also performing a parameter and consistency check before creating the final configuration object.

    The workflow with the class API is as follows:
    1. Init class object with the experiment type string, n_runs and n_epochs.
    For possible experiment types see ConfigurationManager.experiment_types
    1.1. __init__ calls get_default_config_members method to get default configuration members.
    1.2 Based on the experiment type, __init__ calls the appropriate ancillary _init_experiment_type.
    It sets the default parameters specific for the given experiment type.
    2. (Optional step) Call set_config_member methods (e.g. set_setting) in a chain style,
    each methods allows to selectively set any parameter of a corresponding config member to a new arbitrary value,
    while leaving all the parameters not specified by the user intact - with their default values.
    3. Call the get_config method to get a ready configuration object and an experiment class corresponding
    to the experiment type (the latter needed for an easy instantiation of the Experiment)
    3.1 get_config calls _post_init, which inits the parameters which shouldn't be set manually, checks for consistency
    between the related parameters and validates whether each parameter is in an appropriate value range.
    Then, it calls the type specific _post_init_experiment_type method which performs all the same things, but specific
    for the experiment type.
    3.2 get_config creates and returns the final and valid configuration object alongside the experiment class.

    Example of class usage:
    experiment_config, experiment_class = ConfigurationManager(experiment_type='multiunit', n_runs=1, n_epochs=20) \
        .set_logging(log_root_dir=log_root_dir) \
        .set_setting(payment_rule='discriminatory') \
        .set_learning(model_sharing=False) \
        .set_hardware() \
        .get_config()

    experiment_class(experiment_config).run()
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
        self.learning.model_sharing = False
        self.setting.u_lo = [5, 5]
        self.setting.u_hi = [15, 25]

    def _init_single_item_asymmetric_uniform_disjunct(self):
        self.learning.model_sharing = False
        self.setting.u_lo = [0, 6]
        self.setting.u_hi = [5, 7]

    def _init_mineral_rights(self):
        self.setting.n_players = 3
        self.logging.log_metrics = {'opt': True,
                                    'l2': True,
                                    'util_loss': True}
        self.setting.correlation_groups = [[0, 1, 2]]
        self.setting.correlation_types = 'corr_type'
        self.setting.correlation_coefficients = [1.0]
        self.setting.u_lo = 0
        self.setting.u_hi = 1
        self.setting.payment_rule = 'second_price'

    def _init_affiliated_observations(self):
        self.running.n_runs = 1
        self.running.n_epochs = 2000
        self.setting.n_players = 2
        self.logging.log_metrics = {'opt': True,
                                    'l2': True,
                                    'util_loss': True}
        self.setting.correlation_groups = [[0, 1]]
        self.setting.correlation_types = 'corr_type'
        self.setting.correlation_coefficients = [1.0]
        self.setting.u_lo = 0
        self.setting.u_hi = 1
        self.setting.payment_rule = 'first_price'

    def _init_llg(self):
        self.learning.model_sharing = True
        self.setting.u_lo = [0, 0, 0]
        self.setting.u_hi = [1, 1, 2]
        self.setting.n_players = 3
        self.setting.payment_rule = 'nearest_zero'
        self.setting.correlation_groups = [[0, 1], [2]]
        self.setting.gamma = 0.0

    #     self.setting.correlation_types = 'independent'
    #
    # def with_correlation(self, gamma, correlation_type='Bernoulli_weights'):
    #     self.setting.gamma = gamma
    #     self.setting.correlation_types = correlation_type if gamma > 0.0 else 'independent'
    #
    #     if correlation_type == 'constant_weights' and gamma > 0:
    #         if 'opt' in self.logging.log_metrics.keys():
    #             del self.logging.log_metrics['opt']
    #         if 'l2' in self.logging.log_metrics.keys():
    #             del self.logging.log_metrics['l2']
    #         print('BNE in constant weights correlation model not approximated.')
    #
    #     return self

    def _init_llllgg(self):
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
        self.setting.n_units = 2
        self.learning.model_sharing = True
        self.setting.u_lo = [1, 1]
        self.setting.u_hi = [1.4, 1.4]
        self.setting.constant_marginal_values = False
        self.setting.efficiency_parameter = 0.3
        self.logging.log_componentwise_norm = True

    def _post_init(self):
        """Any assignments and checks common to all experiment types"""
        # Learning
        assert len(self.learning.hidden_activations) == len(self.learning.hidden_nodes)
        self.learning.optimizer = ConfigurationManager._set_optimizer(self.learning.optimizer_type)

        # Logging
        # Rationale behind timestamp format: should be ordered chronologically but include weekday.
        # Using . instead of : for compatability with Windows
        # Why do we even need the timestamp field?
        # self.logging.experiment_timestamp = time.strftime('%Y-%m-%d %a %H.%M')
        self.logging.experiment_dir = time.strftime('%Y-%m-%d %a %H.%M')
        if self.logging.experiment_name:
            self.logging.experiment_dir += ' ' + str(self.logging.experiment_name)

        valid_log_metrics = ['opt', 'l2', 'util_loss']
        if self.logging.log_metrics is not None:
            for metric in self.logging.log_metrics:
                assert metric in valid_log_metrics, "Metric not known."
            if self.logging.log_metrics['util_loss'] and self.logging.util_loss_batch_size is None:
                self.logging.util_loss_batch_size = 2 ** 8
                self.logging.util_loss_grid_size = 2 ** 8
            if not self.logging.enable_logging:
                self.logging.save_tb_events_to_csv_aggregate = False
                self.logging.save_tb_events_to_csv_detailed = False
                self.logging.save_tb_events_to_binary_detailed = False
            self.logging.save_models = False
            self.logging.save_figure_to_disk_png = False
            self.logging.save_figure_to_disk_svg = False
            self.logging.save_figure_data_to_disk = False

        # Hardware
        if self.hardware.cuda and not torch.cuda.is_available():
            warnings.warn('Cuda not available. Falling back to CPU!')
            self.hardware.cuda = False
            self.hardware.fallback = True
        self.hardware.device = 'cuda' if self.hardware.cuda else 'cpu'

        if self.hardware.cuda and self.hardware.specific_gpu is not None:
            torch.cuda.set_device(self.hardware.specific_gpu)

        ConfigurationManager.experiment_types[self.experiment_type][2](self)

    def _post_init_single_item_uniform_symmetric(self):
        pass

    def _post_init_single_item_gaussian_symmetric(self):
        pass

    def _post_init_single_item_asymmetric_uniform_overlapping(self):
        pass

    def _post_init_single_item_asymmetric_uniform_disjunct(self):
        pass

    def _post_init_mineral_rights(self):
        pass

    def _post_init_affiliated_observations(self):
        pass

    def _post_init_llg(self):
        # How many of those types are there and how do they correspond to gamma values?
        # I might wrongly understand the relationship here
        if self.setting.gamma == 0.0:
            self.setting.correlation_types = 'independent'
        elif self.setting.gamma > 0.0:
            self.setting.correlation_types = 'Bernoulli_weights'
        elif self.setting.gamma > 1.0:
            raise Exception('Wrong gamma')

    def _post_init_llllgg(self):
        pass

    def _post_init_multiunit(self):
        pass

    def _post_init_splitaward(self):
        pass

    experiment_types = {
        'single_item_uniform_symmetric':
            (UniformSymmetricPriorSingleItemExperiment, _init_single_item_uniform_symmetric,
             _post_init_single_item_uniform_symmetric),
        'single_item_gaussian_symmetric':
            (GaussianSymmetricPriorSingleItemExperiment, _init_single_item_gaussian_symmetric,
             _post_init_single_item_gaussian_symmetric),
        'single_item_asymmetric_uniform_overlapping':
            (TwoPlayerAsymmetricUniformPriorSingleItemExperiment, _init_single_item_asymmetric_uniform_overlapping,
             _post_init_single_item_asymmetric_uniform_overlapping),
        'single_item_asymmetric_uniform_disjunct':
            (TwoPlayerAsymmetricUniformPriorSingleItemExperiment, _init_single_item_asymmetric_uniform_disjunct,
             _post_init_single_item_asymmetric_uniform_disjunct),
        'mineral_rights':
            (MineralRightsExperiment, _init_mineral_rights, _post_init_mineral_rights),
        'affiliated_observations':
            (AffiliatedObservationsExperiment, _init_affiliated_observations, _post_init_affiliated_observations),
        'llg':
            (LLGExperiment, _init_llg, _post_init_llg),
        'llllgg':
            (LLLLGGExperiment, _init_llllgg, _post_init_llllgg),
        'multiunit':
            (MultiUnitExperiment, _init_multiunit, _post_init_multiunit),
        'splitaward':
            (SplitAwardExperiment, _init_splitaward, _post_init_splitaward)}

    def __init__(self, experiment_type: str, n_runs: int, n_epochs: int, seeds: Iterable[int] = None):
        self.experiment_type = experiment_type

        # Common defaults
        self.running, self.setting, self.learning, self.logging, self.hardware = \
            ConfigurationManager.get_default_config_members()

        # Defaults specific to an experiment type
        if self.experiment_type not in ConfigurationManager.experiment_types:
            raise Exception('The experiment type does not exist')
        else:
            ConfigurationManager.experiment_types[self.experiment_type][1](self)

        self.running.n_runs = n_runs
        self.running.n_epochs = n_epochs
        self.running.seeds = seeds

    # pylint: disable=too-many-arguments, unused-argument
    def set_setting(self, n_players: int = 'None', payment_rule: str = 'None', risk: float = 'None',
                    common_prior: torch.distributions.Distribution = 'None', valuation_mean: float = 'None',
                    valuation_std: float = 'None', u_lo: list = 'None', u_hi: list = 'None', gamma: float = 'None',
                    correlation_types: str = 'None', correlation_groups: List[List[int]] = 'None',
                    correlation_coefficients: List[float] = 'None', n_units: int = 'None',
                    pretrain_transform: callable = 'None', constant_marginal_values: bool = 'None',
                    item_interest_limit: int = 'None', efficiency_parameter: float = 'None',
                    core_solver: str = 'None'):
        """Sets only the parameters of setting which were passed, returns self"""
        for arg, v in {key: value for key, value in locals().items() if key != 'self' and value is not 'None'}.items():
            if hasattr(self.setting, arg):
                setattr(self.setting, arg, v)
        return self

    # pylint: disable=too-many-arguments, unused-argument
    def set_learning(self, model_sharing: bool = 'None', learner_hyperparams: dict = 'None', optimizer_type: str = 'None',
                     optimizer_hyperparams: dict = 'None', hidden_nodes: List[int] = 'None', pretrain_iters: int = 'None',
                     batch_size: int = 'None', hidden_activations: List[nn.Module] = 'None'):
        """Sets only the parameters of learning which were passed, returns self"""
        for arg, v in {key: value for key, value in locals().items() if key != 'self' and value is not 'None'}.items():
            if hasattr(self.learning, arg):
                setattr(self.learning, arg, v)
        return self

    # pylint: disable=too-many-arguments, unused-argument
    def set_logging(self, enable_logging: bool = 'None', log_root_dir: str = 'None', util_loss_batch_size: int = 'None',
                    util_loss_grid_size: int = 'None', util_loss_frequency: int = 'None', eval_batch_size: int = 'None',
                    cache_eval_action: bool = 'None', plot_frequency: int = 'None', plot_points: int = 'None',
                    plot_show_inline: bool = 'None', log_metrics: dict = 'None',
                    save_tb_events_to_csv_aggregate: bool = 'None', save_tb_events_to_csv_detailed: bool = 'None',
                    save_tb_events_to_binary_detailed: bool = 'None', save_models: bool = 'None',
                    save_figure_to_disk_png: bool = 'None', save_figure_to_disk_svg: bool = 'None',
                    save_figure_data_to_disk: bool = 'None', stopping_criterion_rel_util_loss_diff: float = 'None',
                    stopping_criterion_frequency: int = 'None', stopping_criterion_duration: int = 'None',
                    stopping_criterion_batch_size: int = 'None', stopping_criterion_grid_size: int = 'None',
                    export_step_wise_linear_bid_function_size: bool = 'None',
                    experiment_dir: str = 'None', experiment_name: str = 'None'):
        """Sets only the parameters of logging which were passed, returns self"""
        for arg, v in {key: value for key, value in locals().items() if key != 'self' and value is not 'None'}.items():
            if hasattr(self.logging, arg):
                setattr(self.logging, arg, v)
        return self

    # pylint: disable=too-many-arguments, unused-argument
    def set_hardware(self, cuda: bool = 'None', specific_gpu: int = 'None', fallback: bool = 'None',
                     max_cpu_threads: int = 'None'):
        """Sets only the parameters of hardware which were passed, returns self"""
        for arg, v in {key: value for key, value in locals().items() if key != 'self' and value is not 'None'}.items():
            if hasattr(self.hardware, arg):
                setattr(self.hardware, arg, v)
        return self

    def get_config(self):
        """
        Performs the _post_init, creates and returns the final ExperimentConfig object
        alongside with the appropriate experiment class
        """
        # Post-inits should insure the consistency of all parameters, no configuration changes beyond this point
        self._post_init()

        experiment_config = ExperimentConfig(experiment_class=self.experiment_type,
                                             running=self.running,
                                             setting=self.setting,
                                             learning=self.learning,
                                             logging=self.logging,
                                             hardware=self.hardware)

        return experiment_config, ConfigurationManager.experiment_types[self.experiment_type][0]

    @staticmethod
    def get_class_by_experiment_type(experiment_type: str):
        """Given an experiment type, returns the corresponding experiment class which could be initialized"""
        return ConfigurationManager.experiment_types[experiment_type][0]

    @staticmethod
    def get_default_config_members() -> (RunningConfig, SettingConfig, LearningConfig, LoggingConfig, HardwareConfig):
        """Creates with default parameters and returns members of the ExperimentConfig"""
        running = RunningConfig(n_runs=1, n_epochs=1)
        setting = SettingConfig(n_players=2,
                                payment_rule='first_price',
                                risk=1.0)
        learning = LearningConfig(model_sharing=True,
                                  learner_hyperparams={'population_size': 64,
                                                       'sigma': 1.,
                                                       'scale_sigma_by_model_size': True},
                                  optimizer_type='adam',
                                  optimizer_hyperparams={'lr': 1e-3},
                                  hidden_nodes=[10, 10],
                                  pretrain_iters=500,
                                  batch_size=2 ** 18,
                                  hidden_activations=[nn.SELU(), nn.SELU()])
        logging = LoggingConfig(enable_logging=True,
                                log_root_dir=os.path.join(os.path.expanduser('~'), 'bnelearn', 'experiments'),
                                plot_frequency=100,
                                plot_points=100,
                                plot_show_inline=True,
                                log_metrics={'opt': True,
                                             'l2': True,
                                             'util_loss': True},
                                log_componentwise_norm=False,
                                save_tb_events_to_csv_aggregate=True,
                                save_tb_events_to_csv_detailed=False,
                                save_tb_events_to_binary_detailed=False,
                                save_models=True,
                                save_figure_to_disk_png=True,
                                save_figure_to_disk_svg=True,
                                save_figure_data_to_disk=True,
                                util_loss_batch_size=2 ** 4,
                                util_loss_grid_size=2 ** 4,
                                util_loss_frequency=100,
                                eval_batch_size=2 ** 22,
                                cache_eval_actions=True,
                                stopping_criterion_rel_util_loss_diff=0.001)
        hardware = HardwareConfig(specific_gpu=0,
                                  cuda=True,
                                  fallback=False,
                                  max_cpu_threads=1)

        return running, setting, learning, logging, hardware

    @staticmethod
    def compare_two_experiment_configs(conf1: ExperimentConfig, conf2: ExperimentConfig) -> bool:
        """
        Checks whether two given configurations are identical (deep comparison)
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

    @staticmethod
    def experiment_config_could_be_saved_properly(exp_config: ExperimentConfig) -> bool:
        """
        Tests whether the given config could be serialized and deserialized properly.
        """
        dir_path = os.path.join(os.getcwd(), 'temp')
        file_path = os.path.join(dir_path, logging_utils._configurations_f_name)
        if not os.path.exists(dir_path):
            try:
                os.mkdir(dir_path)
            except OSError:
                print("Creation of the directory %s failed" % dir_path)

        logging_utils.save_experiment_config(experiment_log_dir=dir_path, experiment_configuration=exp_config)
        exp_retrieved_config = ConfigurationManager.load_experiment_config(experiment_log_dir=dir_path)
        ConfigurationManager.get_class_by_experiment_type(exp_retrieved_config.experiment_class)(exp_retrieved_config)

        if os.path.exists(file_path):
            os.remove(file_path)
            os.removedirs(dir_path)

        return ConfigurationManager.compare_two_experiment_configs(exp_config, exp_retrieved_config)

    @staticmethod
    def load_experiment_config(experiment_log_dir=None):
        """
        Retrieves stored configurations from JSON and turns them into ExperimentConfiguration object
        By default creates configuration from the file stored alongside the running script

        :param experiment_log_dir: full path except for the file name, current working directory by default
        :return: ExperimentConfiguration object
        """
        if experiment_log_dir is None:
            experiment_log_dir = os.path.abspath(os.getcwd())
        f_name = os.path.join(experiment_log_dir, logging_utils._configurations_f_name)

        with open(f_name) as json_file:
            experiment_config_as_dict = json.load(json_file)

        running, setting, learning, logging, hardware = ConfigurationManager.get_default_config_members()

        config_set_name_to_obj = {
            'running': running,
            'setting': setting,
            'learning': learning,
            'logging': logging,
            'hardware': hardware
        }

        experiment_config = ExperimentConfig(experiment_class=experiment_config_as_dict['experiment_class'],
                                             running=running,
                                             setting=setting,
                                             learning=learning,
                                             logging=logging,
                                             hardware=hardware)

        # Parse a dictionary retrieved from JSON into ExperimentConfiguration object
        # Attribute assignment pattern: experiment_config.config_group_name.config_group_object_attr = attr_val
        # e.g. experiment_config.run_config.n_runs = experiment_config_as_dict['run_config']['n_runs']
        # config_group_object assignment pattern: experiment_config.config_group_name = config_group_object
        # e.g. experiment_config.run_config = earlier initialised and filled instance of RunningConfiguration class
        experiment_config_as_dict = {k: v for (k, v) in experiment_config_as_dict.items() if
                                     k != 'experiment_class'}.items()
        for config_set_name, config_group_dict in experiment_config_as_dict:
            for config_set_obj_attr, attr_val in config_group_dict.items():
                setattr(config_set_name_to_obj[config_set_name], config_set_obj_attr, attr_val)
            setattr(experiment_config, config_set_name, config_set_name_to_obj[config_set_name])

        # Create hidden activations object based on the loaded string
        # Tested for SELU only
        hidden_activations_methods = {'SELU': lambda: nn.SELU,
                                      'Threshold': lambda: nn.Threshold,
                                      'ReLU': lambda: nn.ReLU,
                                      'RReLU': lambda: nn.RReLU,
                                      'Hardtanh': lambda: nn.Hardtanh,
                                      'ReLU6': lambda: nn.ReLU6,
                                      'Sigmoid': lambda: nn.Sigmoid,
                                      'Hardsigmoid': lambda: nn.Hardsigmoid,
                                      'Tanh': lambda: nn.Tanh,
                                      'ELU': lambda: nn.ELU,
                                      'CELU': lambda: nn.CELU,
                                      'GLU': lambda: nn.GLU,
                                      'GELU': lambda: nn.GELU,
                                      'Hardshrink': lambda: nn.Hardshrink,
                                      'LeakyReLU': lambda: nn.LeakyReLU,
                                      'LogSigmoid': lambda: nn.LogSigmoid,
                                      'Softplus': lambda: nn.Softplus,
                                      'Softshrink': lambda: nn.Softshrink,
                                      'MultiheadAttention': lambda: nn.MultiheadAttention,
                                      'PReLU': lambda: nn.PReLU,
                                      'Softsign': lambda: nn.Softsign,
                                      'Tanhshrink': lambda: nn.Tanhshrink,
                                      'Softmin': lambda: nn.Softmin,
                                      'Softmax': lambda: nn.Softmax,
                                      'Softmax2d': lambda: nn.Softmax2d,
                                      'LogSoftmax': lambda: nn.LogSoftmax, }

        ha = str(experiment_config.learning.hidden_activations).split('()')
        for symb in ['[', ']', ' ', ',']:
            ha = list(map(lambda s: str(s).replace(symb, ''), ha))
        ha = [i for i in ha if i != '']
        ha = [hidden_activations_methods[layer]()() for layer in ha]
        experiment_config.learning.hidden_activations = ha

        if experiment_config.setting.common_prior != 'None':
            # Create common_prior object based on the loaded string
            common_priors = {'Uniform': torch.distributions.Uniform,
                             'Normal': torch.distributions.Normal,
                             'Bernoulli': torch.distributions.Bernoulli,
                             'Beta': torch.distributions.Beta,
                             'Binomial': torch.distributions.Binomial,
                             'Categorical': torch.distributions.Categorical,
                             'Cauchy': torch.distributions.Cauchy,
                             'Chi2': torch.distributions.Chi2,
                             'ContinuousBernoulli': torch.distributions.ContinuousBernoulli,
                             'Dirichlet': torch.distributions.Dirichlet,
                             'Exponential': torch.distributions.Exponential,
                             'FisherSnedecor': torch.distributions.FisherSnedecor,
                             'Gamma': torch.distributions.Gamma,
                             'Geometric': torch.distributions.Geometric,
                             'Gumbel': torch.distributions.Gumbel,
                             'HalfCauchy': torch.distributions.HalfCauchy,
                             'HalfNormal': torch.distributions.HalfNormal,
                             'Independent': torch.distributions.Independent,
                             'Laplace': torch.distributions.Laplace,
                             'LogNormal': torch.distributions.LogNormal,
                             'LogisticNormal': torch.distributions.LogisticNormal,
                             'LowRankMultivariateNormal': torch.distributions.LowRankMultivariateNormal,
                             'Multinomial': torch.distributions.Multinomial,
                             'MultivariateNormal': torch.distributions.MultivariateNormal,
                             'NegativeBinomial': torch.distributions.NegativeBinomial,
                             'OneHotCategorical': torch.distributions.OneHotCategorical,
                             'Pareto': torch.distributions.Pareto,
                             'RelaxedBernoulli': torch.distributions.RelaxedBernoulli,
                             'RelaxedOneHotCategorical': torch.distributions.RelaxedOneHotCategorical,
                             'StudentT': torch.distributions.StudentT,
                             'Poisson': torch.distributions.Poisson,
                             'VonMises': torch.distributions.VonMises,
                             'Weibull': torch.distributions.Weibull
                             }
            distribution = str(experiment_config.setting.common_prior).split('(')[0]
            if distribution == 'Uniform':
                experiment_config.setting.common_prior = common_priors[distribution](experiment_config.setting.u_lo,
                                                                                     experiment_config.setting.u_hi)
            elif distribution == 'Normal':
                experiment_config.setting.common_prior = common_priors[distribution](
                    experiment_config.setting.valuation_mean,
                    experiment_config.setting.valuation_std)
            else:
                raise NotImplementedError

        return experiment_config

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
