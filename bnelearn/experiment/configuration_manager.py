from typing import List, Type, Iterable

import torch
import torch.nn as nn
from torch.optim import Optimizer

from bnelearn.experiment.combinatorial_experiment import (LLGExperiment,
                                                          LLLLGGExperiment)
from bnelearn.experiment.configurations import (ModelConfiguration,
                                                LearningConfiguration,
                                                LoggingConfiguration,
                                                RunningConfiguration, ExperimentConfiguration, GPUConfiguration)
from bnelearn.experiment.multi_unit_experiment import (MultiUnitExperiment,
                                                       SplitAwardExperiment)
from bnelearn.experiment.single_item_experiment import (
    GaussianSymmetricPriorSingleItemExperiment,
    TwoPlayerAsymmetricUniformPriorSingleItemExperiment,
    UniformSymmetricPriorSingleItemExperiment)


# the lists that are defaults will never be mutated, so we're ok with using them here.
# pylint: disable = dangerous-default-value

# ToDO Some default parameters are still set inside the inheritors of Experiment, should some of the logig of which
#  parameters go with which class be encapsulated here?

class ConfigurationManager:
    """
    Allows to init any type of experiment with some default values and get an ExperimentConfiguration object
    after selectively changing the attributes
    """

    def _init_single_item_uniform_symmetric(self):
        self.model_config.model_sharing = True
        self.model_config.u_lo = 0
        self.model_config.u_hi = 1

    def _init_single_item_gaussian_symmetric(self):
        self.model_config.model_sharing = True
        self.model_config.valuation_mean = 15
        self.model_config.valuation_std = 10

    def _init_single_item_asymmetric_uniform_overlapping(self):
        self.model_config.model_sharing = False
        self.model_config.u_lo = [5, 5]
        self.model_config.u_hi = [15, 25]

        if self.logging_config.eval_batch_size == 2 ** 16:
            print("Using eval_batch_size of 2**16. Use at least 2**22 for proper experiment runs!")

    def _init_single_item_asymmetric_uniform_disjunct(self):
        self.model_config.model_sharing = False
        self.model_config.u_lo = [0, 6]
        self.model_config.u_hi = [5, 7]

    def _init_llg(self):
        self.model_config.model_sharing = True
        self.model_config.u_lo = [0, 0, 0]
        self.model_config.u_hi = [1, 1, 2]
        self.running_config.n_players = 3
        self.model_config.payment_rule = 'nearest_zero'

    def _init_llllgg(self):
        self.model_config.model_sharing = True
        self.model_config.u_lo = [0, 0, 0, 0, 0, 0]
        self.model_config.u_hi = [1, 1, 1, 1, 2, 2]
        self.model_config.core_solver = 'NoCore'
        self.model_config.parallel = 1
        self.running_config.n_players = 6
        self.logging_config.util_loss_frequency = 100
        # self.logging_config.log_metrics = ['util_loss']
        self.logging_config.log_metrics['util_loss'] = True

    def _init_multiunit(self):
        self.model_config.payment_rule = 'vcg'
        self.model_config.n_units = 2
        self.model_config.model_sharing = True
        self.model_config.u_lo = [0, 0]
        self.model_config.u_hi = [1, 1]
        self.model_config.risk = 1.0
        self.model_config.constant_marginal_values = False
        self.logging_config.plot_points = 1000

    def _init_splitaward(self):
        self.model_config.n_units = 2
        self.model_config.model_sharing = True
        self.model_config.u_lo = [1, 1]
        self.model_config.u_hi = [1.4, 1.4]
        self.model_config.constant_marginal_values = False
        self.model_config.efficiency_parameter = 0.3

    experiment_types = {
        'single_item_uniform_symmetric':
            (_init_single_item_uniform_symmetric, UniformSymmetricPriorSingleItemExperiment),
        'single_item_gaussian_symmetric':
            (_init_single_item_gaussian_symmetric, GaussianSymmetricPriorSingleItemExperiment),
        'single_item_asymmetric_uniform_overlapping':
            (_init_single_item_asymmetric_uniform_overlapping, TwoPlayerAsymmetricUniformPriorSingleItemExperiment),
        'single_item_asymmetric_uniform_disjunct':
            (_init_single_item_asymmetric_uniform_disjunct, TwoPlayerAsymmetricUniformPriorSingleItemExperiment),
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
        self.running_config = RunningConfiguration(n_runs=1, n_epochs=5, n_players=2)
        self.model_config = ModelConfiguration(payment_rule='first_price', risk=1.0)
        self.learning_config = LearningConfiguration(optimizer_type='adam',
                                                     pretrain_iters=10,
                                                     batch_size=2 ** 10)
        self.gpu_config = GPUConfiguration(specific_gpu=0, cuda=True)
        self.logging_config = LoggingConfiguration(util_loss_batch_size=2 ** 4,
                                                   util_loss_grid_size=2 ** 4,
                                                   enable_logging=True,
                                                   eval_batch_size=2 ** 16,
                                                   save_tb_events_to_csv_detailed=False,
                                                   save_tb_events_to_binary_detailed=False,
                                                   stopping_criterion_rel_util_loss_diff=0.001)

        # Defaults specific to an experiment type
        if self.experiment_type not in ConfigurationManager.experiment_types:
            raise Exception('The experiment type does not exist')
        else:
            ConfigurationManager.experiment_types[self.experiment_type][0](self)

    def get_config(self, n_runs: int = None, n_epochs: int = None, n_players: int = None, seeds: Iterable[int] = None,
                   payment_rule: str = None, model_sharing: bool = None, risk: float = None, known_bne: bool = None,
                   common_prior: torch.distributions.Distribution = None, valuation_mean: float = None,
                   valuation_std: float = None, u_lo: list = None, u_hi: list = None, gamma: float = None, n_local=None,
                   n_items: int = None, n_units: int = None, pretrain_transform: callable = None,
                   constant_marginal_values: bool = None, item_interest_limit: int = None,
                   efficiency_parameter: float = None, core_solver: str = None, parallel: int = None,
                   learner_hyperparams: dict = None, optimizer_type: str or Type[Optimizer] = None,
                   optimizer_hyperparams: dict = None, hidden_nodes: List[int] = None,
                   hidden_activations: List[nn.Module] = None, pretrain_iters: int = None, batch_size: int = None,
                   enable_logging: bool = None, log_root_dir: str = None, experiment_name: str = None,
                   experiment_timestamp: str = None, plot_frequency: int = None, plot_points: int = None,
                   plot_show_inline: bool = None, log_metrics: dict = None,
                   stopping_criterion_rel_util_loss_diff: float = None, stopping_criterion_frequency: int = None,
                   stopping_criterion_duration: int = None, stopping_criterion_batch_size: int = None,
                   stopping_criterion_grid_size: int = None, util_loss_batch_size: int = None,
                   util_loss_grid_size: int = None, util_loss_frequency: int = None, eval_batch_size: int = None,
                   cache_eval_action: bool = None,save_tb_events_to_csv_aggregate: bool = None,
                   save_tb_events_to_csv_detailed: bool = None, save_tb_events_to_binary_detailed: bool = None,
                   save_models: bool = None,  save_figure_to_disk_png: bool = None,
                   save_figure_to_disk_svg: bool = None, save_figure_data_to_disk: bool = None,
                   cuda: bool = None, specific_gpu: int = None, fallback: bool = None):
        """
        Allows to selectively override any parameter which was set to default on init
        :return: experiment configuration for the Experiment init, experiment class to run it dynamically
        """

        # If a specific parameter was passed, the corresponding config would be assigned
        for arg, v in {key: value for key, value in locals().items() if key != 'self' and value is not None}.items():
            if hasattr(self.running_config, arg):
                setattr(self.running_config, arg, v)
            elif hasattr(self.model_config, arg):
                setattr(self.model_config, arg, v)
            elif hasattr(self.logging_config, arg):
                setattr(self.logging_config, arg, v)
            elif hasattr(self.learning_config, arg):
                setattr(self.learning_config, arg, v)
            elif hasattr(self.gpu_config, arg):
                setattr(self.gpu_config, arg, v)

        experiment_config = ExperimentConfiguration(experiment_class=self.experiment_type,
                                                    run_config=self.running_config,
                                                    model_config=self.model_config,
                                                    learning_config=self.learning_config,
                                                    logging_config=self.logging_config,
                                                    gpu_config=self.gpu_config
                                                    )

        return experiment_config, ConfigurationManager.experiment_types[self.experiment_type][1]

    @staticmethod
    def get_class_by_experiment_type(experiment_type):
        return ConfigurationManager.experiment_types[experiment_type][1]
