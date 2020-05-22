from typing import List, Type, Iterable

from tensorflow_core.python.ops import nn
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

class ConfigurationManager:
    def _init_single_item_uniform_symmetric(self):
        self.model_config.model_sharing = True
        self.model_config.u_lo = 0
        self.model_config.u_hi = 1

    def _init_single_item_gaussian_symmetric(self):
        self.model_config.model_sharing = True
        self.model_config.valuation_mean = 15
        self.model_config.valuation_std = 10

    def _init_single_item_asymmetric_uniform_overlapping(self):
        self.model_config.model_sharing = True
        self.model_config.valuation_mean = 15
        self.model_config.valuation_std = 10

        if self.logging_config.eval_batch_size == 2 ** 16:
            print("Using eval_batch_size of 2**16. Use at least 2**22 for proper experiment runs!")

    def _init_single_item_asymmetric_uniform_disjunct(self):
        self.model_config.model_sharing = False
        self.model_config.u_lo = [5, 5]
        self.model_config.u_hi = [15, 25]
        self.model_config.payment_rule = 'first_price'

    def _init_llg(self):
        self.model_config.model_sharing = True
        self.model_config.u_lo = [0, 0, 0]
        self.model_config.u_hi = [1, 1, 2]
        self.running_config.n_players = 3

    def _init_llllgg(self):
        self.model_config.model_sharing = True
        self.model_config.u_lo = [0, 0, 0, 0, 0, 0]
        self.model_config.u_hi = [1, 1, 1, 1, 2, 2]
        self.model_config.core_solver = 'NoCore'
        self.model_config.parallel = 1
        self.running_config.n_players = 6
        self.logging_config.util_loss_frequency = 100
        self.logging_config.log_metrics = ['util_loss']

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
        self.model_config.payment_rule = 'first_price'
        self.model_config.n_units = 2
        self.model_config.model_sharing = True
        self.model_config.u_lo = [1, 1]
        self.model_config.u_hi = [1.4, 1.4]
        self.model_config.constant_marginal_values = False
        self.model_config.efficiency_parameter = 0.3

    experiment_types = {
        'single_item_uniform_symmetric': _init_single_item_uniform_symmetric,
        'single_item_gaussian_symmetric': _init_single_item_gaussian_symmetric,
        'single_item_asymmetric_uniform_overlapping': _init_single_item_asymmetric_uniform_overlapping,
        'single_item_asymmetric_uniform_disjunct': _init_single_item_asymmetric_uniform_disjunct,
        'llg': _init_llg(),
        'llllgg': _init_llllgg(),
        'multiunit': _init_multiunit(),
        'splitaward': _init_splitaward()
    }

    def __init__(self, experiment_type: str):
        self.experiment_type = experiment_type

        # Common defaults
        self.running_config = RunningConfiguration(n_runs=1, n_epochs=5, n_players=2)
        self.model_config = ModelConfiguration(payment_rule='first_price', risk=1.0)
        self.learning_config = LearningConfiguration(optimizer_type='adam',
                                                     pretrain_iters=50,
                                                     batch_size=2 ** 10)
        self.gpu_config = GPUConfiguration(specific_gpu=0, cuda=True)
        self.logging_config = LoggingConfiguration(log_metrics=['opt', 'l2', 'util_loss'],
                                                   util_loss_batch_size=2 ** 4,
                                                   util_loss_grid_size=2 ** 4,
                                                   enable_logging=True,
                                                   eval_batch_size=2 ** 16,
                                                   save_tb_events_to_csv_detailed=False,
                                                   save_tb_events_to_binary_detailed=False,
                                                   stopping_criterion_rel_util_loss_diff=0.001)

        # Setting defaults specific to an experiment type
        if self.experiment_type not in ConfigurationManager.experiment_types:
            raise Exception('The experiment type does not exist')
        else:
            ConfigurationManager.experiment_types[self.experiment_type]()

    # ToDo Expand the list of params to cover all of configs
    def get_config(self, n_runs: int = None, n_epochs: int = None, n_players: int = None, seeds: Iterable[int] = None,
                   payment_rule: str = None, model_sharing=None, risk: float = None,
                   known_bne=False, u_lo=None, u_hi=None, learner_hyperparams: dict = None,
                   optimizer_type: str or Type[Optimizer] = None, optimizer_hyperparams: dict = None,
                   hidden_nodes: List[int] = None, hidden_activations: List[nn.Module] = None,
                   pretrain_iters: int = None, batch_size: int = None,
                   log_metrics=None, util_loss_batch_size=None,
                   util_loss_grid_size=None, specific_gpu=None, cuda=None,
                   save_tb_events_to_csv_detailed=None,
                   save_tb_events_to_binary_detailed=None,
                   stopping_criterion_rel_util_loss_diff=None,
                   logging=None, eval_batch_size=None):

        for arg, v in filter(lambda v: v is not None, locals().values()):
            setattr(self, arg, v)

        experiment_config = ExperimentConfiguration(experiment_class=self.experiment_type,
                                                    run_config=self.running_config,
                                                    model_config=self.model_config,
                                                    learning_config=self.learning_config,
                                                    logging_config=self.logging_config,
                                                    gpu_config=self.gpu_config
                                                    )

        return experiment_config, self.experiment_type
