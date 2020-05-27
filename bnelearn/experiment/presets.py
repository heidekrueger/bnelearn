from typing import List

from bnelearn.experiment.combinatorial_experiment import (
    LLGExperiment,
    LLLLGGExperiment)
from bnelearn.experiment.configurations import (
    ExperimentConfiguration,
    LearningConfiguration,
    LoggingConfiguration,
    RunningConfiguration)
from bnelearn.experiment import (
    MultiUnitExperiment,
    SplitAwardExperiment,
    CAItemBiddingExperiment,
    GaussianSymmetricPriorSingleItemExperiment,
    TwoPlayerAsymmetricUniformPriorSingleItemExperiment,
    UniformSymmetricPriorSingleItemExperiment)


# the lists that are defaults will never be mutated, so we're ok with using them here.
# pylint: disable = dangerous-default-value 

def single_item_uniform_symmetric(n_runs: int, n_epochs: int,
                                  n_players: List[int], payment_rule: str, model_sharing=True,
                                  u_lo=0, u_hi=1,
                                  risk=1.0,
                                  log_metrics=['opt', 'l2', 'util_loss'], util_loss_batch_size=2 ** 4,
                                  util_loss_grid_size=2 ** 4,
                                  specific_gpu=0,
                                  logging=True):
    running_configuration = RunningConfiguration(n_runs=n_runs, n_epochs=n_epochs, specific_gpu=specific_gpu,
                                                 n_players=n_players)
    logging_configuration = LoggingConfiguration(log_metrics=log_metrics,
                                                 util_loss_batch_size=util_loss_batch_size,
                                                 util_loss_grid_size=util_loss_grid_size,
                                                 enable_logging=logging
                                                 )

    experiment_configuration = ExperimentConfiguration(payment_rule=payment_rule, model_sharing=model_sharing,
                                                       u_lo=u_lo, u_hi=u_hi, risk=risk)
    experiment_class = UniformSymmetricPriorSingleItemExperiment
    return running_configuration, logging_configuration, experiment_configuration, experiment_class


def single_item_gaussian_symmetric(n_runs: int, n_epochs: int,
                                   n_players: [int], payment_rule: str, model_sharing=True, valuation_mean=15,
                                   valuation_std=10,
                                   risk=1.0, eval_batch_size=2 ** 16,
                                   log_metrics=['opt', 'l2', 'util_loss'], util_loss_batch_size=2 ** 8,
                                   util_loss_grid_size=2 ** 8,
                                   specific_gpu=1,
                                   logging=True,
                                   save_tb_events_to_csv_detailed: bool = False,
                                   save_tb_events_to_binary_detailed: bool = False):
    if eval_batch_size == 2 ** 16:
        print("Using eval_batch_size of 2**16. Use at least 2**22 for proper experiment runs!")
    running_configuration = RunningConfiguration(n_runs=n_runs, n_epochs=n_epochs, specific_gpu=specific_gpu,
                                                 n_players=n_players)
    logging_configuration = LoggingConfiguration(log_metrics=log_metrics,
                                                 util_loss_batch_size=util_loss_batch_size,
                                                 util_loss_grid_size=util_loss_grid_size,
                                                 eval_batch_size=eval_batch_size,
                                                 enable_logging=logging,
                                                 save_tb_events_to_csv_detailed=save_tb_events_to_csv_detailed,
                                                 save_tb_events_to_binary_detailed=save_tb_events_to_binary_detailed
                                                 )
    experiment_configuration = ExperimentConfiguration(payment_rule=payment_rule, model_sharing=model_sharing,
                                                       valuation_mean=valuation_mean, valuation_std=valuation_std,
                                                       risk=risk)
    experiment_class = GaussianSymmetricPriorSingleItemExperiment
    return running_configuration, logging_configuration, experiment_configuration, experiment_class


def single_item_asymmetric_uniform_overlapping(
        n_runs: int,
        n_epochs: int,
        payment_rule='first_price',
        model_sharing=False,
        u_lo=[5, 5],
        u_hi=[15, 25],
        risk=1.0,
        eval_batch_size=2 ** 18,
        log_metrics=['opt', 'l2', 'util_loss'],
        util_loss_batch_size=2 ** 8,
        util_loss_grid_size=2 ** 8,
        specific_gpu=1,
        logging=True
    ):
    n_players = [2]
    running_configuration = RunningConfiguration(n_runs=n_runs, n_epochs=n_epochs,
                                                 specific_gpu=specific_gpu, n_players=n_players)
    logging_configuration = LoggingConfiguration(log_metrics=log_metrics,
                                                 util_loss_batch_size=util_loss_batch_size,
                                                 util_loss_grid_size=util_loss_grid_size,
                                                 eval_batch_size=eval_batch_size
                                                 )

    experiment_configuration = ExperimentConfiguration(payment_rule=payment_rule, model_sharing=model_sharing,
                                                       u_lo=u_lo, u_hi=u_hi, risk=risk)
    experiment_class = TwoPlayerAsymmetricUniformPriorSingleItemExperiment
    return running_configuration, logging_configuration, experiment_configuration, experiment_class


def single_item_asymmetric_uniform_disjunct(
        n_runs: int,
        n_epochs: int,
        payment_rule='first_price',
        model_sharing=False,
        u_lo=[0, 6],  # [5, 5],     [0, 6]
        u_hi=[5, 7],  # [15, 25],   [5, 7]
        risk=1.0,
        eval_batch_size=2 ** 18,
        log_metrics=['opt', 'l2', 'util_loss', 'PoA'],
        util_loss_batch_size=2 ** 8,
        util_loss_grid_size=2 ** 8,
        specific_gpu=1,
        logging=True
    ):
    n_players = [2]
    running_configuration = RunningConfiguration(n_runs=n_runs, n_epochs=n_epochs,
                                                 specific_gpu=specific_gpu, n_players=n_players)
    logging_configuration = LoggingConfiguration(log_metrics=log_metrics,
                                                 util_loss_batch_size=util_loss_batch_size,
                                                 util_loss_grid_size=util_loss_grid_size,
                                                 eval_batch_size=eval_batch_size
                                                 )

    experiment_configuration = ExperimentConfiguration(payment_rule=payment_rule, model_sharing=model_sharing,
                                                       u_lo=u_lo, u_hi=u_hi, risk=risk)
    experiment_class = TwoPlayerAsymmetricUniformPriorSingleItemExperiment
    return running_configuration, logging_configuration, experiment_configuration, experiment_class


def llg(n_runs: int, n_epochs: int,
        payment_rule: str, model_sharing=True,
        u_lo=[0, 0, 0], u_hi=[1, 1, 2],
        risk=1.0,
        log_metrics=['opt', 'l2', 'util_loss'], util_loss_batch_size=2 ** 8, util_loss_grid_size=2 ** 8,
        specific_gpu=1,
        logging=True):
    n_players = [3]
    running_configuration = RunningConfiguration(n_runs=n_runs, n_epochs=n_epochs, specific_gpu=specific_gpu,
                                                 n_players=n_players)
    logging_configuration = LoggingConfiguration(log_metrics=log_metrics,
                                                 util_loss_batch_size=util_loss_batch_size,
                                                 util_loss_grid_size=util_loss_grid_size,
                                                 enable_logging=logging
                                                 )

    experiment_configuration = ExperimentConfiguration(payment_rule=payment_rule, model_sharing=model_sharing,
                                                       u_lo=u_lo, u_hi=u_hi, risk=risk)
    experiment_class = LLGExperiment
    return running_configuration, logging_configuration, experiment_configuration, experiment_class


def llllgg(n_runs: int, n_epochs: int,
           payment_rule: str, model_sharing=True,
           u_lo=[0, 0, 0, 0, 0, 0], u_hi=[1, 1, 1, 1, 2, 2],
           risk=1.0, eval_batch_size=2 ** 12, util_loss_frequency=100,
           log_metrics=['util_loss'], util_loss_batch_size=2 ** 12, util_loss_grid_size=2 ** 10,
           core_solver="NoCore", parallel = 1,
           specific_gpu=1,
           logging=True):
    n_players = [6]
    running_configuration = RunningConfiguration(n_runs=n_runs, n_epochs=n_epochs, specific_gpu=specific_gpu,
                                                 n_players=n_players)
    logging_configuration = LoggingConfiguration(log_metrics=log_metrics,
                                                 util_loss_batch_size=util_loss_batch_size,
                                                 util_loss_grid_size=util_loss_grid_size,
                                                 eval_batch_size=eval_batch_size,
                                                 enable_logging=logging,
                                                 util_loss_frequency=util_loss_frequency
                                                 )

    experiment_configuration = ExperimentConfiguration(payment_rule=payment_rule, model_sharing=model_sharing,
                                                       u_lo=u_lo, u_hi=u_hi, risk=risk, core_solver=core_solver,
                                                       parallel = parallel)
    experiment_class = LLLLGGExperiment
    return running_configuration, logging_configuration, experiment_configuration, experiment_class


def multiunit(
        n_runs: int, n_epochs: int,
        n_players: list = [2],
        payment_rule: str = 'vcg',
        n_units=2,
        log_metrics=['opt', 'l2', 'util_loss'],
        model_sharing=True,
        u_lo=[0, 0], u_hi=[1, 1],
        risk=1.0,
        constant_marginal_values: bool = False,
        item_interest_limit: int = None,
        util_loss_batch_size=2 ** 8,
        util_loss_grid_size=2 ** 8,
        specific_gpu=0,
        logging=True
    ):
    running_configuration = RunningConfiguration(
        n_runs=n_runs, n_epochs=n_epochs,
        specific_gpu=specific_gpu, n_players=n_players
    )
    logging_configuration = LoggingConfiguration(
        log_metrics=log_metrics,
        util_loss_batch_size=util_loss_batch_size,
        util_loss_grid_size=util_loss_grid_size,
        plot_points=1000,
        enable_logging=logging
    )
    experiment_configuration = ExperimentConfiguration(
        payment_rule=payment_rule, n_units=n_units,
        model_sharing=model_sharing,
        u_lo=u_lo, u_hi=u_hi, risk=risk,
        constant_marginal_values=constant_marginal_values,
        item_interest_limit=item_interest_limit
    )
    experiment_class = MultiUnitExperiment
    return running_configuration, logging_configuration, experiment_configuration, experiment_class


def splitaward(
        n_runs: int, n_epochs: int,
        n_players: list = [2],
        payment_rule: str = 'first_price',
        n_units=2,
        model_sharing=True,
        log_metrics=['opt', 'l2', 'util_loss'],
        u_lo=[1, 1], u_hi=[1.4, 1.4],
        risk=1.0,
        constant_marginal_values: bool = False,
        item_interest_limit: int = None,
        efficiency_parameter: float = 0.3,
        util_loss_batch_size=2 ** 8,
        util_loss_grid_size=2 ** 8,
        specific_gpu=1,
        logging=True
    ):
    running_configuration = RunningConfiguration(
        n_runs=n_runs, n_epochs=n_epochs,
        specific_gpu=specific_gpu, n_players=n_players
    )
    logging_configuration = LoggingConfiguration(
        log_metrics=log_metrics,
        util_loss_batch_size=util_loss_batch_size,
        util_loss_grid_size=util_loss_grid_size,
        enable_logging=logging
    )

    experiment_configuration = ExperimentConfiguration(
        payment_rule=payment_rule, n_units=n_units,
        model_sharing=model_sharing,
        u_lo=u_lo, u_hi=u_hi, risk=risk,
        constant_marginal_values=constant_marginal_values,
        item_interest_limit=item_interest_limit,
        efficiency_parameter=efficiency_parameter
    )
    experiment_class = SplitAwardExperiment
    return running_configuration, logging_configuration, experiment_configuration, experiment_class


def itembidding(
        n_runs: int, n_epochs: int,
        n_players: list = [2],
        payment_rule: str = 'vcg',
        n_items=2,
        log_metrics=['PoA'],
        model_sharing=True,
        u_lo=[0], u_hi=[1],
        risk=1.0,
        util_loss_batch_size=2 ** 8,
        util_loss_grid_size=2 ** 8,
        specific_gpu=0,
        logging=True
    ):
    if len(u_lo) < n_players[0]:
        u_lo = [u_lo[0]] * n_players[0]
        u_hi = [u_hi[0]] * n_players[0]
    running_configuration = RunningConfiguration(
        n_runs=n_runs, n_epochs=n_epochs,
        specific_gpu=specific_gpu, n_players=n_players
    )
    logging_configuration = LoggingConfiguration(
        log_metrics=log_metrics,
        util_loss_batch_size=util_loss_batch_size,
        util_loss_grid_size=util_loss_grid_size,
        plot_points=1000,
        enable_logging=logging
    )
    experiment_configuration = ExperimentConfiguration(
        payment_rule=payment_rule, n_units=n_items,
        model_sharing=model_sharing,
        u_lo=u_lo, u_hi=u_hi, risk=risk
    )
    experiment_class = CAItemBiddingExperiment
    return running_configuration, logging_configuration, experiment_configuration, experiment_class
