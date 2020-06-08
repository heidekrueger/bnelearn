from typing import List

from bnelearn.experiment.combinatorial_experiment import (LLGExperiment,
                                                          LLLLGGExperiment)
from bnelearn.experiment.configurations import (ExperimentConfiguration,
                                                LearningConfiguration,
                                                LoggingConfiguration,
                                                RunningConfiguration)


# the lists that are defaults will never be mutated, so we're ok with using them here.
# pylint: disable = dangerous-default-value 
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