import os
import sys
import torch

# pylint: disable=wrong-import-position
sys.path.append(os.path.realpath("."))
sys.path.append(os.path.join(os.path.expanduser("~"), "bnelearn"))
# pylint: disable=import-error
from bnelearn.experiment.configuration_manager import ConfigurationManager


from optimizable_experiment import OptimizableExperiment


try:
    # the full logs for the experiments, not the bayes_opt log
    log_root_dir = os.path.join(os.path.expanduser("~"), "bnelearn", "experiments")

    # n_runs here means # of seeds
    experiment_config, experiment_class = (
        ConfigurationManager(
            experiment_type="single_item_uniform_symmetric", n_runs=1, n_epochs=10
        )
        .set_setting(risk=1.1)
        .set_logging(log_root_dir=log_root_dir, save_tb_events_to_csv_detailed=True)
        .set_learning(pretrain_iters=5)
        .set_logging(eval_batch_size=2 ** 22)
        .set_hardware(specific_gpu=7)
        .get_config()
    )

    #TODO Adam hp's (maybe)
    #TODO multiple GPUs
    experiment = OptimizableExperiment(
        experiment_class=experiment_class,
        experiment_config=experiment_config,
        hp_bounds={"lr": (0.0001, 0.01)}, #TODO add transforms to the log space
        metric_name="epsilon_relative",
        n_runs_budget=5,
        minimize=True,
        verbose=True,
        log=True,
    )
    experiment.optimize(init_points=3)
    print(os.getcwd())

except KeyboardInterrupt:
    print("\nKeyboardInterrupt: released memory after interruption")
    torch.cuda.empty_cache()

