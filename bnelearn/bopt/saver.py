import os
import sys
import torch
import numpy as np
import json

# pylint: disable=wrong-import-position
sys.path.append(os.path.realpath("."))
sys.path.append(os.path.join(os.path.expanduser("~"), "bnelearn"))

from bnelearn.experiment.configuration_manager import (
    ConfigurationManager,
)  # pylint: disable=import-error

saved_results = {
    "directory": "",
    "experiment_type": "",
    "hyperparameters": [],
    "reults": {},
}

# TODO rework the saver, it should just receive the results, not run experiments
class Saver:
    """
    Accumulates and aggregates resuts from multiple experiments of the same type but with different hyperparameters. Saves them on request. 
    Seeds averaged out. In a single json only single hyperparameter varies, but all other fixed are logged
    """

    def __init__(self):
        """        
        n_runs : this is the number of full runs, actual number of runs (learning procedures) would be n_runs * # of seeds per run
        """

        self.log = {
            "dir": [],
            "n_seeds": [],
            "epoch": [],
            "lr": [],
            "avrg_eps_rel": [],
            "eps_rel_var": [],
        }

    def add_experiment_result():
        pass

    def _add_basic_run_info(self):
        log["dir"].append(experiment_config.logging.experiment_dir)

    def _add_single_run_result(self, single_run_res):
        eps_rel = []
        for metric_entry in single_run_res:
            if metric_entry[2] == "eval/epsilon_relative":
                print(metric_entry)
                eps_rel.append(metric_entry[4])

            np.array(eps_rel)
            print(np.average(eps_rel))

    def _save_results(self):
        pass


if __name__ == "__main__":
    log_root_dir = os.path.join(os.path.expanduser("~"), "bnelearn", "experiments")

    # n_runs here means # of seeds
    experiment_config, experiment_class = (
        ConfigurationManager(
            experiment_type="single_item_uniform_symmetric", n_runs=2, n_epochs=3
        )
        .set_setting(risk=1.1)
        .set_logging(log_root_dir=log_root_dir, save_tb_events_to_csv_detailed=True)
        .set_learning(pretrain_iters=5)
        .set_logging(eval_batch_size=2 ** 22)
        .set_hardware(specific_gpu=7)
        .get_config()
    )

    saver = Saver(experiment_config, experiment_class)
    saver.run()

    # result = result.to_dict()
    # print(json.dumps(result, sort_keys=False, indent=4))
    # result = result.to_numpy()
    # print(result)

    # if ConfigurationManager.experiment_config_could_be_saved_properly(experiment_config):
    #        pass
    # else:
    #    raise Exception('Unable to perform the correct serialization')

