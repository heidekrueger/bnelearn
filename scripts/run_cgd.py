"""
Runs predefined experiments with competitive gradient descent
"""
import os
import sys

import torch

# put bnelearn imports after this.
# pylint: disable=wrong-import-position
sys.path.append(os.path.realpath('.'))
sys.path.append(os.path.join(os.path.expanduser('~'), 'bnelearn'))

from bnelearn.experiment.configuration_manager import ConfigurationManager  # pylint: disable=import-error


if __name__ == '__main__':

    # TODO @Christ: This is the main script for staring experiments

    # Path is user-specific
    log_root_dir = os.path.join(
        os.path.expanduser('~'), 'bnelearn', 'experiments', 'cgd'
    )

    learner_types = ["PGLearner",] 
    # learner_types = ["PGLearner", "CGDLearner"]

    for learner_type in learner_types:
        experiment_config, experiment_class = \
            ConfigurationManager(
                experiment_type="single_item_uniform_symmetric",
                n_runs=1,
                n_epochs=1000,
                ) \
            .set_learning(
                learner_type=learner_type,
                mixed_strategy="normal",
                batch_size=2**18,
                pretrain_iters=2000,
                # model_sharing=model_sharing,
                ) \
            .set_hardware(
                specific_gpu=1,
                ) \
            .set_logging(
                eval_batch_size=2**22,
                cache_eval_actions=False,
                util_loss_batch_size=2**10,
                util_loss_grid_size=2**12,
                util_loss_frequency=2000,
                best_response=True,
                log_root_dir=log_root_dir,
                save_tb_events_to_csv_detailed=True,
                plot_frequency=50,
                ) \
            .get_config()
        experiment = experiment_class(experiment_config)
        experiment.run()

        torch.cuda.empty_cache()
