"""
Runs predefined experiments with individual parameters
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

    # Path is user-specific
    log_root_dir = os.path.join(
        os.path.expanduser('~'), 'bnelearn', 'experiments', 'mixed-strategies')

    for learner_type in ['ReinforceLearner', 'ESPGLearner']:
        for model_sharing in [True, False]:
            log_dir = os.path.join(log_root_dir, learner_type)

            print(f"\n\nRunning: {learner_type} with {model_sharing}")

            # Set up experiment
            experiment_config, experiment_class = \
                ConfigurationManager(
                    experiment_type="single_item_uniform_symmetric",
                    n_runs=1,
                    n_epochs=10,
                    ) \
                .set_learning(
                    learner_type=learner_type,
                    batch_size=2**17,
                    pretrain_iters=500,
                    mixed_strategy='normal',
                    model_sharing=model_sharing,
                    ) \
                .set_hardware(
                    specific_gpu=1,
                    ) \
                .set_logging(
                    eval_batch_size=2**22,
                    cache_eval_actions=False,
                    util_loss_batch_size=2**10,
                    util_loss_grid_size=2**12,
                    util_loss_frequency=100,
                    best_response=True,
                    log_root_dir=log_dir,
                    save_tb_events_to_csv_detailed=True,
                    plot_frequency=20,
                    ) \
                .get_config()
            experiment = experiment_class(experiment_config)
            experiment.run()

    torch.cuda.empty_cache()
