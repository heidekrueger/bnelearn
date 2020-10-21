import os
import sys
import traceback
import torch

sys.path.append(os.path.realpath('.'))
sys.path.append(os.path.join(os.path.expanduser('~'), 'bnelearn'))

#pylint: disable=wrong-import-position
from bnelearn.experiment.configuration_manager import ConfigurationManager


if __name__ == '__main__':
    #pylint: disable=pointless-string-statement
    """
    Runs predefined experiments with interdependencies.
    """

    # User parameters
    log_root_dir = os.path.join(
        os.path.expanduser('~'), 'bnelearn', 'experiments', 'comp_statics'
    )
    specific_gpu = 7
    n_runs = 10
    n_epochs = 1000
    eval_batch_size = 2**23
    model_sharing = False
    risk = 0.1

    def run(experiment_config, experiment_class):
        """Run a experiment class from config"""
        experiment = experiment_class(experiment_config)
        if experiment.known_bne:
            experiment.logging.log_metrics = {
                'opt': True,
                'l2': True,
                'util_loss': True,
                'efficiency': True,
            }
        experiment.logging.log_metrics['efficiency'] = True

        experiment.logging.util_loss_batch_size = 2**10
        experiment.logging.util_loss_grid_size = 2**6
        experiment.logging.util_loss_frequency = 50#n_epochs
        experiment.logging.best_response = True

        try:
            experiment.run()

        except Exception: #pylint: disable=broad-except
            traceback.print_exc()
            torch.cuda.empty_cache()


    ### Run all settings with interdependencies ###
    #LLG
    payment_rules = ['vcg'] #['nearest_zero'] #, 'vcg', 'nearest_bid', 'nearest_vcg']
    corr_models = ['constant_weights'] # 'Bernoulli_weights', 'constant_weights'
    for payment_rule in payment_rules:
        for corr_model in corr_models:
            experiment_config, experiment_class = ConfigurationManager(experiment_type='llg') \
                .with_correlation(gamma=0.5) \
                .get_config(
                    log_root_dir=log_root_dir,
                    n_runs=n_runs,
                    n_epochs=n_epochs,
                    correlation_types=corr_model,
                    payment_rule=payment_rule,
                    specific_gpu=specific_gpu,
                    eval_batch_size=eval_batch_size,
                    model_sharing=model_sharing,
                    # pretrain_iters=10,
                    risk=risk
                )
            run(experiment_config, experiment_class)

    # Mineral rights
    # n_players = 3
    # experiment_config, experiment_class = ConfigurationManager(experiment_type='mineral_rights') \
    #     .get_config(log_root_dir=log_root_dir, n_runs=n_runs, n_epochs=n_epochs,
    #         specific_gpu=specific_gpu,
    #         n_players=n_players,
    #         correlation_groups=[list(range(n_players))],
    #         # pretrain_iters=10
    #     )
    # run(experiment_config, experiment_class)

    # Affiliated observations
    # experiment_config, experiment_class = ConfigurationManager(experiment_type='affiliated_observations') \
    #     .get_config(log_root_dir=log_root_dir, n_runs=n_runs, n_epochs=n_epochs, specific_gpu=specific_gpu,
    #         # pretrain_iters=10
    #     )
    # run(experiment_config, experiment_class)

    # experiment_config, experiment_class = ConfigurationManager(experiment_type='single_item_uniform_symmetric') \
    #    .get_config(log_root_dir=log_root_dir, n_runs=n_runs, n_epochs=n_epochs, specific_gpu=specific_gpu)
    # run(experiment_config, experiment_class)

    ### Run LLG setting for different correlation strengths ###
    # for gamma in [g/10 for g in range(0, 11)]:
    #     experiment_config, experiment_class = ConfigurationManager(experiment_type='llg') \
    #         .with_correlation(gamma=gamma) \
    #         .get_config(
    #             log_root_dir=log_root_dir,
    #             n_runs=n_runs,
    #             n_epochs=n_epochs,
    #             correlation_types='Bernoulli_weights',
    #             eval_batch_size=eval_batch_size,
    #             specific_gpu=specific_gpu,
    #             model_sharing=model_sharing,
    #         )
    #     run(experiment_config, experiment_class)
