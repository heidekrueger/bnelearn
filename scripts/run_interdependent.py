"""
Script for running experiments that specifically adress dependencies within
the games.
"""
import os
import sys
import traceback
import torch

sys.path.append(os.path.realpath('.'))
sys.path.append(os.path.join(os.path.expanduser('~'), 'bnelearn'))

# pylint: disable=wrong-import-position
from bnelearn.experiment.configuration_manager import ConfigurationManager


if __name__ == '__main__':
    # pylint: disable=pointless-string-statement
    """Runs predefined experiments with interdependencies."""

    # User parameters
    specific_gpu = 7
    n_runs = 1
    n_epochs = 1000
    eval_batch_size = 2**23
    model_sharing = False
    pretrain_iters = None

    def run(experiment_config, experiment_class): # pylint: disable=redefined-outer-name
        """Run a experiment class from config"""
        experiment = experiment_class(experiment_config)
        if experiment.known_bne:
            experiment.logging.log_metrics = {
                'opt': True,
                'l2': True,
                'util_loss': True
            }
        experiment.logging.log_metrics['efficiency'] = True

        experiment.logging.util_loss_batch_size = 2**12
        experiment.logging.util_loss_grid_size = 2**10
        experiment.logging.util_loss_frequency = n_epochs
        experiment.logging.best_response = True

        try:
            experiment.run()

        except Exception: # pylint: disable=broad-except
            traceback.print_exc()
            torch.cuda.empty_cache()


    ### Run all settings with interdependencies ###############################
    # log_root_dir = os.path.join(
    #     os.path.expanduser('~'), 'bnelearn', 'experiments', 'interdependence',
    #     'individual_experiments'
    # )

    # # LLG
    # payment_rules = ['nearest_zero', 'vcg', 'nearest_bid', 'nearest_vcg']
    # corr_models = ['Bernoulli_weights', 'constant_weights']
    # for payment_rule in payment_rules:
    #     for corr_model in corr_models:
    #         for risk in risks:
    #             for gamma in gammas:
    #                 experiment_config, experiment_class = \
    #                     ConfigurationManager(experiment_type='llg') \
    #                     .with_correlation(gamma=gamma) \
    #                     .get_config(
    #                         log_root_dir=log_root_dir,
    #                         n_runs=n_runs,
    #                         n_epochs=n_epochs,
    #                         correlation_types=corr_model,
    #                         payment_rule=payment_rule,
    #                         specific_gpu=specific_gpu,
    #                         eval_batch_size=eval_batch_size,
    #                         model_sharing=model_sharing,
    #                         pretrain_iters=pretrain_iters,
    #                         risk=risk
    #                     )
    #                 run(experiment_config, experiment_class)

    # # Mineral rights
    # n_players = 3
    # experiment_config, experiment_class = \
    #     ConfigurationManager(experiment_type='mineral_rights') \
    #         .get_config(
    #             log_root_dir=log_root_dir,
    #             n_runs=n_runs,
    #             n_epochs=n_epochs,
    #             specific_gpu=specific_gpu,
    #             n_players=n_players,
    #             correlation_groups=[list(range(n_players))],
    #             # pretrain_iters=10
    #         )
    # run(experiment_config, experiment_class)

    # # Affiliated observations
    # experiment_config, experiment_class = \
    #     ConfigurationManager(experiment_type='affiliated_observations') \
    #         .get_config(
    #             log_root_dir=log_root_dir,
    #             n_runs=n_runs,
    #             n_epochs=n_epochs,
    #             specific_gpu=specific_gpu,
    #             pretrain_iters=pretrain_iters
    #         )
    # run(experiment_config, experiment_class)


    ### Run LLG nearest-vcg for different risks / correlations ################
    log_root_dir = os.path.join(
        os.path.expanduser('~'), 'bnelearn', 'experiments', 'interdependence',
        'risk_vs_correlation'
    )
    risks = list(i/10 for i in range(1, 10))
    gammas = list(i/10 for i in range(0, 10))
    payment_rule = 'nearest_vcg'
    corr_models = ['constant_weights'] #['Bernoulli_weights', 'constant_weights']
    for corr_model in corr_models:
        for risk in risks:
            for gamma in gammas:
                experiment_config, experiment_class = \
                    ConfigurationManager(experiment_type='llg') \
                        .with_correlation(gamma=gamma) \
                        .get_config(
                            log_root_dir=log_root_dir,
                            n_runs=n_runs,
                            n_epochs=n_epochs,
                            correlation_types=corr_model,
                            payment_rule=payment_rule,
                            specific_gpu=specific_gpu,
                            eval_batch_size=eval_batch_size,
                            model_sharing=model_sharing,
                            pretrain_iters=pretrain_iters,
                            risk=risk
                        )
                run(experiment_config, experiment_class)


    ### Run LLG setting for different correlation strengths ###################
    # log_root_dir = os.path.join(
    #     os.path.expanduser('~'), 'bnelearn', 'experiments', 'interdependence',
    #     'different_correlations'
    # )
    # for gamma in [g/10 for g in range(0, 11)]:
    #     experiment_config, experiment_class = \
    #         ConfigurationManager(experiment_type='llg') \
    #             .with_correlation(gamma=gamma) \
    #             .get_config(
    #                 log_root_dir=log_root_dir,
    #                 n_runs=n_runs,
    #                 n_epochs=n_epochs,
    #                 correlation_types='Bernoulli_weights',
    #                 eval_batch_size=eval_batch_size,
    #                 specific_gpu=specific_gpu,
    #                 model_sharing=model_sharing,
    #             )
    #     run(experiment_config, experiment_class)
