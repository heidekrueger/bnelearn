"""
Script for running experiments that specifically adress dependencies within
the games.
"""
import os
import sys
import traceback
import torch
from torch import nn

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
    n_epochs = 2000

    # model_sharing = True
    pretrain_iters = 500

    batch_size = 2**17
    eval_batch_size = 2**22
    util_loss_frequency = 2000
    util_loss_batch_size = 2**12
    util_loss_grid_size = 2**10


    # def run(experiment_config, experiment_class): # pylint: disable=redefined-outer-name
    #     """Run a experiment class from config"""
    #     experiment = experiment_class(experiment_config)
    #     if experiment.known_bne:
    #         experiment.logging.log_metrics = {
    #             'opt': True,
    #             'l2': True,
    #             'util_loss': True
    #         }
    #     experiment.logging.log_metrics['efficiency'] = True
    #     experiment.logging.log_metrics['revenue'] = True

    #     experiment.logging.util_loss_batch_size = 2**12
    #     experiment.logging.util_loss_grid_size = 2**10
    #     experiment.logging.util_loss_frequency = n_epochs
    #     experiment.logging.best_response = True

    #     try:
    #         experiment.run()

    #     except Exception: # pylint: disable=broad-except
    #         traceback.print_exc()
    #         torch.cuda.empty_cache()


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
    # log_root_dir = os.path.join(
    #     os.path.expanduser('~'), 'bnelearn', 'experiments'
    # )
    # experiment_config, experiment_class = \
    #     ConfigurationManager(
    #         experiment_type='affiliated_observations',
    #         n_runs=n_runs,
    #         n_epochs=n_epochs
    #         ) \
    #     .set_setting(
    #         # risk=risk,
    #         # gamma=gamma,
    #         ) \
    #     .set_learning(
    #         batch_size=batch_size,
    #         pretrain_iters=pretrain_iters,
    #         ) \
    #     .set_logging(
    #         log_root_dir=log_root_dir,
    #         util_loss_frequency=util_loss_frequency,
    #         util_loss_batch_size=util_loss_batch_size,
    #         util_loss_grid_size=util_loss_grid_size,
    #         eval_batch_size=eval_batch_size)
    #     .set_hardware(specific_gpu=specific_gpu) \
    #     .get_config()
    # experiment = experiment_class(experiment_config)
    # experiment.run()


    # Run LLG nearest-vcg for different risks / correlations ################
    log_root_dir = os.path.join(
        os.path.expanduser('~'), 'bnelearn', 'experiments', 'regret_winner'
    )
    regrets = [0.0, 0.1, 0.2, 0.3, 0.4]
    gammas = [0.0, 0.5, 1.0]  # list(i/10 for i in range(0, 11))
    payment_rules = ['nearest_vcg', 'first_price']
    corr_models = ['constant_weights', 'Bernoulli_weights']
    for corr_model in corr_models:
        for gamma in gammas:
            for payment_rule in payment_rules:
                for regret in regrets:
                    experiment_config, experiment_class = \
                        ConfigurationManager(
                            experiment_type='llg',
                            n_runs=n_runs,
                            n_epochs=n_epochs) \
                            .set_setting(
                                payment_rule=payment_rule,
                                gamma=gamma,
                                correlation_types=corr_model,
                                # loss_aversion = True,
                                regret=regret
                                ) \
                            .set_learning(
                                batch_size=batch_size,
                                pretrain_iters=pretrain_iters,
                                model_sharing=False,
                                ) \
                            .set_logging(
                                log_root_dir=log_root_dir,
                                util_loss_frequency=util_loss_frequency,
                                util_loss_batch_size=util_loss_batch_size,
                                util_loss_grid_size=util_loss_grid_size,
                                eval_batch_size=eval_batch_size) \
                            .set_hardware(specific_gpu=specific_gpu) \
                            .get_config()
                    experiment = experiment_class(experiment_config)
                    experiment.run()



    ### Run multi-unit settings ###############################################
    # log_root_dir = os.path.join(
    #     os.path.expanduser('~'), 'bnelearn', 'experiments',
    #     'test_statics_multiunit_correlated', 'larger_market'
    # )
    # n_players_list = [2, 3, 3]
    # n_units_list = [3, 3, 2]
    # payment_rules = ['first_price']  # ['first_price', 'vcg', 'uniform']
    # risks = [1.]  # list(i/10 for i in range(1, 11))
    # gammas = list(i/10 for i in range(0, 11))
    # for n_players, n_units in zip(n_players_list, n_units_list):
    # # for n_players in n_players_list:
    # #     for n_units in n_units_list:
    #         for payment_rule in payment_rules:
    #             for risk in risks:
    #                 for gamma in gammas:
    #                     experiment_config, experiment_class = \
    #                         ConfigurationManager(
    #                             experiment_type='multiunit',
    #                             n_runs=n_runs,
    #                             n_epochs=n_epochs) \
    #                             .set_setting(
    #                                 payment_rule=payment_rule,
    #                                 n_players=n_players,
    #                                 n_units=n_units,
    #                                 risk=risk,
    #                                 gamma=gamma,
    #                                 correlation_types='additive'
    #                                 ) \
    #                             .set_learning(
    #                                 batch_size=batch_size,
    #                                 pretrain_iters=pretrain_iters,
    #                                 # hidden_nodes=[20],
    #                                 # hidden_activations=[nn.SELU()]
    #                                 ) \
    #                             .set_logging(
    #                                 log_root_dir=log_root_dir,
    #                                 util_loss_frequency=util_loss_frequency,
    #                                 util_loss_batch_size=util_loss_batch_size,
    #                                 util_loss_grid_size=util_loss_grid_size,
    #                                 eval_batch_size=eval_batch_size) \
    #                             .set_hardware(specific_gpu=specific_gpu) \
    #                             .get_config()
    #                     experiment = experiment_class(experiment_config)
    #                     experiment.run()
