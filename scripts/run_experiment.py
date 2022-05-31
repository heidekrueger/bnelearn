"""
Runs predefined experiments with individual parameters
fire.Fire() asks you to decide for one of the experiments defined above
by writing its name and define the required (and optional) parameters
e.g.:
    experiment.py single_item_uniform_symmetric 1 20 [2,3] 'first_price'

alternatively instead of fire.Fire() use, e.g.:
    single_item_uniform_symmetric(1,20,[2,3],'first_price')

"""
import os
import sys

import torch

# put bnelearn imports after this.
# pylint: disable=wrong-import-position
sys.path.append(os.path.realpath('.'))
sys.path.append(os.path.join(os.path.expanduser('~'), 'bnelearn'))

sys.path.remove('/home/kohring/bnelearn')

from bnelearn.experiment.configuration_manager import ConfigurationManager  # pylint: disable=import-error


if __name__ == '__main__':

    # path is user-specific
    torch.autograd.set_detect_anomaly(True)

    log_root_dir = os.path.join(os.path.expanduser('~'), 'bnelearn', 'abstract_plots', 'updated_weak', '3')

    # # Contest Experiments
    experiment_config, experiment_class = ConfigurationManager(experiment_type='single_item_uniform_symmetric', n_runs=1, n_epochs=5000)\
        .set_setting(n_players=5, payment_rule='first_price', u_hi=[10], risk=0.1) \
        .set_logging(log_root_dir=log_root_dir, util_loss_frequency=1000, save_models=True) \
        .set_hardware(specific_gpu=5) \
        .set_learning(pretrain_iters=500, use_valuation=True, batch_size=2 ** 18, learner_type='PGLearner', mixed_strategy="normal") \
        .get_config()

    # Crowdsourcing Contest
    # experiment_config, experiment_class = ConfigurationManager(experiment_type='tullock_contest', n_runs=1, n_epochs=5000) \
    #     .set_setting(n_players=2, impact_factor=5, impact_function="tullock_contest") \
    #     .set_logging(log_root_dir=log_root_dir, util_loss_frequency=500, save_models=True, plot_frequency=500) \
    #     .set_hardware(specific_gpu=5) \
    #     .set_learning(pretrain_iters=500, use_valuation=True) \
    #     .get_config()

    # to be changed
    # valuations = torch.tensor([1.0, 0.0, 0.0])
    # np = 2
    # gpu = 4
    # use_valuation = True
    # # fixed
    # exp_type = "crowdsourcing"

    # experiment_config, experiment_class = ConfigurationManager(experiment_type=exp_type, n_runs=1, n_epochs=3500) \
    #     .set_setting(valuations=valuations, n_players=np) \
    #     .set_logging(log_root_dir=log_root_dir, util_loss_frequency=100, save_models=True, plot_frequency=100) \
    #     .set_hardware(specific_gpu=gpu) \
    #     .set_learning(pretrain_iters=500, use_valuation=use_valuation) \
    #     .get_config()

    # experiment_config, experiment_class = ConfigurationManager(experiment_type='all_pay', n_runs=1, n_epochs=15000) \
    #     .set_setting(n_players=2, valuations=torch.tensor([1.0, 0.0, 0.0])) \
    #     .set_logging(log_root_dir=log_root_dir, util_loss_frequency=1000, best_response=True, plot_frequency=100) \
    #     .set_hardware(specific_gpu=1) \
    #     .set_learning(pretrain_iters=500, use_valuation=True, learner_hyperparams={'population_size': 64,
    #                              'sigma': 5,
    #                              'scale_sigma_by_model_size': True}) \
    #     .get_config()

    # Run exps that contain integration
    # experiment_types = ['splitaward']  # ['single_item_gaussian_symmetric', 'multiunit', 'splitaward']
    # for experiment_type in experiment_types:
    #     experiment_config, experiment_class = \
    #         ConfigurationManager(
    #             experiment_type=experiment_type,
    #             n_runs=1,
    #             n_epochs=200
    #             ) \
    #         .set_setting(
    #             payment_rule='first_price' if experiment_type == 'multiunit' else 'None',
    #             constant_marginal_values=True,
    #             # correlation_groups=[[0, 1, 2]],
    #             # correlation_types='independent',
    #             # gamma=0.0
    #             ) \
    #         .set_logging(
    #             eval_batch_size=2**9,
    #             util_loss_batch_size=2**9,
    #             util_loss_grid_size=2**10,
    #             util_loss_frequency=50,
    #             best_response=True,
    #             cache_eval_actions=True,
    #             log_root_dir=log_root_dir,
    #             ) \
    #         .set_learning(
    #             # model_sharing=False
    #             ) \
    #         .set_hardware(
    #             specific_gpu=4,
    #             max_cpu_threads=1,
    #         ) \
    #         .get_config()
    #     experiment = experiment_class(experiment_config)
    #     experiment.run()
    #     torch.cuda.empty_cache()

    # Run from a file
    # experiment_config = logging.get_experiment_config_from_configurations_log()
    # experiment_class = ConfigurationManager \
    #    .get_class_by_experiment_type(experiment_config.experiment_class)



    ### SINGLE ITEM EXPERIMENTS ###

    # experiment_config, experiment_class = \
    #     ConfigurationManager(
    #         experiment_type='single_item_gaussian_symmetric',
    #         n_runs=1,
    #         n_epochs=200
    #         ) \
    #     .set_setting(
    #         # correlation_groups=[[0, 1, 2]],
    #         # correlation_types='independent',
    #         # gamma=0.0
    #         ) \
    #     .set_logging(
    #         eval_batch_size=2**18,
    #         util_loss_batch_size=2**10,
    #         util_loss_grid_size=2**10,
    #         util_loss_frequency=10,
    #         cache_eval_actions=True,
    #         log_root_dir=log_root_dir,
    #         save_tb_events_to_csv_detailed=True
    #         ) \
    #     .set_learning(
    #         model_sharing=False
    #         ) \
    #     .set_hardware(
    #         specific_gpu=7
    #     ) \
    #     .get_config()

    # experiment_config, experiment_class = ConfigurationManager(experiment_type='single_item_gaussian_symmetric',
    #                                                            n_runs=2, n_epochs=200)\
    #     .set_logging(log_root_dir=log_root_dir, save_tb_events_to_csv_detailed=True).get_config()

    #All three next experiments get AssertionError: scalar should be 0D
    # experiment_config, experiment_class = ConfigurationManager(
    #    experiment_type='single_item_asymmetric_uniform_overlapping',
    #    n_runs=1, n_epochs=20000
    # ) \
    #     .set_logging(log_root_dir=log_root_dir) \
    # #     .get_config()
    # experiment_config, experiment_class = \
    #     ConfigurationManager(
    #         experiment_type='single_item_uniform_symmetric',
    #         n_runs=1, n_epochs=500
    #     ) \
    #     .set_learning(
    #         learner_type='PGLearner',
    #         pretrain_iters=500
    #     ) \
    #     .set_logging(
    #         eval_batch_size=2**18,
    #         util_loss_batch_size=2**10,
    #         util_loss_grid_size=2**10,
    #         util_loss_frequency=50,
    #         plot_frequency=10,
    #         log_root_dir=log_root_dir) \
    #     .get_config()

    # experiment_config, experiment_class = ConfigurationManager(experiment_type='llg', n_runs=1, n_epochs=3) \
    #     .set_setting(gamma=0.5) \
    #     .set_logging(log_root_dir=log_root_dir,  util_loss_batch_size=2 ** 7, util_loss_grid_size=2 ** 6,
    #                  util_loss_frequency=1).get_config()

    # experiment_config, experiment_class = ConfigurationManager(experiment_type='llllgg', n_runs=1, n_epochs=200) \
    #     .set_logging(log_root_dir=log_root_dir) \
    #     .get_config()

    # experiment_config, experiment_class = ConfigurationManager(experiment_type='llllrrg', n_runs=10, n_epochs=4000) \
    #     .set_logging(log_root_dir=log_root_dir) \
    #     .set_hardware(specific_gpu=1) \
    #     .get_config()

    # RuntimeError: Sizes of tensors must match

    # experiment_config, experiment_class = \
    #     ConfigurationManager(
    #         experiment_type='llg_full',
    #         n_runs=1, n_epochs=500,
    #         seeds=[69]
    #     ) \
    #     .set_setting(
    #         payment_rule='mrcs_favored',
    #     ) \
    #     .set_learning(
    #         batch_size=2**15,
    #         model_sharing=True,
    #         pretrain_iters=500,
    #     ) \
    #     .set_logging(
    #         eval_batch_size=2**10,
    #         util_loss_batch_size=2**8,
    #         util_loss_grid_size=2**10,
    #         util_loss_frequency=50,
    #         best_response=True,
    #         plot_frequency=25,
    #         cache_eval_actions=True,
    #         log_root_dir=log_root_dir,
    #     ) \
    #     .set_hardware(
    #         specific_gpu=7
    #     ) \
    #     .get_config()

    # experiment_config, experiment_class = ConfigurationManager(
    #       experiment_type='splitaward',n_runs=1, n_epochs=200
    # ) \
    #     .set_logging(log_root_dir=log_root_dir) \
    #     .get_config()
    # experiment_config, experiment_class = ConfigurationManager(
    #    experiment_type='multiunit', n_runs=1, n_epochs=2
    # ) \
    #     .set_logging(log_root_dir=log_root_dir,
    #                  save_tb_events_to_csv_detailed=True) \
    #     .set_setting().set_learning().set_hardware() \
    #     .get_config()


    ### COMBINATRORIAL EXPERIMENTS ###
    # experiment_config, experiment_class = ConfigurationManager(
    #     experiment_type='llg', n_runs=1, n_epochs=1000
    # ) \
    #     .set_setting(gamma=0.5) \
    #     .set_logging(
    #        log_root_dir=log_root_dir,
    #        util_loss_batch_size=2 ** 10,
    #        util_loss_grid_size=2 ** 8,
    #        util_loss_frequency=20) \
    #     .get_config()
    # experiment_config, experiment_class = ConfigurationManager(
    #     experiment_type='llg_full', n_runs=1, n_epochs=10000) \
    #     .set_setting(payment_rule='mrcs_favored') \
    #     .set_learning(batch_size=2**18) \
    #     .set_logging(
    #         eval_batch_size=2**18,
    #         log_root_dir=log_root_dir,
    #         util_loss_batch_size=2**10,
    #         util_loss_grid_size=2**10,
    #         util_loss_frequency=1000,
    #         plot_frequency=10) \
    #     .set_hardware(specific_gpu=3) \
    #     .get_config()
    # experiment_config, experiment_class = ConfigurationManager(
    #    experiment_type='llllgg', n_runs=1, n_epochs=200
    # ) \
    #     .set_learning(batch_size=2**7) \
    #     .set_setting(core_solver='mpc', payment_rule='nearest_vcg') \
    #     .set_logging(log_root_dir=log_root_dir, log_metrics={'util_loss': True},
    #                  util_loss_frequency=5, plot_frequency=5) \
    #     .get_config()


    ### INTERDEPENDENT EXPERIMENTS ###
    # experiment_config, experiment_class = ConfigurationManager(
    #     experiment_type='mineral_rights', n_runs=1, n_epochs=1000
    # ) \
    #     .set_learning(pretrain_iters=3) \
    #     .set_logging(
    #         log_root_dir=log_root_dir,
    #         util_loss_frequency=10) \
    #     .set_hardware(specific_gpu=7) \
    #     .get_config()
    # experiment_config, experiment_class = ConfigurationManager(
    #     experiment_type='affiliated_observations', n_runs=1, n_epochs=100
    # ) \
    #     .set_learning(pretrain_iters=1) \
    #     .set_logging(log_root_dir=log_root_dir) \
    #     .set_hardware(specific_gpu=1) \
    #     .get_config()

    experiment = experiment_class(experiment_config)
    experiment.run()
    torch.cuda.empty_cache()
