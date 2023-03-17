"""Script for reproducing the reported experiments from Bichler et. al (2023).
"""
import os
import sys

import torch

sys.path.append(os.path.realpath('.'))

from bnelearn.experiment.configuration_manager import ConfigurationManager  # pylint: disable=import-error


if __name__ == '__main__':

    # User parameters
    sigma = .1
    specific_gpu = 2
    log_root_dir = os.path.join(
        os.path.expanduser('~'), 'bnelearn', 'experiments', 'asymmetric',
    )

    # 1. Individual experiments
    # 1.1 Single item
    if False:
        # Common parameters #######################################################
        n_epochs = 2000
        n_runs = 10

        # Single-item asymmetric experiments ######################################
        experiment_types = [
            'single_item_asymmetric_uniform_overlapping',
            'single_item_asymmetric_uniform_disjunct',
            'single_item_asymmetric_beta',
            ]

        for experiment_type in experiment_types:
            experiment_config, experiment_class = \
                ConfigurationManager(
                    experiment_type=experiment_type,
                    n_runs=n_runs, n_epochs=n_epochs,
                    ) \
                .set_learning(
                    learner_hyperparams={
                        'population_size': 64,
                        'sigma': sigma,
                        'scale_sigma_by_model_size': True
                        },
                    ) \
                .set_logging(
                    util_loss_batch_size=2**12,
                    util_loss_grid_size=2**10,
                    eval_frequency=n_epochs,
                    best_response=True,
                    plot_frequency=500,
                    cache_eval_actions=True,
                    log_root_dir=log_root_dir,
                    save_models=True,
                    ) \
                .set_hardware(
                    specific_gpu=specific_gpu,
                ) \
                .get_config()
            experiment = experiment_class(experiment_config)
            experiment.run()
            torch.cuda.empty_cache()


        # Asymmetric LLG experiment ###############################################
        experiment_config, experiment_class = \
            ConfigurationManager(
                experiment_type='llg_full',
                n_runs=n_runs, n_epochs=n_epochs,
                ) \
            .set_setting(
                payment_rule='mrcs_favored',
                ) \
            .set_learning(
                batch_size=2**17,
                learner_hyperparams={
                    'population_size': 64,
                    'sigma': sigma,
                    'scale_sigma_by_model_size': True
                    },
                ) \
            .set_logging(
                eval_batch_size=2**17,
                util_loss_batch_size=2**11,
                util_loss_grid_size=2**10,
                eval_frequency=n_epochs,
                best_response=True,
                plot_frequency=500,
                cache_eval_actions=True,
                log_root_dir=log_root_dir,
                save_models=True,
                ) \
            .set_hardware(
                specific_gpu=specific_gpu
                ) \
            .get_config()
        experiment = experiment_class(experiment_config)
        experiment.run()
        torch.cuda.empty_cache()


        # Split-award auction #####################################################
        experiment_config, experiment_class = \
            ConfigurationManager(
                experiment_type='splitaward',
                n_runs=n_runs, n_epochs=n_epochs
                ) \
            .set_learning(
                learner_hyperparams={
                    'population_size': 64,
                    'sigma': sigma,
                    'scale_sigma_by_model_size': True
                    },
                ) \
            .set_logging(
                util_loss_batch_size=2**12,
                util_loss_grid_size=2**10,
                eval_frequency=n_epochs,
                best_response=True,
                plot_frequency=500,
                cache_eval_actions=True,
                log_root_dir=log_root_dir,
                save_models=True,
                ) \
            .set_hardware(
                specific_gpu=specific_gpu
                ) \
            .get_config()
        experiment = experiment_class(experiment_config)
        experiment.run()
        torch.cuda.empty_cache()


    # 1.2 Multi-unit asymmetric setting
    if False:
        for n_items, util_loss_grid_size in zip([4, 8, 12], [2**14, 2**16, 2**22]):
            print(f'\nminimum_number_of_points {util_loss_grid_size**(1/n_items)}')
            experiment_config, experiment_class = \
                ConfigurationManager(
                    experiment_type='multiunit',
                    n_runs=3, n_epochs=150
                    ) \
                .set_setting(
                    payment_rule='uniform',
                    n_items=n_items,
                    n_players=3,
                    u_lo=[0, 0, 0],
                    u_hi=[1, 1, 2]
                    ) \
                .set_learning(
                    # batch_size=2**15,
                    pretrain_iters=500,
                    learner_hyperparams={
                        'population_size': 64,
                        'sigma': sigma,
                        'scale_sigma_by_model_size': True
                        },
                    ) \
                .set_logging(
                    log_metrics={
                        'efficiency': True,
                        'revenue': True,
                        'util_loss': True,
                        },
                    util_loss_batch_size=2**10,
                    util_loss_grid_size=util_loss_grid_size,
                    eval_frequency=150,
                    plot_frequency=50,
                    cache_eval_actions=True,
                    log_root_dir=log_root_dir,
                    save_tb_events_to_csv_detailed=True,
                    save_models=True,
                    ) \
                .set_hardware(
                    specific_gpu=specific_gpu
                    ) \
                .get_config()
            experiment = experiment_class(experiment_config)
            experiment.positive_output_point = None
            experiment.run()
            torch.cuda.empty_cache()


    # 1.3 LLLLGG combinatorial experiment
    # 1.3.1 All
    if False:
        n_runss = [10, 2]
        n_epochss = [5000, 1000]
        payment_rules = ['first_price', 'nearest_vcg']
        population_sizes = [64, 32]
        batch_sizes = [2**18, 2**14]
        util_loss_batch_sizes = [2**12, 2**7]
        util_loss_grid_sizes = [2**10, 2**8]

        for n_runs, n_epochs, payment_rule, population_size, batch_size, \
            util_loss_batch_size, util_loss_grid_size in zip(n_runss, \
            n_epochss, payment_rules, population_sizes, batch_sizes, \
            util_loss_batch_sizes, util_loss_grid_sizes):
            experiment_config, experiment_class = \
                ConfigurationManager(
                    experiment_type='llllgg',
                    n_runs=n_runs, n_epochs=n_epochs
                    ) \
                .set_setting(
                    core_solver='mpc',
                    payment_rule=payment_rule
                    ) \
                .set_learning(
                    batch_size=batch_size,
                    learner_hyperparams={
                        'population_size': population_size,
                        'sigma': sigma,
                        'scale_sigma_by_model_size': True
                        },
                    ) \
                .set_logging(
                    util_loss_batch_size=util_loss_batch_size,
                    util_loss_grid_size=util_loss_grid_size,
                    eval_frequency=n_epochs,
                    best_response=True,
                    plot_frequency=500,
                    cache_eval_actions=True,
                    log_root_dir=log_root_dir,
                    save_models=True,
                    ) \
                .set_hardware(
                    specific_gpu=specific_gpu
                    ) \
                .get_config()
            experiment = experiment_class(experiment_config)
            experiment.run()
            torch.cuda.empty_cache()


    # 1.3.2 FPSB only: for plot
    if False:
        log_root_dir = os.path.join(log_root_dir, 'llllgg_plot')
        experiment_config, experiment_class = \
            ConfigurationManager(
                experiment_type='llllgg',
                n_runs=10, n_epochs=1000
                ) \
            .set_setting(
                payment_rule='first_price'
                ) \
            .set_learning(
                batch_size=2**18,
                learner_hyperparams={
                    'population_size': 64,
                    'sigma': sigma,
                    'scale_sigma_by_model_size': True
                    },
                ) \
            .set_logging(
                util_loss_batch_size=2**12,
                util_loss_grid_size=2**10,
                eval_frequency=50,
                best_response=True,
                plot_frequency=500,
                cache_eval_actions=True,
                log_root_dir=log_root_dir,
                save_models=True,
                ) \
            .set_hardware(
                specific_gpu=specific_gpu
                ) \
            .get_config()
        experiment = experiment_class(experiment_config)
        experiment.run()
        torch.cuda.empty_cache()


    # 1.4 New large LLLLRRG combinatorial experiment
    if False:
        experiment_config, experiment_class = \
            ConfigurationManager(
                experiment_type='llllrrg',
                n_runs=3, n_epochs=5000
                ) \
            .set_setting(
                ) \
            .set_learning(
                # batch_size=2**18,
                learner_hyperparams={
                    'population_size': 64,
                    'sigma': sigma,
                    'scale_sigma_by_model_size': True
                    },
                ) \
            .set_logging(
                util_loss_batch_size=2**12,
                util_loss_grid_size=2**10,
                eval_frequency=100,
                best_response=True,
                plot_frequency=500,
                cache_eval_actions=True,
                log_root_dir=log_root_dir,
                save_models=True,
                ) \
            .set_hardware(
                specific_gpu=specific_gpu
                ) \
            .get_config()
        experiment = experiment_class(experiment_config)
        experiment.run()
        torch.cuda.empty_cache()


    # 2. Scalability experiment
    if False:
        log_root_dir += "/asymmetric-performance-analysis/"
        n_epochs = 500

        log_root_dir += "varied-population-size/"
        population_sizes = [16, 32, 64]
        batch_size = 2**18
        for population_size in population_sizes:

        # log_root_dir += "varied-batch-size/"
        # batch_sizes = [2**6, 2**10, 2**18]
        # population_size = 64
        # for batch_size in batch_sizes:

            experiment_config, experiment_class = \
                ConfigurationManager(
                    experiment_type='llllgg',
                    n_runs=3, n_epochs=n_epochs
                    ) \
                .set_setting(
                    core_solver='mpc',
                    payment_rule='first_price'
                    ) \
                .set_learning(
                    batch_size=batch_size,
                    pretrain_iters=500,
                    # optimizer_hyperparams={'lr': 1e-3},
                    learner_hyperparams={
                        'population_size': population_size,
                        'sigma': sigma,
                        'scale_sigma_by_model_size': True
                        },
                    ) \
                .set_logging(
                    # eval_batch_size=2**14,
                    log_root_dir=log_root_dir,
                    util_loss_batch_size=2**12,
                    util_loss_grid_size=2**10,
                    eval_frequency=50,
                    plot_frequency=n_epochs,
                    cache_eval_actions=True,
                    save_tb_events_to_csv_detailed=True,
                    save_models=True,
                    ) \
                .set_hardware(
                    specific_gpu=specific_gpu
                    ) \
                .get_config()
            experiment = experiment_class(experiment_config)
            experiment.run()
            torch.cuda.empty_cache()
