"""
Runs predefined double auction experiments with individual parameters
"""
import os
import sys
from itertools import product
import numpy as np
import torch
import torch.nn as nn

# pylint: disable=wrong-import-position
sys.path.append(os.path.realpath('.'))
sys.path.append(os.path.join(os.path.expanduser('~'), 'bnelearn'))
from bnelearn.experiment.configuration_manager import ConfigurationManager  # pylint: disable=import-error


if __name__ == '__main__':

    ### EXP-1 BB 1/2-DA & VCG -------------------------------------------------
    if False:
        log_root_dir = os.path.join('/home/pieroth/projects/bnelearn/experiments/bargaining_paper_results', 'exp-1_experiment')
        payment_rules = ['k_price', 'vcg']
        for payment_rule in payment_rules:
            experiment_config, experiment_class = \
                ConfigurationManager(
                    experiment_type='double_auction_single_item_uniform_symmetric',
                    n_runs=10,  # repeat exp. for different random seeds
                    n_epochs=2000,
                ) \
                .set_setting(
                    payment_rule=payment_rule,
                    k=0.5,
                ) \
                .set_learning(
                    batch_size=2**18,  # default value -> may need to be decreased for larger markets
                    model_sharing=True,
                    # hidden_nodes=[100],
                    # hidden_activations=[nn.SELU()],
                    # dropout=0.1,
                ) \
                .set_hardware(
                    specific_gpu=4,
                ) \
                .set_logging(
                    eval_batch_size=2**22,  # needed for exact utility-loss (epsilon_relative)
                    cache_eval_actions=True,

                    # needed for estimated utility-loss (estimated_relative_ex_ante_util_loss)
                    util_loss_batch_size=2**13,  # default value -> may needs to be decreased for larger markets
                    util_loss_grid_size=2**10,  # default value -> may needs to be decreased for larger markets
                    util_loss_frequency=2000,  # don't want to calculate that often as it takes long

                    best_response=True,  # only needed for best response plots
                    log_root_dir=log_root_dir,
                    save_tb_events_to_csv_detailed=True,
                    save_models=True,  # needed if you want to plot bid functions afterward
                    plot_frequency=500,  # don't want to waste much disk space
                ) \
                .get_config()
            experiment = experiment_class(experiment_config)
            experiment.run()
            torch.cuda.empty_cache()

    ### EXP-2 risk experiments ------------------------------------------------
    if False:
        log_root_dir = os.path.join(os.path.expanduser('~'), 'bnelearn', \
            'experiments', 'debug', 'exp-2_experiment')
        risks = [i/10. for i in range(1, 11)]
        payment_rules = ['vcg', 'k_price']
        for risk in risks:
            for payment_rule in payment_rules:
                experiment_config, experiment_class = \
                    ConfigurationManager(
                        experiment_type='double_auction_single_item_uniform_symmetric',
                        n_runs=10,  # repeat exp. for different random seeds
                        n_epochs=2000,
                    ) \
                    .set_setting(
                        payment_rule=payment_rule,
                        k=0.5,
                        risk=risk  # 1.0 is default (risk-neutral)
                    ) \
                    .set_learning(
                        batch_size=2**18,  # default value -> may need to be decreased for larger markets
                        model_sharing=True,
                    ) \
                    .set_hardware(
                        specific_gpu=5,
                    ) \
                    .set_logging(
                        eval_batch_size=2**22,  # needed for exact utility-loss (epsilon_relative)
                        cache_eval_actions=True,

                        # needed for estimated utility-loss (estimated_relative_ex_ante_util_loss)
                        util_loss_batch_size=2**13,  # default value -> may needs to be decreased for larger markets
                        util_loss_grid_size=2**10,  # default value -> may needs to be decreased for larger markets
                        util_loss_frequency=2000,  # don't want to calculate that often as it takes long

                        best_response=True,  # only needed for best response plots
                        log_root_dir=log_root_dir,
                        save_tb_events_to_csv_detailed=True,
                        save_models=True,  # needed if you want to plot bid functions afterward
                        plot_frequency=500,  # don't want to waste much disk space
                    ) \
                    .get_config()
                experiment = experiment_class(experiment_config)
                experiment.run()
                torch.cuda.empty_cache()

    ### EXP-3 different pretraining -------------------------------------------
    if False:
        log_root_dir = os.path.join(os.path.expanduser('~'), 'bnelearn', \
            'experiments', 'debug', 'exp-3_experiment')
        pretrainings = np.linspace(0.01, 0.49, 12)
        for pretraining in pretrainings[6:]:
            experiment_config, experiment_class = \
                ConfigurationManager(
                    experiment_type='double_auction_single_item_uniform_symmetric',
                    n_runs=10,  # repeat exp. for different random seeds
                    n_epochs=2000
                ) \
                .set_setting(
                    payment_rule='k_price',
                    k=0.5,
                    pretrain_transform=pretraining,
                ) \
                .set_learning(
                    batch_size=2**18,  # default value -> may need to be decreased for larger markets
                    model_sharing=True,
                    pretrain_iters=10000,
                    # hidden_nodes=[100],
                    # hidden_activations=[nn.SELU()]
                ) \
                .set_hardware(
                    specific_gpu=2,
                ) \
                .set_logging(
                    eval_batch_size=2**22,  # needed for exact utility-loss (epsilon_relative)
                    cache_eval_actions=True,

                    # needed for estimated utility-loss (estimated_relative_ex_ante_util_loss)
                    util_loss_batch_size=2**13,  # default value -> may needs to be decreased for larger markets
                    util_loss_grid_size=2**10,  # default value -> may needs to be decreased for larger markets
                    util_loss_frequency=5000,  # don't want to calculate that often as it takes long

                    best_response=True,  # only needed for best response plots
                    log_root_dir=log_root_dir,
                    save_tb_events_to_csv_detailed=True,
                    save_models=True,  # needed if you want to plot bid functions afterward
                    plot_frequency=500,  # don't want to waste much disk space
                ) \
                .get_config()
            experiment = experiment_class(experiment_config)
            experiment.run()
            torch.cuda.empty_cache()
    
    ### EXP-4 normal distributed prior -------------------------------------------
    if False:
        log_root_dir = os.path.join(os.path.expanduser('~'), 'bnelearn', \
            'experiments', 'debug', 'exp-4_experiment')
        payment_rules = ['k_price', 'vcg']
        for payment_rule in payment_rules:
            experiment_config, experiment_class = \
                ConfigurationManager(
                    experiment_type='double_auction_single_item_gaussian_symmetric',
                    n_runs=10,  # repeat exp. for different random seeds
                    n_epochs=2000,
                ) \
                .set_setting(
                    payment_rule=payment_rule,
                    k=0.5,
                    common_prior='Normal',
                    valuation_mean=15.0,
                    valuation_std=5.0
                ) \
                .set_learning(
                    batch_size=2**18,  # default value -> may need to be decreased for larger markets
                    model_sharing=True,
                    # hidden_nodes=[100],
                    # hidden_activations=[nn.SELU()],
                    # dropout=0.1,
                ) \
                .set_hardware(
                    specific_gpu=2,
                ) \
                .set_logging(
                    eval_batch_size=2**22,  # needed for exact utility-loss (epsilon_relative)
                    cache_eval_actions=True,

                    # needed for estimated utility-loss (estimated_relative_ex_ante_util_loss)
                    util_loss_batch_size=2**13,  # default value -> may needs to be decreased for larger markets
                    util_loss_grid_size=2**10,  # default value -> may needs to be decreased for larger markets
                    util_loss_frequency=2000,  # don't want to calculate that often as it takes long

                    best_response=True,  # only needed for best response plots
                    log_root_dir=log_root_dir,
                    save_tb_events_to_csv_detailed=True,
                    save_models=True,  # needed if you want to plot bid functions afterward
                    plot_frequency=500,  # don't want to waste much disk space
                ) \
                .get_config()
            experiment = experiment_class(experiment_config)
            experiment.run()
            torch.cuda.empty_cache()

    ### EXP-5 two buyers and one seller -------------------------------------------
    if False:
        log_root_dir = os.path.join(os.path.expanduser('~'), 'bnelearn', \
            'experiments', 'debug', 'exp-5_experiment')
        payment_rules = ['k_price', 'vcg']
        for payment_rule in payment_rules:
            experiment_config, experiment_class = \
                ConfigurationManager(
                    experiment_type='double_auction_single_item_uniform_symmetric',
                    n_runs=10,  # repeat exp. for different random seeds
                    n_epochs=2000,
                ) \
                .set_setting(
                    payment_rule=payment_rule,
                    k=0.5,
                    n_players = 3,
                    n_buyers = 2,
                    n_sellers = 1,
                ) \
                .set_learning(
                    batch_size=2**18,  # default value -> may need to be decreased for larger markets
                    model_sharing=True,
                    # hidden_nodes=[100],
                    # hidden_activations=[nn.SELU()],
                    # dropout=0.1,
                ) \
                .set_hardware(
                    specific_gpu=2,
                ) \
                .set_logging(
                    eval_batch_size=2**22,  # needed for exact utility-loss (epsilon_relative)
                    cache_eval_actions=True,

                    log_metrics={
                        'opt': True,
                        'util_loss': True,
                    },

                    # needed for estimated utility-loss (estimated_relative_ex_ante_util_loss)
                    util_loss_batch_size=2**10,  # default value -> may needs to be decreased for larger markets
                    util_loss_grid_size=2**10,  # default value -> may needs to be decreased for larger markets
                    util_loss_frequency=2000,  # don't want to calculate that often as it takes long

                    best_response=True,  # only needed for best response plots
                    log_root_dir=log_root_dir,
                    save_tb_events_to_csv_detailed=True,
                    save_models=True,  # needed if you want to plot bid functions afterward
                    plot_frequency=500,  # don't want to waste much disk space
                ) \
                .get_config()
            experiment = experiment_class(experiment_config)
            experiment.run()
            torch.cuda.empty_cache()

    ### EXP-5-2 three buyers and one seller -------------------------------------------
    if False:
        log_root_dir = os.path.join(os.path.expanduser('~'), 'bnelearn', \
            'experiments', 'debug', 'exp-5-2_experiment')
        payment_rules = ['k_price', 'vcg']
        for payment_rule in payment_rules:
            experiment_config, experiment_class = \
                ConfigurationManager(
                    experiment_type='double_auction_single_item_uniform_symmetric',
                    n_runs=10,  # repeat exp. for different random seeds
                    n_epochs=2000,
                ) \
                .set_setting(
                    payment_rule=payment_rule,
                    k=0.5,
                    n_players = 4,
                    n_buyers = 3,
                    n_sellers = 1,
                ) \
                .set_learning(
                    batch_size=2**18,  # default value -> may need to be decreased for larger markets
                    model_sharing=True,
                    # hidden_nodes=[100],
                    # hidden_activations=[nn.SELU()],
                    # dropout=0.1,
                ) \
                .set_hardware(
                    specific_gpu=4,
                ) \
                .set_logging(
                    eval_batch_size=2**22,  # needed for exact utility-loss (epsilon_relative)
                    cache_eval_actions=True,

                    log_metrics={
                        'opt': True,
                        'util_loss': True,
                    },

                    # needed for estimated utility-loss (estimated_relative_ex_ante_util_loss)
                    util_loss_batch_size=2**10,  # default value -> may needs to be decreased for larger markets
                    util_loss_grid_size=2**10,  # default value -> may needs to be decreased for larger markets
                    util_loss_frequency=2000,  # don't want to calculate that often as it takes long

                    best_response=True,  # only needed for best response plots
                    log_root_dir=log_root_dir,
                    save_tb_events_to_csv_detailed=True,
                    save_models=True,  # needed if you want to plot bid functions afterward
                    plot_frequency=500,  # don't want to waste much disk space
                ) \
                .get_config()
            experiment = experiment_class(experiment_config)
            experiment.run()
            torch.cuda.empty_cache()

    ### EXP-5-3 four buyers and one seller -------------------------------------------
    if False:
        log_root_dir = os.path.join(os.path.expanduser('~'), 'bnelearn', \
            'experiments', 'debug', 'exp-5-3_experiment')
        payment_rules = ['k_price', 'vcg']
        for payment_rule in payment_rules:
            experiment_config, experiment_class = \
                ConfigurationManager(
                    experiment_type='double_auction_single_item_uniform_symmetric',
                    n_runs=10,  # repeat exp. for different random seeds
                    n_epochs=2000,
                ) \
                .set_setting(
                    payment_rule=payment_rule,
                    k=0.5,
                    n_players = 5,
                    n_buyers = 4,
                    n_sellers = 1,
                ) \
                .set_learning(
                    batch_size=2**18,  # default value -> may need to be decreased for larger markets
                    model_sharing=True,
                    # hidden_nodes=[100],
                    # hidden_activations=[nn.SELU()],
                    # dropout=0.1,
                ) \
                .set_hardware(
                    specific_gpu=5,
                ) \
                .set_logging(
                    eval_batch_size=2**22,  # needed for exact utility-loss (epsilon_relative)
                    cache_eval_actions=True,

                    log_metrics={
                        'opt': True,
                        'util_loss': True,
                    },

                    # needed for estimated utility-loss (estimated_relative_ex_ante_util_loss)
                    util_loss_batch_size=2**10,  # default value -> may needs to be decreased for larger markets
                    util_loss_grid_size=2**10,  # default value -> may needs to be decreased for larger markets
                    util_loss_frequency=2000,  # don't want to calculate that often as it takes long

                    best_response=True,  # only needed for best response plots
                    log_root_dir=log_root_dir,
                    save_tb_events_to_csv_detailed=True,
                    save_models=True,  # needed if you want to plot bid functions afterward
                    plot_frequency=500,  # don't want to waste much disk space
                ) \
                .get_config()
            experiment = experiment_class(experiment_config)
            experiment.run()
            torch.cuda.empty_cache()

    ### EXP-6 one buyer and two sellers -------------------------------------------
    if False:
        log_root_dir = os.path.join(os.path.expanduser('~'), 'bnelearn', \
            'experiments', 'debug', 'exp-6_experiment')
        payment_rules = ['k_price', 'vcg']
        for payment_rule in payment_rules:
            experiment_config, experiment_class = \
                ConfigurationManager(
                    experiment_type='double_auction_single_item_uniform_symmetric',
                    n_runs=10,  # repeat exp. for different random seeds
                    n_epochs=2000,
                ) \
                .set_setting(
                    payment_rule=payment_rule,
                    k=0.5,
                    n_players = 3,
                    n_buyers = 1,
                    n_sellers = 2,
                ) \
                .set_learning(
                    batch_size=2**18,  # default value -> may need to be decreased for larger markets
                    model_sharing=True,
                    # hidden_nodes=[100],
                    # hidden_activations=[nn.SELU()],
                    # dropout=0.1,
                ) \
                .set_hardware(
                    specific_gpu=6,
                ) \
                .set_logging(
                    eval_batch_size=2**22,  # needed for exact utility-loss (epsilon_relative)
                    cache_eval_actions=True,

                    log_metrics={
                        'opt': True,
                        'util_loss': True,
                    },

                    # needed for estimated utility-loss (estimated_relative_ex_ante_util_loss)
                    util_loss_batch_size=2**10,  # default value -> may needs to be decreased for larger markets
                    util_loss_grid_size=2**10,  # default value -> may needs to be decreased for larger markets
                    util_loss_frequency=2000,  # don't want to calculate that often as it takes long

                    best_response=True,  # only needed for best response plots
                    log_root_dir=log_root_dir,
                    save_tb_events_to_csv_detailed=True,
                    save_models=True,  # needed if you want to plot bid functions afterward
                    plot_frequency=500,  # don't want to waste much disk space
                ) \
                .get_config()
            experiment = experiment_class(experiment_config)
            experiment.run()
            torch.cuda.empty_cache()
    
    ### EXP-6-no-model-sharing one buyer and two sellers -------------------------------------------
    if False:
        log_root_dir = os.path.join(os.path.expanduser('~'), 'bnelearn', \
            'experiments', 'debug', 'exp-6-no-model-sharing_experiment')
        payment_rules = ['k_price', 'vcg']
        for payment_rule in payment_rules:
            experiment_config, experiment_class = \
                ConfigurationManager(
                    experiment_type='double_auction_single_item_uniform_symmetric',
                    n_runs=10,  # repeat exp. for different random seeds
                    n_epochs=2000,
                ) \
                .set_setting(
                    payment_rule=payment_rule,
                    k=0.5,
                    n_players = 3,
                    n_buyers = 1,
                    n_sellers = 2,
                ) \
                .set_learning(
                    batch_size=2**18,  # default value -> may need to be decreased for larger markets
                    model_sharing=False,
                    # hidden_nodes=[100],
                    # hidden_activations=[nn.SELU()],
                    # dropout=0.1,
                ) \
                .set_hardware(
                    specific_gpu=6,
                ) \
                .set_logging(
                    eval_batch_size=2**22,  # needed for exact utility-loss (epsilon_relative)
                    cache_eval_actions=True,

                    log_metrics={
                        'opt': True,
                        'util_loss': True,
                    },

                    # needed for estimated utility-loss (estimated_relative_ex_ante_util_loss)
                    util_loss_batch_size=2**10,  # default value -> may needs to be decreased for larger markets
                    util_loss_grid_size=2**10,  # default value -> may needs to be decreased for larger markets
                    util_loss_frequency=2000,  # don't want to calculate that often as it takes long

                    best_response=True,  # only needed for best response plots
                    log_root_dir=log_root_dir,
                    save_tb_events_to_csv_detailed=True,
                    save_models=True,  # needed if you want to plot bid functions afterward
                    plot_frequency=500,  # don't want to waste much disk space
                ) \
                .get_config()
            experiment = experiment_class(experiment_config)
            experiment.run()
            torch.cuda.empty_cache()
    
    ### EXP-6-2 one buyer and three sellers -------------------------------------------
    if False:
        log_root_dir = os.path.join(os.path.expanduser('~'), 'bnelearn', \
            'experiments', 'debug', 'exp-6-2_experiment')
        payment_rules = ['k_price', 'vcg']
        for payment_rule in payment_rules:
            experiment_config, experiment_class = \
                ConfigurationManager(
                    experiment_type='double_auction_single_item_uniform_symmetric',
                    n_runs=10,  # repeat exp. for different random seeds
                    n_epochs=2000,
                ) \
                .set_setting(
                    payment_rule=payment_rule,
                    k=0.5,
                    n_players = 4,
                    n_buyers = 1,
                    n_sellers = 3,
                ) \
                .set_learning(
                    batch_size=2**18,  # default value -> may need to be decreased for larger markets
                    model_sharing=True,
                    # hidden_nodes=[100],
                    # hidden_activations=[nn.SELU()],
                    # dropout=0.1,
                ) \
                .set_hardware(
                    specific_gpu=4,
                ) \
                .set_logging(
                    eval_batch_size=2**22,  # needed for exact utility-loss (epsilon_relative)
                    cache_eval_actions=True,

                    log_metrics={
                        'opt': True,
                        'util_loss': True,
                    },

                    # needed for estimated utility-loss (estimated_relative_ex_ante_util_loss)
                    util_loss_batch_size=2**10,  # default value -> may needs to be decreased for larger markets
                    util_loss_grid_size=2**10,  # default value -> may needs to be decreased for larger markets
                    util_loss_frequency=2000,  # don't want to calculate that often as it takes long

                    best_response=True,  # only needed for best response plots
                    log_root_dir=log_root_dir,
                    save_tb_events_to_csv_detailed=True,
                    save_models=True,  # needed if you want to plot bid functions afterward
                    plot_frequency=500,  # don't want to waste much disk space
                ) \
                .get_config()
            experiment = experiment_class(experiment_config)
            experiment.run()
            torch.cuda.empty_cache()
    
    ### EXP-6-3 one buyer and four sellers -------------------------------------------
    if False:
        log_root_dir = os.path.join(os.path.expanduser('~'), 'bnelearn', \
            'experiments', 'debug', 'exp-6-3_experiment')
        payment_rules = ['k_price', 'vcg']
        for payment_rule in payment_rules:
            experiment_config, experiment_class = \
                ConfigurationManager(
                    experiment_type='double_auction_single_item_uniform_symmetric',
                    n_runs=10,  # repeat exp. for different random seeds
                    n_epochs=2000,
                ) \
                .set_setting(
                    payment_rule=payment_rule,
                    k=0.5,
                    n_players = 5,
                    n_buyers = 1,
                    n_sellers = 4,
                ) \
                .set_learning(
                    batch_size=2**18,  # default value -> may need to be decreased for larger markets
                    model_sharing=True,
                    # hidden_nodes=[100],
                    # hidden_activations=[nn.SELU()],
                    # dropout=0.1,
                ) \
                .set_hardware(
                    specific_gpu=5,
                ) \
                .set_logging(
                    eval_batch_size=2**22,  # needed for exact utility-loss (epsilon_relative)
                    cache_eval_actions=True,

                    log_metrics={
                        'opt': True,
                        'util_loss': True,
                    },

                    # needed for estimated utility-loss (estimated_relative_ex_ante_util_loss)
                    util_loss_batch_size=2**10,  # default value -> may needs to be decreased for larger markets
                    util_loss_grid_size=2**10,  # default value -> may needs to be decreased for larger markets
                    util_loss_frequency=2000,  # don't want to calculate that often as it takes long

                    best_response=True,  # only needed for best response plots
                    log_root_dir=log_root_dir,
                    save_tb_events_to_csv_detailed=True,
                    save_models=True,  # needed if you want to plot bid functions afterward
                    plot_frequency=500,  # don't want to waste much disk space
                ) \
                .get_config()
            experiment = experiment_class(experiment_config)
            experiment.run()
            torch.cuda.empty_cache()
    
    ### EXP-7 two buyers and sellers -------------------------------------------
    if False:
        log_root_dir = os.path.join(os.path.expanduser('~'), 'bnelearn', \
            'experiments', 'debug', 'exp-7_experiment')
        payment_rules = ['k_price', 'vcg']
        for payment_rule in payment_rules:
            experiment_config, experiment_class = \
                ConfigurationManager(
                    experiment_type='double_auction_single_item_uniform_symmetric',
                    n_runs=10,  # repeat exp. for different random seeds
                    n_epochs=2000,
                ) \
                .set_setting(
                    payment_rule=payment_rule,
                    k=0.5,
                    n_players = 4,
                    n_buyers = 2,
                    n_sellers = 2,
                ) \
                .set_learning(
                    batch_size=2**18,  # default value -> may need to be decreased for larger markets
                    model_sharing=True,
                    # hidden_nodes=[100],
                    # hidden_activations=[nn.SELU()],
                    # dropout=0.1,
                ) \
                .set_hardware(
                    specific_gpu=7,
                ) \
                .set_logging(
                    eval_batch_size=2**22,  # needed for exact utility-loss (epsilon_relative)
                    cache_eval_actions=True,

                    log_metrics={
                        'opt': True,
                        'util_loss': True,
                    },

                    # needed for estimated utility-loss (estimated_relative_ex_ante_util_loss)
                    util_loss_batch_size=2**10,  # default value -> may needs to be decreased for larger markets
                    util_loss_grid_size=2**10,  # default value -> may needs to be decreased for larger markets
                    util_loss_frequency=2000,  # don't want to calculate that often as it takes long

                    best_response=True,  # only needed for best response plots
                    log_root_dir=log_root_dir,
                    save_tb_events_to_csv_detailed=True,
                    save_models=True,  # needed if you want to plot bid functions afterward
                    plot_frequency=500,  # don't want to waste much disk space
                ) \
                .get_config()
            experiment = experiment_class(experiment_config)
            experiment.run()
            torch.cuda.empty_cache()

    ### EXP-7-2 three buyers and sellers -------------------------------------------
    if False:
        log_root_dir = os.path.join(os.path.expanduser('~'), 'bnelearn', \
            'experiments', 'debug', 'exp-7-2_experiment')
        payment_rules = ['k_price', 'vcg']
        for payment_rule in payment_rules:
            experiment_config, experiment_class = \
                ConfigurationManager(
                    experiment_type='double_auction_single_item_uniform_symmetric',
                    n_runs=10,  # repeat exp. for different random seeds
                    n_epochs=4000,
                ) \
                .set_setting(
                    payment_rule=payment_rule,
                    k=0.5,
                    n_players = 6,
                    n_buyers = 3,
                    n_sellers = 3
                ) \
                .set_learning(
                    batch_size=2**18,  # default value -> may need to be decreased for larger markets
                    model_sharing=True,
                    # hidden_nodes=[100],
                    # hidden_activations=[nn.SELU()],
                    # dropout=0.1,
                ) \
                .set_hardware(
                    specific_gpu=6,
                ) \
                .set_logging(
                    eval_batch_size=2**22,  # needed for exact utility-loss (epsilon_relative)
                    cache_eval_actions=True,

                    log_metrics={
                        'opt': True,
                        'util_loss': True,
                    },

                    # needed for estimated utility-loss (estimated_relative_ex_ante_util_loss)
                    util_loss_batch_size=2**10,  # default value -> may needs to be decreased for larger markets
                    util_loss_grid_size=2**10,  # default value -> may needs to be decreased for larger markets
                    util_loss_frequency=2000,  # don't want to calculate that often as it takes long

                    best_response=True,  # only needed for best response plots
                    log_root_dir=log_root_dir,
                    save_tb_events_to_csv_detailed=True,
                    save_models=True,  # needed if you want to plot bid functions afterward
                    plot_frequency=500,  # don't want to waste much disk space
                ) \
                .get_config()
            experiment = experiment_class(experiment_config)
            experiment.run()
            torch.cuda.empty_cache()
    
    ### EXP-7-3 four buyers and sellers -------------------------------------------
    if False:
        log_root_dir = os.path.join(os.path.expanduser('~'), 'bnelearn', \
            'experiments', 'debug', 'exp-7-3_experiment')
        payment_rules = ['k_price', 'vcg']
        for payment_rule in payment_rules:
            experiment_config, experiment_class = \
                ConfigurationManager(
                    experiment_type='double_auction_single_item_uniform_symmetric',
                    n_runs=10,  # repeat exp. for different random seeds
                    n_epochs=4000,
                ) \
                .set_setting(
                    payment_rule=payment_rule,
                    k=0.5,
                    n_players = 8,
                    n_buyers = 4,
                    n_sellers = 4
                ) \
                .set_learning(
                    batch_size=2**18,  # default value -> may need to be decreased for larger markets
                    model_sharing=True,
                    # hidden_nodes=[100],
                    # hidden_activations=[nn.SELU()],
                    # dropout=0.1,
                ) \
                .set_hardware(
                    specific_gpu=7,
                ) \
                .set_logging(
                    eval_batch_size=2**22,  # needed for exact utility-loss (epsilon_relative)
                    cache_eval_actions=True,

                    log_metrics={
                        'opt': True,
                        'util_loss': True,
                    },

                    # needed for estimated utility-loss (estimated_relative_ex_ante_util_loss)
                    util_loss_batch_size=2**10,  # default value -> may needs to be decreased for larger markets
                    util_loss_grid_size=2**10,  # default value -> may needs to be decreased for larger markets
                    util_loss_frequency=2000,  # don't want to calculate that often as it takes long

                    best_response=True,  # only needed for best response plots
                    log_root_dir=log_root_dir,
                    save_tb_events_to_csv_detailed=True,
                    save_models=True,  # needed if you want to plot bid functions afterward
                    plot_frequency=500,  # don't want to waste much disk space
                ) \
                .get_config()
            experiment = experiment_class(experiment_config)
            experiment.run()
            torch.cuda.empty_cache()

    ### EXP-9 known BNE for risk -------------------------------------------
    if False:
        log_root_dir = os.path.join(os.path.expanduser('~'), 'bnelearn', \
            'experiments', 'debug', 'exp-9_experiment')
        risks = [i/10. for i in range(1, 11)]
        for risk in risks:
            experiment_config, experiment_class = \
                ConfigurationManager(
                    experiment_type='double_auction_single_item_uniform_symmetric',
                    n_runs=5,  # repeat exp. for different random seeds
                    n_epochs=2000,
                ) \
                .set_setting(
                    payment_rule='k_price',
                    k=0.5,
                    risk=risk  # 1.0 is default (risk-neutral)
                ) \
                .set_learning(
                    batch_size=2**18,  # default value -> may need to be decreased for larger markets
                    model_sharing=True,
                ) \
                .set_hardware(
                    specific_gpu=6,
                ) \
                .set_logging(
                    eval_batch_size=2**22,  # needed for exact utility-loss (epsilon_relative)
                    cache_eval_actions=True,

                    # needed for estimated utility-loss (estimated_relative_ex_ante_util_loss)
                    util_loss_batch_size=2**13,  # default value -> may needs to be decreased for larger markets
                    util_loss_grid_size=2**10,  # default value -> may needs to be decreased for larger markets
                    util_loss_frequency=2000,  # don't want to calculate that often as it takes long

                    best_response=True,  # only needed for best response plots
                    log_root_dir=log_root_dir,
                    save_tb_events_to_csv_detailed=True,
                    save_models=True,  # needed if you want to plot bid functions afterward
                    plot_frequency=200,  # don't want to waste much disk space
                ) \
                .get_config()
            experiment = experiment_class(experiment_config)
            experiment.run()
            torch.cuda.empty_cache()
