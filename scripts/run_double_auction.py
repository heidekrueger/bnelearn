"""
Runs predefined double auction experiments with individual parameters
"""
import os
import sys
from itertools import product
import torch

# pylint: disable=wrong-import-position
sys.path.append(os.path.realpath('.'))
sys.path.append(os.path.join(os.path.expanduser('~'), 'bnelearn'))

from bnelearn.experiment.configuration_manager import ConfigurationManager  # pylint: disable=import-error


if __name__ == '__main__':

    ### EXP-1 BB 1/2-DA & VCG -------------------------------------------------
    if False:
        log_root_dir = os.path.join(os.path.expanduser('~'), 'bnelearn', \
            'experiments', 'debug', 'exp-1_experiment')
        payment_rules = ['k_price', 'vcg']
        for payment_rule in payment_rules:
            experiment_config, experiment_class = \
                ConfigurationManager(
                    experiment_type='double_auction_single_item_uniform_symmetric',
                    n_runs=10,  # repeat exp. for 10 different random seeds
                    n_epochs=1000,
                ) \
                .set_setting(
                    payment_rule=payment_rule,
                    k=0.5,
                ) \
                .set_learning(
                    batch_size=2**18,  # default value -> may need to be decreased for larger markets
                    model_sharing=True,
                ) \
                .set_hardware(
                    specific_gpu=7,
                ) \
                .set_logging(
                    eval_batch_size=2**22,  # needed for exact utility-loss (epsilon_relative)
                    cache_eval_actions=True,

                    # needed for estimated utility-loss (estimated_relative_ex_ante_util_loss)
                    util_loss_batch_size=2**12,  # default value -> may needs to be decreased for larger markets
                    util_loss_grid_size=2**10,  # default value -> may needs to be decreased for larger markets
                    util_loss_frequency=200,  # don't want to calculate that often as it takes long

                    best_response=True,  # only needed for best response plots
                    log_root_dir=log_root_dir,
                    save_tb_events_to_csv_detailed=True,
                    save_models=True,  # needed if you want to plot bid functions afterward
                    plot_frequency=1000,  # don't want to waste much disk space
                ) \
                .get_config()
            experiment = experiment_class(experiment_config)
            experiment.run()
            torch.cuda.empty_cache()

    ### EXP-2 risk experiments ------------------------------------------------
    if False:
        log_root_dir = os.path.join(os.path.expanduser('~'), 'bnelearn', \
            'experiments', 'debug', 'exp-2_experiment')
        risks = [i/10. for i in range(1, 11, 3)]
        for risk in risks:
            experiment_config, experiment_class = \
                ConfigurationManager(
                    experiment_type='double_auction_single_item_uniform_symmetric',
                    n_runs=10,  # repeat exp. for 10 different random seeds
                    n_epochs=1000,
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
                    util_loss_batch_size=2**12,  # default value -> may needs to be decreased for larger markets
                    util_loss_grid_size=2**10,  # default value -> may needs to be decreased for larger markets
                    util_loss_frequency=1000,  # don't want to calculate that often as it takes long

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

    ### EXP-3 different pretraining -------------------------------------------
    if True:
        log_root_dir = os.path.join(os.path.expanduser('~'), 'bnelearn', \
            'experiments', 'debug', 'exp-3_experiment')
        pretrainings = ['transform-1', 'transform-2']
        for pretraining in pretrainings:
            experiment_config, experiment_class = \
                ConfigurationManager(
                    experiment_type='double_auction_single_item_uniform_symmetric',
                    n_runs=1,  # repeat exp. for 10 different random seeds
                    n_epochs=3000,
                    seeds=[69]
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
                    specific_gpu=6,
                ) \
                .set_logging(
                    eval_batch_size=2**18,  # needed for exact utility-loss (epsilon_relative)
                    cache_eval_actions=True,

                    # needed for estimated utility-loss (estimated_relative_ex_ante_util_loss)
                    util_loss_batch_size=2**12,  # default value -> may needs to be decreased for larger markets
                    util_loss_grid_size=2**10,  # default value -> may needs to be decreased for larger markets
                    util_loss_frequency=1000,  # don't want to calculate that often as it takes long

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
