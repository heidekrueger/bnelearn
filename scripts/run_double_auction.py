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

    log_root_dir = os.path.join(os.path.expanduser('~'), 'bnelearn', 'experiments', 'debug')

    r"""
    For the final experiments of double-auctions, we want to report (at least)
    the following metrics:
    ```
    ALIASES = {
        'eval/L_2':                                  '$L_2$',  # aka RMSE
        'eval/epsilon_relative':                     '$\mathcal{L}$',
        'eval/estimated_relative_ex_ante_util_loss': '$\hat{\mathcal{L}}$',
    }
    ```
    These can be found in `log_root_dir\<exp_dir>\log_root_dir.csv`.
    """

    # loop over the different paramters you want to try out -> extend & varry!
    model_sharings = [True, False]
    n_buyers = range(1, 3)
    n_sellers = range(1, 3)
    ks = [0.0, 0.5, 1.0]
    # ...

    for (model_sharing, n_buyer, n_seller, k) in product(model_sharings, n_buyers, n_sellers, ks):

        experiment_config, experiment_class = \
            ConfigurationManager(
                experiment_type='double_auction_single_item_uniform_symmetric',
                n_runs=10,  # repeat exp. for 10 different random seeds
                n_epochs=2000,
            ) \
            .set_setting(
                payment_rule='k_price',
                k=k,
                # TODO
                # n_buyer=n_buyer,
                # n_seller=n_seller,
                # risk=1.0  # 1.0 is default (risk-neutral)
            ) \
            .set_learning(
                batch_size=2**18,  # default value -> may needs to be decreased for larger markets
                model_sharing=model_sharing,
                # pretrain_iters=500,
                learner_hyperparams={
                    'population_size': 64,
                    'sigma': 1.,
                    'scale_sigma_by_model_size': True
                }
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
                util_loss_frequency=1000,  # don't want to calculate that often as it takes long

                best_response=True,  # only needed for best response plots
                log_root_dir=log_root_dir,
                save_tb_events_to_csv_detailed=True,
                stopping_criterion_frequency=1e8,  # don't want to use this
                save_models=True,  # needed if you want to plot bid functions afterward
                plot_frequency=1000,  # don't want to waste much disk space
            ) \
            .get_config()
        experiment = experiment_class(experiment_config)
        experiment.run()
        torch.cuda.empty_cache()
