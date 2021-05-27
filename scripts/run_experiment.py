"""
Script for reproducing the reported experiments

NOTE: if you haven't installed bnelearn via pip, this script needs to be called
      from the bnelearn root directory, rather than from /scripts/
"""
import os
import sys
from itertools import product
import torch

sys.path.append(os.path.realpath('.'))

from bnelearn.experiment.configuration_manager import ConfigurationManager


if __name__ == '__main__':

    # User parameters
    n_epochs = 1000
    n_runs = 1
    model_sharing = False
    pretrain_iters = 500
    batch_size = 2**17

    risks = [i/10 for i in range(1, 11)]
    gammas = [i/10 for i in range(11)]
    payment_rules = ['nearest_vcg', 'first_price']
        # Alternatives: 'vcg', 'nearest_bid', 'nearest_zero', 'nearest_vcg'
    corr_models = ['constant_weights', 'Bernoulli_weights']
        # Alternatives: 'independent'

    eval_batch_size = 2**17
    util_loss_frequency = 1000
    util_loss_batch_size = 2**12
    util_loss_grid_size = 2**10
    stopping_criterion_frequency = 100000  # don't use

    specific_gpu = 0
    log_root_dir = os.path.join(
        os.path.expanduser('~'), 'bnelearn', 'experiments',
    )

    # Run LLG nearest-vcg for different risks / correlations
    for prod in product(risks, gammas, payment_rules, corr_models):
        risk, gamma, payment_rule, corr_model = prod

        experiment_config, experiment_class = \
            ConfigurationManager(
                experiment_type='llg',
                n_runs=n_runs,
                n_epochs=n_epochs,
            ) \
            .set_setting(
                payment_rule=payment_rule,
                gamma=gamma,
                correlation_types=corr_model,
                risk=risk,
            ) \
            .set_learning(
                batch_size=batch_size,
                pretrain_iters=pretrain_iters,
                model_sharing=model_sharing,
            ) \
            .set_logging(
                log_root_dir=log_root_dir,
                util_loss_frequency=util_loss_frequency,
                util_loss_batch_size=util_loss_batch_size,
                util_loss_grid_size=util_loss_grid_size,
                eval_batch_size=eval_batch_size,
                stopping_criterion_frequency=stopping_criterion_frequency,
                log_metrics={
                    'opt': True,
                    'util_loss': True,
                    'efficiency': True,
                    'revenue': True,
                },
            ) \
            .set_hardware(
                specific_gpu=specific_gpu,
            ) \
            .get_config()
        experiment = experiment_class(experiment_config)
        experiment.run()

        torch.cuda.empty_cache()
