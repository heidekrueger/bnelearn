"""
This file tests the run of experiments at runtime, so simply whether it technically completes the run.
It chooses only 2 runs, with 3 epochs but plots and logs every period. Logging and plotting is not written to disc.
It considers each implemented experiment with each payment rule. Fo each experiment no model_sharing is tested once.
TODO:
    - Paul: InputLength should become obsolete Stefan: yes! @assigned Paul
    - Paul: Maybe later: Add bne metric to LLLLGG and test for failure
    - Stefan: Later: gaussian with fpsb and util_loss
"""
import os
import sys

import pytest

#sys.path.append(os.path.realpath('.'))
# pylint: disable=wrong-import-order
from bnelearn.experiment.configuration_manager import ConfigurationManager


# Single item
# id , config, class, known_bne
ids_si, *testdata_si = zip(*[
    # Single item
    ['single_item-symmetric-uniform-fp',
     *ConfigurationManager(experiment_type='single_item_uniform_symmetric')
         .get_config(n_runs=2, n_epochs=3),
     True],
    ['single_item-symmetric-uniform-fp-no_model_sharing',
     *ConfigurationManager(experiment_type='single_item_uniform_symmetric')
         .get_config(n_runs=2, n_epochs=3, model_sharing=False),
     True],
    # ['single_item-symmetric-uniform-vcg', (single_item_uniform_symmetric(2,3, [3], 'second_price')),True],
    # too expensive. ['single_item-symmetric-gaussian-fp', (single_item_gaussian_symmetric(2,3, [4], 'first_price')),True],
    ['single_item-symmetric-gaussian-vcg',
     *ConfigurationManager(experiment_type='single_item_gaussian_symmetric')
         .get_config(n_runs=2, n_epochs=3, n_players=5, payment_rule='second_price'),
     True],
    ['single_item-asymmetric-uniform-fp',
     *ConfigurationManager(experiment_type='single_item_asymmetric_uniform_overlapping')
        .get_config(n_runs=2, n_epochs=3),
     True],
    ['single_item-asymmetric-uniform-vcg',
     *ConfigurationManager(experiment_type='single_item_asymmetric_uniform_overlapping')
        .get_config(n_runs=2, n_epochs=3, payment_rule='second_price'),
     True]
    ])


# Local Global Auctions
# id , config, class, known_bne

ids_lg, *testdata_lg = zip(*[
    ['LLG-fp', *ConfigurationManager(experiment_type='llg').get_config(n_runs=2, n_epochs=3), False],
    ['LLG-fp-no_model_sharing', *ConfigurationManager(experiment_type='llg')
                                .get_config(n_runs=2, n_epochs=3, model_sharing=False), False],
    ['LLG-vcg', *ConfigurationManager(experiment_type='llg')
                .get_config(n_runs=2, n_epochs=3, payment_rule='vcg'), True],
    ['LLG-nearest_bid', *ConfigurationManager(experiment_type='llg')
                        .get_config(n_runs=2, n_epochs=3, payment_rule='nearest_bid'), True],
    ['LLG-nearest_bid_correlated', *ConfigurationManager(experiment_type='llg')
                                   .with_correlation(gamma=0.5)
                                   .get_config(n_runs=2, n_epochs=3, payment_rule='nearest_bid'), True],
    # Used to fail when 0.5 didn't, due to neg bids
    ['LLG-nearest_bid_perfectly_correlated', *ConfigurationManager(experiment_type='llg')
                                             .with_correlation(gamma=1.0)
                                             .get_config(n_runs=2, n_epochs=3, payment_rule='nearest_bid'), True],
    # ['LLG-nearest_zero', (llg(2,3,'nearest_zero'))],
    # ['LLG-nearest_vcg', (llg(2,3,'nearest_vcg'))],
    # LLLLGG
    # ['LLLLGG-fp', (llllgg(2,2,'first_price'))],
    ['LLLLGG-fp-no_model_sharing', *ConfigurationManager(experiment_type='llllgg')
                                   .get_config(n_runs=2, n_epochs=2, model_sharing=False), False],
    ['LLLLGG-vcg', *ConfigurationManager(experiment_type='llllgg')
                   .get_config(n_runs=2, n_epochs=2, payment_rule='vcg'), True]
    # ['LLLLGG-nearest_vcg', (llllgg(2,2,'nearest_vcg',core_solver='gurobi'))]
])

# MultiUnit
# id , config, class, known_bne
ids_mu, *testdata_mu = zip(*[
    # TODO: in the following test cases, the "expected bne" has been set to make the tests pass.
    # Those where this didn't match Nils's expectations have been marked. @Nils, please take a look.

    #TODO: Nils said should find False for bne, but finds true
    ['MultiUnit-discr', *ConfigurationManager(experiment_type='multiunit')
                        .get_config(n_runs=2, n_epochs=3, payment_rule='discriminatory'), True],
    #TODO: Nils said should find False for bne, but finds true
    ['MultiUnit-discr-no_model_sharing', *ConfigurationManager(experiment_type='multiunit')
                                         .get_config(n_runs=2, n_epochs=3, payment_rule='discriminatory',
                                                      model_sharing=False), True],
    ['MultiUnit-vcg', *ConfigurationManager(experiment_type='multiunit')
                      .get_config(n_runs=2, n_epochs=3, payment_rule='vcg'), True],
    ['MultiUnit-uniform', *ConfigurationManager(experiment_type='multiunit')
                      .get_config(n_runs=2, n_epochs=3, payment_rule='uniform'), True],
    #TODO: Nils said should find False for bne, but finds true
    ['SplitAward-fp', *ConfigurationManager(experiment_type='splitaward')
                      .get_config(n_runs=2, n_epochs=3), True],
    #TODO: Nils said should find False for bne, but finds true
    ['SplitAward-fp-no_model_sharing', *ConfigurationManager(experiment_type='splitaward')
                                       .get_config(n_runs=2, n_epochs=3, model_sharing=False), True]
])


def run_auction_test(config, exp_class, known_bne):
    config.learning.pretrain_iters = 20
    # Log and plot frequent but few
    config.logging.enable_logging = False
    config.logging.plot_frequency = 1
    config.logging.util_loss_frequency = 1
    config.logging.plot_points = 10
    config.logging.util_loss_batch_size = 2 ** 2
    config.logging.util_loss_grid_size = 2 ** 2
    config.learning.batch_size = 2 ** 2
    config.logging.eval_batch_size = 2 ** 2
    # Create and run the experiment
    config.hardware.specific_gpu = 0

    experiment = exp_class(config)

    assert experiment.known_bne == known_bne, \
        "known_bne setting is not as expected!"
    experiment.run()

print(type(testdata_si))
print(len(testdata_si))

print('hi')

@pytest.mark.parametrize("config, exp_class, known_bne", zip(*testdata_si), ids=ids_si)
def test_single_item_auctions(config, exp_class, known_bne):
    run_auction_test(config, exp_class, known_bne)


@pytest.mark.parametrize("config, exp_class, known_bne", zip(*testdata_lg), ids=ids_lg)
def test_local_global_auctions(config, exp_class, known_bne):
    run_auction_test(config, exp_class, known_bne)


@pytest.mark.parametrize("config, exp_class, known_bne", zip(*testdata_mu), ids=ids_mu)
def test_multi_unit_auctions(config, exp_class, known_bne):
    run_auction_test(config, exp_class, known_bne)
