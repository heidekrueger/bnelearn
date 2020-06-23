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
from bnelearn.experiment.configuration_manager import ConfigurationManager

sys.path.append(os.path.realpath('.'))

ids_single_item, testdata_single_item = zip(*[
    # Single item
    ['single_item-symmetric-uniform-fp',
     ConfigurationManager(experiment_type='single_item_uniform_symmetric')
                                            .set_running(n_runs=2, n_epochs=3)
                                            .get_config()],
    ['single_item-symmetric-uniform-fp-no_model_sharing',
     ConfigurationManager(experiment_type='single_item_uniform_symmetric')
                                            .set_running(n_runs=2, n_epochs=3)
                                            .set_learning(model_sharing=False)
                                            .get_config()],
    # ['single_item-symmetric-uniform-vcg', (single_item_uniform_symmetric(2,3, [3], 'second_price'))],
    # too expensive. ['single_item-symmetric-gaussian-fp', (single_item_gaussian_symmetric(2,3, [4], 'first_price'))],
    ['single_item-symmetric-gaussian-vcg', ConfigurationManager(experiment_type='single_item_gaussian_symmetric')
                                            .set_running(n_runs=2, n_epochs=3)
                                            .set_setting(n_players=5, payment_rule='second_price')
                                            .get_config()],
    ['single_item-asymmetric-uniform-fp',
     ConfigurationManager(experiment_type='single_item_asymmetric_uniform_overlapping')
                                            .set_running(n_runs=2, n_epochs=3)
                                            .get_config()],
    ['single_item-asymmetric-uniform-vcg',
     ConfigurationManager(experiment_type='single_item_asymmetric_uniform_overlapping')
                                            .set_running(n_runs=2, n_epochs=3)
                                            .set_setting(payment_rule='second_price')
                                            .get_config()]
])
ids_local_global, testdata_local_global = zip(*[
    # LLG
    ['LLG-fp', ConfigurationManager(experiment_type='llg')
                                              .set_running(n_runs=2, n_epochs=3)
                                              .get_config()],
    ['LLG-fp-no_model_sharing', ConfigurationManager(experiment_type='llg')
                                              .set_running(n_runs=2, n_epochs=3)
                                              .set_learning(model_sharing=False)
                                              .get_config()],
    ['LLG-vcg', ConfigurationManager(experiment_type='llg')
                                              .set_running(n_runs=2, n_epochs=3)
                                              .set_setting(payment_rule='vcg')
                                              .get_config()],
    ['LLG-nearest_bid', ConfigurationManager(experiment_type='llg')
                                              .set_running(n_runs=2, n_epochs=3)
                                              .set_setting(payment_rule='nearest_bid')
                                              .get_config()],
    ['LLG-nearest_bid_correlated', ConfigurationManager(experiment_type='llg')
                                              .set_running(n_runs=2, n_epochs=3)
                                              .set_setting(gamma=0.5, payment_rule='nearest_bid')
                                              .get_config()],
    # Used to fail when 0.5 didn't, due to neg bids
    ['LLG-nearest_bid_perfectly_correlated', ConfigurationManager(experiment_type='llg')
                                              .set_running(n_runs=2, n_epochs=3)
                                              .set_setting(gamma=1.0, payment_rule='nearest_bid')
                                              .get_config()],
    # ['LLG-nearest_zero', (llg(2,3,'nearest_zero'))],
    # ['LLG-nearest_vcg', (llg(2,3,'nearest_vcg'))],
    # LLLLGG
    # ['LLLLGG-fp', (llllgg(2,2,'first_price'))],
    ['LLLLGG-fp-no_model_sharing', ConfigurationManager(experiment_type='llllgg')
                                              .set_running(n_runs=2, n_epochs=2)
                                              .set_learning(model_sharing=False)
                                              .get_config()],
    ['LLLLGG-vcg', ConfigurationManager(experiment_type='llllgg')
                                              .set_running(n_runs=2, n_epochs=2)
                                              .set_setting(payment_rule='vcg')
                                              .get_config()]
    # ['LLLLGG-nearest_vcg', (llllgg(2,2,'nearest_vcg',core_solver='gurobi'))]
])
ids_multi_unit, testdata_multi_unit = zip(*[
    # MultiUnit
    ['MultiUnit-discr', ConfigurationManager(experiment_type='multiunit')
                                          .set_running(n_runs=2, n_epochs=3)
                                          .set_setting(payment_rule='discriminatory')
                                          .get_config()],
    ['MultiUnit-discr-no_model_sharing', ConfigurationManager(experiment_type='multiunit')
                                          .set_running(n_runs=2, n_epochs=3)
                                          .set_setting(payment_rule='discriminatory')
                                          .set_learning(model_sharing=False)
                                          .get_config()],
    ['MultiUnit-vcg', ConfigurationManager(experiment_type='multiunit')
                                          .set_running(n_runs=2, n_epochs=2)
                                          .set_setting(payment_rule='vcg')
                                          .get_config()],
    ['MultiUnit-vcg', ConfigurationManager(experiment_type='multiunit')
                                          .set_running(n_runs=2, n_epochs=2)
                                          .set_setting(payment_rule='uniform')
                                          .get_config()],
    ['SplitAward-fp', ConfigurationManager(experiment_type='splitaward')
                                          .set_running(n_runs=2, n_epochs=3)
                                          .get_config()],
    ['SplitAward-fp-no_model_sharing', ConfigurationManager(experiment_type='splitaward')
                                          .set_running(n_runs=2, n_epochs=3)
                                          .set_learning(model_sharing=False)
                                          .get_config()]
])


def run_auction_test(create_auction_function):
    experiment_configuration, experiment_class = create_auction_function
    experiment_configuration.learning.pretrain_iters = 20
    # Log and plot frequent but few
    experiment_configuration.logging.enable_logging = False
    experiment_configuration.logging.plot_frequency = 1
    experiment_configuration.logging.util_loss_frequency = 1
    experiment_configuration.logging.plot_points = 10
    experiment_configuration.logging.util_loss_batch_size = 2 ** 2
    experiment_configuration.logging.util_loss_grid_size = 2 ** 2
    experiment_configuration.learning.batch_size = 2 ** 2
    experiment_configuration.logging.eval_batch_size = 2 ** 2
    # Create and run the experiment
    experiment_configuration.hardware.specific_gpu = 0
    experiment_class(experiment_configuration).run()


@pytest.mark.parametrize("auction_function_with_params", testdata_single_item, ids=ids_single_item)
def test_single_item_auctions(auction_function_with_params):
    run_auction_test(auction_function_with_params)


@pytest.mark.parametrize("auction_function_with_params", testdata_local_global, ids=ids_local_global)
def test_local_global_auctions(auction_function_with_params):
    run_auction_test(auction_function_with_params)


@pytest.mark.parametrize("auction_function_with_params", testdata_multi_unit, ids=ids_multi_unit)
def test_multi_unit_auctions(auction_function_with_params):
    run_auction_test(auction_function_with_params)
