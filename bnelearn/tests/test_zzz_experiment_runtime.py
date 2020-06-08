"""
This file tests the run of experiments at runtime, so simply whether it technically completes the run.
It chooses only 2 runs, with 3 epochs but plots and logs every period. Logging and plotting is not written to disc.
It considers each implemented experiment with each payment rule. Fo each experiment no model_sharing is tested once.
TODO:
    - : InputLength should become obsolete : yes! @assigned 
    - : Maybe later: Add bne metric to LLLLGG and test for failure
    - : Later: gaussian with fpsb and util_loss
"""
import os
import sys

import pytest

from bnelearn.experiment.configurations import LearningConfiguration
from bnelearn.experiment.gpu_controller import GPUController
from bnelearn.experiment.presets import (llg, llllgg, multiunit,
                                         single_item_asymmetric_uniform_overlapping,
                                         single_item_gaussian_symmetric,
                                         single_item_uniform_symmetric,
                                         splitaward)

sys.path.append(os.path.realpath('.'))

ids_single_item, testdata_single_item = zip(*[
    # Single item
    ['single_item-symmetric-uniform-fp', (single_item_uniform_symmetric(2,3, [2], 'first_price'))],
    ['single_item-symmetric-uniform-fp-no_model_sharing',
        (single_item_uniform_symmetric(2,3, [3], 'first_price',model_sharing=False))],
    #['single_item-symmetric-uniform-vcg', (single_item_uniform_symmetric(2,3, [3], 'second_price'))],
    # too expensive. ['single_item-symmetric-gaussian-fp', (single_item_gaussian_symmetric(2,3, [4], 'first_price'))],
    ['single_item-symmetric-gaussian-vcg', (single_item_gaussian_symmetric(2,3, [5], 'second_price'))],
    ['single_item-asymmetric-uniform-fp', (single_item_asymmetric_uniform_overlapping(2,3, 'first_price'))],
    ['single_item-asymmetric-uniform-vcg', (single_item_asymmetric_uniform_overlapping(2,3, 'second_price'))]
    ])
ids_local_global, testdata_local_global = zip(*[
    # LLG
    ['LLG-fp', (llg(2,3,'first_price',log_metrics=['util_loss']))],
    ['LLG-fp-no_model_sharing', (llg(2,3,'first_price', model_sharing=False,log_metrics=['util_loss']))],
    ['LLG-vcg', (llg(2,3,'vcg'))],
    ['LLG-nearest_bid', (llg(2,3,'nearest_bid'))],
    #['LLG-nearest_zero', (llg(2,3,'nearest_zero'))],
    #['LLG-nearest_vcg', (llg(2,3,'nearest_vcg'))],
    # LLLLGG
    #['LLLLGG-fp', (llllgg(2,2,'first_price'))],
    ['LLLLGG-fp-no_model_sharing', (llllgg(2,2,'first_price',model_sharing=False))],
    ['LLLLGG-vcg', (llllgg(2,2,'vcg'))],
    #['LLLLGG-nearest_vcg', (llllgg(2,2,'nearest_vcg',core_solver='gurobi'))]
    ])
ids_multi_unit, testdata_multi_unit = zip(*[
    # MultiUnit
    ['MultiUnit-discr', (multiunit(2, 3, [2], 'discriminatory'))],
    ['MultiUnit-discr-no_model_sharing', (multiunit(2, 3, [2], 'discriminatory',model_sharing=False))],
    ['MultiUnit-vcg', (multiunit(2, 3, [2], 'vcg'))],
    ['MultiUnit-vcg', (multiunit(2, 3, [2], 'uniform'))],
    ['SplitAward-fp', (splitaward(2, 3, [2]))],
    ['SplitAward-fp-no_model_sharing', (splitaward(2, 3, [2], model_sharing=False))]
    ])

def run_auction_test(create_auction_function):
    learning_configuration = LearningConfiguration(pretrain_iters=20)
    running_configuration, logging_configuration, experiment_configuration, experiment_class = create_auction_function
    # Log and plot frequent but few
    logging_configuration.enable_logging = False
    logging_configuration.plot_frequency=1
    logging_configuration.util_loss_frequency=1
    logging_configuration.plot_points=10
    logging_configuration.util_loss_batch_size = 2**2
    logging_configuration.util_loss_grid_size = 2**2
    learning_configuration.batch_size = 2**2
    experiment_configuration.n_players = running_configuration.n_players[0]
    # Create and run the experiment
    gpu_configuration = GPUController(specific_gpu=0)
    experiment = experiment_class(experiment_configuration, learning_configuration,
                                    logging_configuration, gpu_configuration)
    experiment.run(epochs=running_configuration.n_epochs, n_runs=running_configuration.n_runs)

@pytest.mark.parametrize("auction_function_with_params", testdata_single_item, ids=ids_single_item)
def test_single_item_auctions(auction_function_with_params):
    run_auction_test(auction_function_with_params)

@pytest.mark.parametrize("auction_function_with_params", testdata_local_global, ids=ids_local_global)
def test_local_global_auctions(auction_function_with_params):
    run_auction_test(auction_function_with_params)

@pytest.mark.parametrize("auction_function_with_params", testdata_multi_unit, ids=ids_multi_unit)
def test_multi_unit_auctions(auction_function_with_params):
    run_auction_test(auction_function_with_params)
