# """
# This file tests the run of experiments at runtime, so simply whether it technically completes the run.
# It chooses only 2 runs, with 3 epochs but plots and logs every period. Logging and plotting is not written to disc.
# It considers each implemented experiment with each payment rule. Fo each experiment no model_sharing is tested once.
# TODO:
#     - Paul: InputLength should become obsolete Stefan: yes!
#     - Paul: Disable logging (by a variable that is checked just before writing)
#     - Paul: Maybe later: Add bne metric to LLLLGG and test for failure
#     - Stefan: Later: gaussian with fpsb and regret
#     - Stefan: some experiments fail with too low batch size (utility mismatch), what to do with those
# """
# import pytest
# import sys
# import os
# sys.path.append(os.path.realpath('.'))
# from bnelearn.experiment.run_experiment \
#     import run_single_item_uniform_symmetric, run_single_item_gaussian_symmetric, \
#            run_single_item_asymmetric_uniform, run_llg, run_llllgg, run_multiunit, run_splitaward
# from bnelearn.experiment.configurations import LearningConfiguration
# from bnelearn.experiment.gpu_controller import GPUController

# ids_single_item, testdata_single_item = zip(*[
#     # Single item
#     ['single_item-symmetric-uniform-fp', (run_single_item_uniform_symmetric(2,3, [2], 'first_price'),1)],
#     ['single_item-symmetric-uniform-fp-no_model_sharing',
#         (run_single_item_uniform_symmetric(2,3, [3], 'first_price',model_sharing=False),1)],
#     ['single_item-symmetric-uniform-vcg', (run_single_item_uniform_symmetric(2,3, [3], 'second_price'),1)],
#     ['single_item-symmetric-gaussian-fp', (run_single_item_gaussian_symmetric(2,3, [4], 'first_price'),1)],
#     ['single_item-symmetric-gaussian-vcg', (run_single_item_gaussian_symmetric(2,3, [5], 'second_price'),1)],
#     ['single_item-asymmetric-uniform-fp', (run_single_item_asymmetric_uniform(2,3, 'first_price'),1)],
#     ['single_item-asymmetric-uniform-vcg', (run_single_item_asymmetric_uniform(2,3, 'second_price'),1)]
#     ])
# ids_local_global, testdata_local_global = zip(*[
#     # LLG
#     ['LLG-fp', (run_llg(2,3,'first_price',log_metrics=['regret']),1)],
#     ['LLG-fp-no_model_sharing', (run_llg(2,3,'first_price', model_sharing=False,log_metrics=['regret']),1)],
#     ['LLG-vcg', (run_llg(2,3,'vcg'),1)],
#     ['LLG-nearest_bid', (run_llg(2,3,'nearest_bid'),1)],
#     ['LLG-nearest_zero', (run_llg(2,3,'nearest_zero'),1)],
#     ['LLG-nearest_vcg', (run_llg(2,3,'nearest_vcg'),1)],
#     # LLLLGG
#     ['LLLLGG-fp', (run_llllgg(2,3,'first_price'),2)],
#     ['LLLLGG-fp-no_model_sharing', (run_llllgg(2,3,'first_price',model_sharing=False),2)],
#     ['LLLLGG-vcg', (run_llllgg(2,3,'vcg'),2)],
#     ['LLLLGG-nearest_vcg', (run_llllgg(2,3,'nearest_vcg',core_solver='gurobi'),2)]
#     ])
# ids_multi_unit, testdata_multi_unit = zip(*[
#     # MultiUnit
#     ['MultiUnit-discr', (run_multiunit(2, 3, [2], 'discriminatory'),2)],
#     ['MultiUnit-discr-no_model_sharing', (run_multiunit(2, 3, [2], 'discriminatory',model_sharing=False),2)],
#     ['MultiUnit-vcg', (run_multiunit(2, 3, [2], 'vcg'),2)],
#     ['MultiUnit-vcg', (run_multiunit(2, 3, [2], 'uniform'),2)],
#     ['SplitAward-fp', (run_splitaward(2, 3, [2]),2)],
#     ['SplitAward-fp-no_model_sharing', (run_splitaward(2, 3, [2], model_sharing=False),2)]
#     ])

# def run_auction_test(create_auction_function, input_length):
#     learning_configuration = LearningConfiguration(input_length=input_length, pretrain_iters=20)
#     running_configuration, logging_configuration, experiment_configuration, experiment_class = create_auction_function
#     # Log and plot frequent but few
#     logging_configuration.plot_frequency=1
#     logging_configuration.regret_frequency=1
#     logging_configuration.plot_points=10
#     logging_configuration.regret_batch_size = 2**2
#     logging_configuration.regret_grid_size = 2**2
#     learning_configuration.batch_size = 2**2
#     experiment_configuration.n_players = running_configuration.n_players[0]
#     experiment_configuration.test_run = True
#     # Create and run the experiment
#     gpu_configuration = GPUController(specific_gpu=2)
#     experiment = experiment_class(experiment_configuration, learning_configuration,
#                                     logging_configuration, gpu_configuration)
#     experiment.run(epochs=running_configuration.n_epochs, n_runs=running_configuration.n_runs)

# @pytest.mark.parametrize("auction_function_with_params, input_length", testdata_single_item, ids=ids_single_item)
# def test_single_item_auctions(auction_function_with_params, input_length):
#     run_auction_test(auction_function_with_params,input_length)

# @pytest.mark.parametrize("auction_function_with_params, input_length", testdata_local_global, ids=ids_local_global)
# def test_local_global_auctions(auction_function_with_params, input_length):
#     run_auction_test(auction_function_with_params,input_length)

# @pytest.mark.parametrize("auction_function_with_params, input_length", testdata_multi_unit, ids=ids_multi_unit)
# def test_multi_unit_auctions(auction_function_with_params, input_length):
#     run_auction_test(auction_function_with_params,input_length)
