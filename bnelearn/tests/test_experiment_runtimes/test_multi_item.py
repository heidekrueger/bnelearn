"""DO NOT UNCOMMENT THE TESTS IN THIS FILE ON ORIGIN BEFORE THE FOLLOWING ARE DONE
   OTHERWISE, THE TESTS WILL WRITE LOTS OF DATA TO DISC AS THE GITLAB-RUNNER USER!!!

    TODO:
        * add a DRY RUN option (without writing to disk) to experiment
        * OR make sure this writes to temporary files
        * Especially, disable writing png/svg to disk for tests

    This test file tests whether single_item auction experiments run without
    runtime errors (for extremely cheap experiment parameters)

"""


# import pytest

# import sys
# import torch
# import torch.nn as nn

# from bnelearn.experiment.gpu_controller import GPUController
# from bnelearn.experiment.learning_configuration import LearningConfiguration
# from bnelearn.experiment.logger import SingleItemAuctionLogger
# from bnelearn.experiment.multi_unit_experiment import (
#     MultiItemVickreyAuction2x2,
#     MultiItemUniformPriceAuction2x2,
#     MultiItemUniformPriceAuction2x3limit2,
#     MultiItemDiscriminatoryAuction2x2,
#     MultiItemDiscriminatoryAuction2x2CMV,
#     FPSBSplitAwardAuction2x2
# )

# gpu_config = GPUController(specific_gpu=0)
# learner_hyperparams = {
#     'population_size': 64,
#     'sigma': 1.,
#     'scale_sigma_by_model_size': True
# }
# optimizer_hyperparams = {
#     'lr': 3e-3
# }
# experiment_params = {
#     'risk': 1.0,
#     'regret_batch_size': 2**8,
#     'regret_grid_size': 2**8,
#     'model_sharing': True,
#     'u_lo': [1] * 2,
#     'u_hi': [2] * 2,
# }

# input_length = 2
# hidden_nodes = [2]
# hidden_activations = [nn.SELU()]
# epochs = 2 ** 1

# l_config = LearningConfiguration(
#     learner_hyperparams=learner_hyperparams,
#     optimizer_type='adam',
#     optimizer_hyperparams=optimizer_hyperparams,
#     input_length=input_length,
#     hidden_nodes=hidden_nodes,
#     hidden_activations=hidden_activations,
#     pretrain_iters=16,
#     batch_size=2 ** 18,
#     eval_batch_size=2 ** 18,
#     cache_eval_actions=False
# )


# def test_multi_item_vickrey_2x2():
#     experiment = MultiItemVickreyAuction2x2(
#         experiment_params,
#         gpu_config = gpu_config,
#         l_config = l_config
#     )
#     experiment.run(epochs=epochs, n_runs=1)

# def test_multi_item_uniform_2x2():
#     experiment = MultiItemUniformPriceAuction2x2(
#         experiment_params,
#         gpu_config = gpu_config,
#         l_config = l_config
#     )
#     experiment.run(epochs=epochs, n_runs=1)

# def test_multi_item_uniform_2x3_limit2():
#     experiment = MultiItemUniformPriceAuction2x3limit2(
#         experiment_params,
#         gpu_config = gpu_config,
#         l_config = l_config
#     )
#     experiment.run(epochs=epochs, n_runs=1)

# def test_multi_item_discriminatory_2x2():
#     experiment = MultiItemDiscriminatoryAuction2x2(
#         experiment_params,
#         gpu_config = gpu_config,
#         l_config = l_config
#     )
#     experiment.run(epochs=epochs, n_runs=1)

# def test_multi_item_discriminatory_2x2_cmv():
#     experiment = MultiItemDiscriminatoryAuction2x2CMV(
#         experiment_params,
#         gpu_config = gpu_config,
#         l_config = l_config
#     )
#     experiment.run(epochs=epochs, n_runs=1)

# def test_multi_item_splitaward_2x2():
#     experiment_params['efficiency_parameter'] = 0.3
#     experiment = FPSBSplitAwardAuction2x2(
#         experiment_params,
#         gpu_config = gpu_config,
#         l_config = l_config
#     )
#     experiment.run(epochs=epochs, n_runs=1)
