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
# from bnelearn.experiment.single_item_experiment import UniformSymmetricPriorSingleItemExperiment, \
#     GaussianSymmetricPriorSingleItemExperiment

# from bnelearn.experiment.combinatorial_experiment import LLGExperiment, LLLLGGExperiment
# import warnings
# gpu_config = GPUController(specific_gpu=0)



# def test_uniform_symmetric_model_sharing():
#     learner_hyperparams = {
#         'population_size': 64,
#         'sigma': 1.,
#         'scale_sigma_by_model_size': True
#     }
#     optimizer_hyperparams = {
#         'lr': 3e-3
#     }
#     experiment_params = {
#         'model_sharing': True,
#         'u_lo': 0,
#         'u_hi': 1,
#         'payment_rule': 'first_price',
#         'risk': 1.0,
#         'regret_batch_size': 2**8,
#         'regret_grid_size': 2**8
#     }

#     input_length = 1
#     hidden_nodes = [5, 5]
#     hidden_activations = [nn.SELU(), nn.SELU()]

#     l_config = LearningConfiguration(learner_hyperparams=learner_hyperparams,
#                                     optimizer_type='adam',
#                                     optimizer_hyperparams=optimizer_hyperparams,
#                                     input_length=input_length,
#                                     hidden_nodes=hidden_nodes,
#                                     hidden_activations=hidden_activations,
#                                     pretrain_iters=50, batch_size=2 ** 18,
#                                     eval_batch_size=2 ** 18,
#                                     cache_eval_actions=False)

#     for n in [2,3]:
#         experiment_params['n_players'] = n
#         experiment = UniformSymmetricPriorSingleItemExperiment(experiment_params, gpu_config=gpu_config, l_config=l_config)
#         experiment.run(epochs=101, n_runs=2)