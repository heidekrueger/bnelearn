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

# import torch.nn as nn

# from bnelearn.experiment.gpu_controller import GPUController
# from bnelearn.experiment.learning_configuration import LearningConfiguration
# from bnelearn.experiment.logger import SingleItemAuctionLogger

# from bnelearn.experiment.combinatorial_experiment import LLGExperiment, LLLLGGExperiment
# import warnings
# gpu_config = GPUController(specific_gpu=0)


# learner_hyperparams = {
#     'population_size': 64,
#     'sigma': 1.,
#     'scale_sigma_by_model_size': True
# }
# optimizer_hyperparams = {
#     'lr': 3e-3
# }
# experiment_params_llg = {
#     # 'model_sharing': set in test functions below
#     'setting': 'LLG',
#     'n_players': 3,
#     'u_lo': 0,
#     'u_hi': [1,1,2],
#     'payment_rule': 'proxy',
#     'risk': 1.0,
#     'regret_batch_size': 2**8,
#     'regret_grid_size': 2**8
# }

# experiment_params_llllgg = {
#     'setting': 'LLLLGG',
#     'payment_rule': 'first_price',
#     'model_sharing': True,
#     'n_players': 6,
#     'u_lo': 0,
#     'u_hi': [1,1,1,1,2,2],
#     'risk': 1.0,
#     'regret_batch_size': 2**8,
#     'regret_grid_size': 2**8
# }

# hidden_nodes = [5, 5]
# hidden_activations = [nn.SELU(), nn.SELU()]

# l_config_llg = LearningConfiguration(learner_hyperparams=learner_hyperparams,
#                                 optimizer_type='adam',
#                                 optimizer_hyperparams=optimizer_hyperparams,
#                                 input_length=1,
#                                 hidden_nodes=hidden_nodes,
#                                 hidden_activations=hidden_activations,
#                                 pretrain_iters=50, batch_size=2 ** 18,
#                                 eval_batch_size=2 ** 18,
#                                 cache_eval_actions=False)

# l_config_llllgg = LearningConfiguration(learner_hyperparams=learner_hyperparams,
#                                 optimizer_type='adam',
#                                 optimizer_hyperparams=optimizer_hyperparams,
#                                 input_length=2,
#                                 hidden_nodes=hidden_nodes,
#                                 hidden_activations=hidden_activations,
#                                 pretrain_iters=50, batch_size=2 ** 18,
#                                 eval_batch_size=2 ** 18,
#                                 cache_eval_actions=False)


# def test_independent_valuation_llg_model_sharing():
#     experiment_params_llg['model_sharing'] = True
#     experiment = LLGExperiment(experiment_params_llg, gpu_config, l_config_llg)
#     experiment.run(epochs = 101, n_runs=2)

# def test_independent_valuation_llg_no_model_sharing():
#     experiment_params_llg['model_sharing'] = False
#     experiment = LLGExperiment(experiment_params_llg, gpu_config, l_config_llg)
#     experiment.run(epochs = 101, n_runs=2)


# def test_llllgg_model_sharing():
#     experiment = LLLLGGExperiment(experiment_params_llllgg, gpu_config, l_config_llllgg)
#     experiment.run(epochs = 11, n_runs=2)