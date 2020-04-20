# ToDo
# The idea is to create a test which would run all the types of experiments in with very minimalistic settings to check
# that nothing is broken. I am not sure which assertions to make, so strictly speaking this is not yet a test but
# simply a script which should run without runtime errors. Also it should be logging to a separate folder.
# Also parameters are rather meaningless as I don't understand the boundaries for values for each specific experiment
import sys
import os
sys.path.append(os.path.realpath('.'))
import torch.nn as nn
from bnelearn.experiment.run_experiment import *
from bnelearn.experiment.configurations import *
from bnelearn.experiment.gpu_controller import *

#TODO: Make nice testing!
def test_auction(create_auction_function, input_length):
    learning_configuration = LearningConfiguration(input_length=input_length)
    running_configuration, logging_configuration, experiment_configuration, experiment_class = create_auction_function
    experiment_configuration.n_players = running_configuration.n_players[0]
    experiment = experiment_class(experiment_configuration, learning_configuration,
                                    logging_configuration, gpu_configuration)
    experiment.run(epochs=running_configuration.n_epochs, n_runs=running_configuration.n_runs)

gpu_configuration = GPUController(specific_gpu=1)
test_auction(run_single_item_uniform_symmetric(2,110, [2], 'first_price'),1)
test_auction(run_single_item_gaussian_symmetric(1,110, [2], 'second_price', log_metrics = []),1)
test_auction(run_llg(1,110,'nearest_vcg'),1)
test_auction(run_llllgg(1,110,'first_price'),2)
test_auction(run_multiunit(1, 110, [2], 'vickrey'), input_length=2)
# Not working yet
test_auction(run_splitaward(1, 110, [2]),2)

