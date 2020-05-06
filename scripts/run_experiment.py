# TODO: essentially move this file into /scripts, this will require moving the run.... 
# functions into bnelearn.experiment.presets or similar. Assigned @Paul
import sys
import os
import torch
import fire

sys.path.append(os.path.realpath('.'))
sys.path.append(os.path.join(os.path.expanduser('~'), 'bnelearn'))

from bnelearn.experiment.gpu_controller import GPUController
from bnelearn.experiment.configurations import RunningConfiguration, ExperimentConfiguration, \
                                               LoggingConfiguration, LearningConfiguration
from bnelearn.experiment.single_item_experiment import UniformSymmetricPriorSingleItemExperiment, \
                                                       GaussianSymmetricPriorSingleItemExperiment, \
                                                       TwoPlayerAsymmetricUniformPriorSingleItemExperiment
from bnelearn.experiment.combinatorial_experiment import LLGExperiment, LLLLGGExperiment
from bnelearn.experiment.multi_unit_experiment import MultiUnitExperiment, SplitAwardExperiment
import warnings
from bnelearn.experiment import presets

from dataclasses import dataclass, field, asdict



if __name__ == '__main__':
    '''
    Runs predefined experiments with individual parameters
    fire.Fire() asks you to decide for one of the experiments defined above
    by writing its name and define the required (and optional) parameters
    e.g.:
        presets.experiment.py presets.single_item_uniform_symmetric 1 20 [2,3] 'first_price'

    alternatively instead of fire.Fire() use, e.g.:
        presets.single_item_uniform_symmetric(1,20,[2,3],'first_price')

    '''
    # n_runs, n_epochs, n_players, specific_gpu, input_length, experiment_class, experiment_params = fire.Fire()
    # n_runs, n_epochs, n_players, specific_gpu, input_length, experiment_class, experiment_params = presets.llg(1,20,'vcg')
    # n_runs, n_epochs, n_players, specific_gpu, input_length, experiment_class, experiment_params = \
    #       presets.single_item_uniform_symmetric(1,20, 2, 'first_price')

    # running_configuration, logging_configuration, experiment_configuration, experiment_class = \
    #     presets.single_item_uniform_symmetric(2, 100, [2], 'first_price', model_sharing=True)
    # logging_configuration.save_tb_events_to_binary_detailed = True
    # logging_configuration.save_tb_events_to_csv_detailed = True

    # running_configuration, logging_configuration, experiment_configuration, experiment_class = \
    #     presets.single_item_gaussian_symmetric(1,20, [2], 'second_price')
    running_configuration, logging_configuration, experiment_configuration, experiment_class =\
       presets.llg(1,110,'nearest_zero',specific_gpu=1)
    # running_configuration, logging_configuration, experiment_configuration, experiment_class = \
    #    presets.llllgg(1,310,'first_price')#,model_sharing=False)
    # running_configuration, logging_configuration, experiment_configuration, experiment_class = \
    #   presets.multiunit(n_runs=100, n_epochs=4000, n_players=[2], n_units=2, payment_rule='first_price')
    # running_configuration, logging_configuration, experiment_configuration, experiment_class = \
    #   presets.splitaward(1, 500, [2])
    # running_configuration, logging_configuration, experiment_configuration, experiment_class = \
    #    presets.single_item_asymmetric_uniform(n_runs=1, n_epochs=4000)

    gpu_configuration = GPUController(specific_gpu=running_configuration.specific_gpu)
    input_length = experiment_configuration.n_units \
        if experiment_configuration.n_units is not None else 1
    learning_configuration = LearningConfiguration(
        input_length=1,
        pretrain_iters=500
    )

    try:
        for i in running_configuration.n_players:
            experiment_configuration.n_players = i
            experiment = experiment_class(experiment_configuration, learning_configuration,
                                          logging_configuration, gpu_configuration)
            experiment.run(epochs=running_configuration.n_epochs, n_runs=running_configuration.n_runs)

    except KeyboardInterrupt:
        print('\nKeyboardInterrupt: released memory after interruption')
        torch.cuda.empty_cache()
