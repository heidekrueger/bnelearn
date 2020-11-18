import torch

from bnelearn.experiment.configuration_manager import ConfigurationManager

experiment_config = ConfigurationManager.load_experiment_config()
experiment_class = ConfigurationManager.get_class_by_experiment_type(experiment_config.experiment_class)

try:
    experiment_class(experiment_config).run()
except KeyboardInterrupt:
    print('\nKeyboardInterrupt: released memory after interruption')
    torch.cuda.empty_cache()
