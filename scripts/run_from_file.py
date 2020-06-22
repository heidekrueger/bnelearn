import torch

from bnelearn.experiment.configuration_manager import ConfigurationManager
from bnelearn.util import logging

experiment_config = logging.get_experiment_config_from_configurations_log()
experiment_class = ConfigurationManager.get_class_by_experiment_type(experiment_config.experiment_class)

try:
    experiment_class(experiment_config).run()
except KeyboardInterrupt:
    print('\nKeyboardInterrupt: released memory after interruption')
    torch.cuda.empty_cache()
