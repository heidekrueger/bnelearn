import os

from bnelearn.experiment import Experiment
from bnelearn.experiment.configuration_manager import ConfigurationManager
from bnelearn.util import logging


# ToDO Implement the comparison
def compare_two_experiments(exp1: Experiment, exp2: Experiment) -> bool:
    return True


def test_experiments_equality():
    # Not sure about the proper path to use
    log_root_dir = os.path.join(os.path.expanduser('~'), 'bnelearn', 'experiments')
    experiment_config, experiment_class = ConfigurationManager(experiment_type='single_item_uniform_symmetric') \
        .get_config(save_tb_events_to_csv_detailed=True, log_root_dir=log_root_dir)
    # some parts of the config are changed in the children of the experiment class, so it's necessary to instantiate it
    experiment = experiment_class(experiment_config)

    logging.log_experiment_configurations(experiment_log_dir=log_root_dir, experiment_configuration=experiment_config)
    retrieved_experiment_config = logging.get_experiment_config_from_configurations_log(experiment_log_dir=log_root_dir)
    retrieved_experiment_class = ConfigurationManager.get_class_by_experiment_type(experiment_config.experiment_class)

    retrieved_experiment = retrieved_experiment_class(retrieved_experiment_config)
    equality = compare_two_experiments(retrieved_experiment, experiment)
    path = os.path.join(log_root_dir, logging._configurations_f_name)
    if os.path.exists(path):
        os.remove(path)

    assert equality

# experiment_config, experiment_class = ConfigurationManager(experiment_type='single_item_gaussian_symmetric') \
#     .get_config(log_root_dir=log_root_dir)

# All three next experiments get AssertionError: scalar should be 0D
# experiment_config, experiment_class = \
#     ConfigurationManager(experiment_type='single_item_asymmetric_uniform_overlapping') \
#     .get_config(log_root_dir=log_root_dir)
# experiment_config, experiment_class = \
#     ConfigurationManager(experiment_type='single_item_asymmetric_uniform_disjunct') \
#     .get_config(log_root_dir=log_root_dir)
# experiment_config, experiment_class = ConfigurationManager(experiment_type='llg') \
#     .get_config(log_root_dir=log_root_dir)

# experiment_config, experiment_class = ConfigurationManager(experiment_type='llllgg') \
#     .get_config(log_root_dir=log_root_dir)
