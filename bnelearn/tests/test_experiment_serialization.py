from bnelearn.experiment.configuration_manager import ConfigurationManager
import pytest


def test_all_experiments_serialization():
    """
    Checks all types of experiments are serialized and deserialized properly with default parameters
    """

    for experiment_type in list(ConfigurationManager.experiment_types.keys()):
        exp_config, experiment_class = ConfigurationManager(experiment_type=experiment_type, n_runs=2, n_epochs=3) \
            .set_logging(eval_batch_size=2**2).get_config()  # otherwise long runtime for some settings!
        experiment_class(exp_config)  # There are configs set on init
        assert ConfigurationManager.experiment_config_could_be_saved_properly(exp_config)
