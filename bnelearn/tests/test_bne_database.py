"""Testing correctness of the BNE utilities database."""

from bnelearn.experiment.configuration_manager import ConfigurationManager


def test_bne_utility_database():
    """Testing correctness of the BNE utilities database."""
    specific_gpu = 0

    experiment_config, experiment_class = ConfigurationManager(experiment_type='llg', n_runs=1, n_epochs=1)\
        .set_hardware(specific_gpu=specific_gpu)\
        .set_learning(pretrain_iters=0, batch_size=2).set_logging(enable_logging=False) \
        .get_config()

    _ = experiment_class(experiment_config)
