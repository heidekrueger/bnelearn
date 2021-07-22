"""This module tests the calculation of market outcomes."""
import warnings
import pytest
import torch
from bnelearn.experiment.configuration_manager import ConfigurationManager
# from bnelearn.environment import AuctionEnvironment
from bnelearn.mechanism.auctions_single_item import FirstPriceSealedBidAuction
from bnelearn.mechanism.auctions_combinatorial import LLGAuction

cuda = torch.cuda.is_available()
device = 'cuda' if cuda else 'cpu'
specific_gpu = None
if cuda and specific_gpu:
    torch.cuda.set_device(specific_gpu)


def test_revenue():
    """Test the calculation of revenue for general auction markets"""
    experiment_config, experiment_class = ConfigurationManager(
        experiment_type='single_item_uniform_symmetric',
        n_runs=0, n_epochs=0
    ).get_config()
    experiment = experiment_class(experiment_config)
    revenue = experiment.bne_env.get_revenue()
    # Average revenue should be 1/3 in this BNE
    assert abs(revenue.item() - 1/3.) < 1e-3


# Format: id, experiment_type, setting_params, true_efficiency
test_ids, *test_data = zip(*[
    [
        '1 - single-item',
        'single_item_uniform_symmetric',
        {},
        1
    ],
    [
        '2 - LLG',
        'llg',
        {},
        0.96561  # See Ausubel & Baranov 2019, Fig. 5
    ]
    # TODO [optional]: add multi-unit, etc. tests
])

def run_efficiency_test(experiment_type = str, setting_params = dict, true_efficiency = float):
    """Test the calculation of efficiency for general auction markets"""
    experiment_config, experiment_class = ConfigurationManager(
        experiment_type=experiment_type,
        n_runs=0, n_epochs=0
    ) \
        .set_setting(**setting_params) \
        .get_config()
    experiment = experiment_class(experiment_config)

    efficiency = experiment.bne_env.get_efficiency()
    assert abs(efficiency.item() - true_efficiency) < 1e-3

@pytest.mark.parametrize("experiment_type, setting_params, true_efficiency", zip(*test_data), ids=test_ids)
def test_efficiency(experiment_type, setting_params, true_efficiency):
    run_efficiency_test(experiment_type, setting_params, true_efficiency)
