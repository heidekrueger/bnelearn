"""This module tests the calculation of market outcomes."""
import warnings
import pytest
import torch
# from bnelearn.experiment.configuration_manager import ConfigurationManager
# from bnelearn.environment import AuctionEnvironment
from bnelearn.mechanism.auctions_single_item import FirstPriceSealedBidAuction
from bnelearn.mechanism.auctions_combinatorial import LLGAuction

cuda = torch.cuda.is_available()
device = 'cuda' if cuda else 'cpu'
specific_gpu = None
if cuda and specific_gpu:
    torch.cuda.set_device(specific_gpu)

class DummyEnv():
    """Dummy environment for tests."""
    def __init__(self, batch_size=2**23, n_players=2, n_actions=1):
        self.batch_size = batch_size
        self.n_players = n_players
        self.n_actions = n_actions
        class DummyAgent():
            def __init__(self, player_position):
                self.player_position = player_position
                self.bid_size = n_actions
                self.valuations = torch.rand(
                    (batch_size, n_actions), device=device)
            def get_welfare(self, a, v):
                return (v * a).sum(dim=-1)
        self.agents = [DummyAgent(i) for i in range(n_players)]
        self._observations = torch.cat(
            [a.valuations for a in self.agents], axis=1)[:, :, None]
    def _generate_agent_actions(self):
        for i in range(self.n_players):
            yield (i, torch.rand((self.batch_size, self.n_actions), device=device))

def test_revenue():
    """Test the calculation of revenue for general auction markets"""
    mechanism = FirstPriceSealedBidAuction()
    env = DummyEnv()
    revenue = mechanism.get_revenue(env)
    # Average revenue should be the max of two uniform RVs = 2/3
    assert abs(revenue.item() - (2./3.)) < 1e-3

def test_efficiency():
    """Test the calculation of efficiency for general auction markets"""
    mechanism = FirstPriceSealedBidAuction()
    env = DummyEnv()
    efficiency = mechanism.get_efficiency(env)
    # Average efficiency should be the ratio of a 50-50 chance (actual_welfare)
    # and the max of two uniform RVs = 2/3 (maximum_welfare)
    assert abs(efficiency.item() - (0.5 / (2./3.))) < 1.5e-2

def test_llg_efficiency():
    """Test the calculation of efficiency for the specific LLG market"""
    mechanism = LLGAuction()
    env = DummyEnv(n_players=3)
    env.agents[-1].valuations *= 2
    mean_max_valuation = 1.28239631  # for precise calc. see Irwin–Hall distribution
    efficiency = mechanism.get_efficiency(env)
    # Average efficiency should be the actual_welfare and the maximum_welfare
    assert abs(efficiency.item() - (1. / mean_max_valuation)) < 2e-1
