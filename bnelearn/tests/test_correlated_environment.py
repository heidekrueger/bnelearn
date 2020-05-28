import numpy as np

import torch

import bnelearn.bidder as b
import bnelearn.strategy as s
import bnelearn.correlation_device as cd
from bnelearn.mechanism import FirstPriceSealedBidAuction
from bnelearn.environment import AuctionEnvironment

device = 'cuda' if torch.cuda.is_available() else 'cpu'
u_lo = 0.
u_hi = 10.

def test_independent_correlation_device_draw():
    strat = s.TruthfulStrategy()
    bidder1 = b.Bidder.uniform(u_lo,u_hi,strat, batch_size = 2**18)
    bidder2 = b.Bidder.uniform(u_lo,u_hi,strat, batch_size = 2**18)

    corrdev = cd.IndependentValuationDevice()
    common_part, weights = corrdev.get_component_and_weights()

    bidder1.draw_valuations_(common_part, weights)
    bidder2.draw_valuations_(common_part, weights)

    corr = np.corrcoef(torch.stack((bidder1.valuations.flatten(),
                                    bidder2.valuations.flatten())).cpu())
        
    assert abs(corr[0,1] ) < 0.01, \
            f'Bidders are not independent! Found correlation {corr[0,1]:.3f}, exp. {0.000:.3f}!'


def test_perfectly_correlated_valuation_draw():
    """Drawing valuations perfectly correlated to some tensor should lead to that tensor"""

    strat = s.TruthfulStrategy()
    batch_size = 2**10
    dist = torch.distributions.Uniform(u_lo, u_hi)

    bidder1 = b.Bidder(dist, strat, batch_size = batch_size)
    bidder2 = b.Bidder(dist, strat, batch_size = batch_size)

    corrdev = cd.BernoulliWeightsCorrelationDevice(dist, batch_size, 1, 1.0)
    common_part, weights = corrdev.get_component_and_weights()

    bidder1.draw_valuations_(common_part, weights)
    bidder2.draw_valuations_(common_part, weights)

    corr = np.corrcoef(torch.stack((bidder1.valuations.flatten(),
                                    bidder2.valuations.flatten())).cpu())
        
    assert abs(corr[0,1] - 1.0) < 1e-4, f"valuations are not perfectly correlated! got {corr[0,1]:.3f}"

    corrdev = cd.ConstantWeightsCorrelationDevice(dist, batch_size, 1, 1.0)
    common_part, weights = corrdev.get_component_and_weights()

    bidder1.draw_valuations_(common_part, weights)
    bidder2.draw_valuations_(common_part, weights)

    corr = np.corrcoef(torch.stack((bidder1.valuations.flatten(),
                                    bidder2.valuations.flatten())
                           ).cpu())
        
    assert abs(corr[0,1] - 1.0) < 1e-4, f"valuations are not perfectly correlated! got {corr[0,1]:.3f}"

def test_correlated_valuation_draw_constant_weights():
    """Tests whether the constant weights model returns valuations with the correct correlation.

    Note: this requires affected bidders and the common component to be drawn from the same distribution!
    Note 2: In the constant weights model, the correlation gamma between two bidders is not the same as the
    correlation between each bidder and the common additive component! (which is sqrt(gamma))
    """

    strat = s.TruthfulStrategy()
    batch_size = 2**16
    dist = torch.distributions.Uniform(u_lo, u_hi)

    bidder1 = b.Bidder(dist, strat, batch_size=batch_size)
    bidder2 = b.Bidder(dist, strat, batch_size=batch_size)

    correlations = torch.linspace(0, 1, 10)

    for gamma in correlations:
        corrdev = cd.ConstantWeightsCorrelationDevice(dist, batch_size, 1, gamma)
        common_part, weights = corrdev.get_component_and_weights()

        bidder1.draw_valuations_(common_part, weights)
        bidder2.draw_valuations_(common_part, weights)

        corr = np.corrcoef(torch.stack((common_part.flatten().cpu(),
                                        bidder1.valuations.flatten().cpu(),
                                        bidder2.valuations.flatten().cpu())
                           ))
        
        assert abs(corr[0,1]**2 - gamma) < 0.01, \
            f'Wrong cor between common and bidder 1, got {corr[0,1]:.3f}, exp. {gamma.sqrt():.3f}!'
        assert abs(corr[0,2]**2 - gamma) < 0.01, \
            f'Wrong cor between common and bidder 2, got {corr[0,2]:.3f}, exp. {gamma.sqrt():.3f}!'
        assert abs(corr[1,2] - gamma) < 0.01, \
            f'Wrong cor between bidder 1 and bidder 2, got {corr[1,2]:.3f}, exp. {gamma:.3f}!'

def test_correlated_valuation_draw_Bernoulli_weights():
    """Tests whether the Bernoulli weights model returns valuations with the correct correlation.

    Note: this requires affected bidders and the common component to be drawn from the same distribution!
    """
    strat = s.TruthfulStrategy()
    batch_size = 2**16
    dist = torch.distributions.Uniform(u_lo, u_hi)

    bidder1 = b.Bidder(dist, strat, batch_size=batch_size)
    bidder2 = b.Bidder(dist, strat, batch_size=batch_size)

    correlations = torch.linspace(0, 1, 10)

    for gamma in correlations:
        corrdev = cd.BernoulliWeightsCorrelationDevice(dist, batch_size, 1, gamma)
        common_part, weights = corrdev.get_component_and_weights()

        bidder1.draw_valuations_(common_part, weights)
        bidder2.draw_valuations_(common_part, weights)

        corr = np.corrcoef(torch.stack((common_part.flatten().cpu(),
                                        bidder1.valuations.flatten().cpu(),
                                        bidder2.valuations.flatten().cpu())
                           ))
        
        assert abs(corr[0,1] - gamma) < 0.01, \
            f'Wrong cor between common and bidder 1, got {corr[0,1]:.3f}, exp. {gamma:.3f}!'
        assert abs(corr[0,2] - gamma) < 0.01, \
            f'Wrong cor between common and bidder 2, got {corr[0,2]:.3f}, exp. {gamma:.3f}!'
        assert abs(corr[1,2] - gamma) < 0.01, \
            f'Wrong cor between bidders 1 and 2, got {corr[1,2]:.3f}, exp. {gamma:.3f}!'

def test_correlated_drawing_in_environment():
    """
    Tests drawing of valuations in 6p environment where first two and second
    two players are correlated, each with their own correlation device.
    """

    strat = s.TruthfulStrategy

    def s2p(s):
        # should not be called in test
        raise NotImplementedError()

    mechanism = FirstPriceSealedBidAuction()

    dist = torch.distributions.Uniform(u_lo, u_hi)
    batch = 2**18
    items = 1
    n = 6
    
    bidders = [b.Bidder(dist, strat, batch_size=batch) for _ in range(n)]
    gamma1 = 0.5
    gamma2 = 0.75

    correlation_groups = [[0,1], [2,3], [4,5]]
    correlation_devices = [cd.BernoulliWeightsCorrelationDevice(dist, batch, items, gamma1),
                           cd.ConstantWeightsCorrelationDevice(dist, batch, items, gamma2),
                           cd.IndependentValuationDevice()]
    env = AuctionEnvironment(mechanism, bidders, batch, n, s2p,
                             correlation_groups, correlation_devices)

    env.draw_valuations_()

    corr = np.corrcoef(torch.stack(tuple(
        [b.valuations.flatten() for b in bidders])).cpu())

    expected_corr = np.array([
        [1.,gamma1, 0, 0,  0, 0],
        [gamma1,1,  0, 0,  0, 0],
        [0, 0,  1,gamma2,  0, 0],
        [0, 0,  gamma2,1,  0, 0],
        [0, 0,    0, 0,    1, 0],
        [0, 0,    0, 0,    0, 1],
    ])

    assert np.allclose(corr, expected_corr, atol = 0.01)
