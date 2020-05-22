""" This test file checks whether bidders and strategies have expected behaviour"""

import time
import numpy as np

import pytest
import torch

import bnelearn.bidder as b
import bnelearn.strategy as s

device = 'cuda' if torch.cuda.is_available() else 'cpu'
u_lo = 0.
u_hi = 10.
mu = u_lo + (u_hi - u_lo)/2

def test_truthful_strategy():
    """Truthful strategy should bid truthfully"""
    strat = s.TruthfulStrategy()
    bidder = b.Bidder.uniform(0,10,strat, batch_size = 2**10)

    assert torch.equal(bidder.valuations, bidder.get_action())

def test_closure_strategy_basic():
    """Closure Strategy should return expected result"""
    closure = lambda x: x+1.

    strat = s.ClosureStrategy(closure)
    bidder = b.Bidder.uniform(0,10,strat, batch_size = 2**3)

    assert torch.equal(bidder.get_action(), bidder.valuations+1), \
        "Closure strategy returned invalid results."

def test_closure_strategy_invalid_input():
    """Invalid closures should raise exception"""
    closure = 5

    with pytest.raises(ValueError):
        _ = s.ClosureStrategy(closure)

def test_bidder_with_cached_actions():
    """Bidder with action caching should not re-evaluate until valuations have changed."""
    def closure(x):
        time.sleep(0.25)
        return x

    strat = s.ClosureStrategy(closure)
    bidder = b.Bidder.uniform(
        u_lo,u_hi,strat,
        batch_size=2**3, cache_actions=True)

    actions = bidder.get_action()

    ## Rerunning actions should not reevaluate the closure!
    tic = time.time()
    new_actions = bidder.get_action()
    toc = time.time()

    assert toc - tic < 0.25, "Call took too long, was the closure re-evaluated?"
    assert torch.equal(actions, new_actions), "Second call returned different result."

    # Rerunning with new evaluations should reevaluate the closure!

    bidder.draw_valuations_()
    tic = time.time()
    new_actions = bidder.get_action()
    toc = time.time()

    assert toc - tic > 0.25, "Call was too fast, closure can't have been reevaluated."
    assert torch.equal(bidder.valuations, new_actions), "invalid results of re-evaluation."


def test_action_caching_with_manual_valuation_change():
    """Manually changing bidder valuations should not break action caching logic."""
    def closure(x):
        return x

    strat = s.ClosureStrategy(closure)
    bidder = b.Bidder.uniform(
        u_lo, u_hi, strat, batch_size=2**3, cache_actions = True
    )
    _ = bidder.get_action()

    zeros = torch.zeros_like(bidder.valuations)
    
    bidder.valuations = zeros
    assert torch.allclose(bidder.get_action(), zeros), "Bidder returned incorrect actions!"

def test_perfectly_correlated_valuation_draw():
    """Drawing valuations perfectly correlated to some tensor should lead to that tensor"""

    strat = s.TruthfulStrategy()
    bidder = b.Bidder.uniform(u_lo,u_hi,strat, batch_size = 2**10)

    common_part = torch.ones_like(bidder.valuations)

    bidder.draw_valuations_(correlation_type='anything_except_none', correlation_strength=1.0,
                            common_component=common_part)

    assert torch.equal(bidder.valuations, common_part), "valuations are not perfectly correlated!"

def test_correlated_valuation_draw_constant_weights():
    """Tests whether the constant weights model returns valuations with the correct correlation.

    Note: this requires affected bidders and the common component to be drawn from the same distribution!
    Note 2: In the constant weights model, the correlation gamma between two bidders is not the same as the
    correlation between each bidder and the common additive component! (which is sqrt(gamma))
    """

    strat = s.TruthfulStrategy()
    bidder1 = b.Bidder.uniform(u_lo,u_hi,strat, batch_size = 2**16)
    bidder2 = b.Bidder.uniform(u_lo,u_hi,strat, batch_size = 2**16)

    correlations = torch.linspace(0, 1, 10)

    # Note: relu in valuations might break correlation, so ensure no negative common parts
    common = torch.ones_like(bidder1.valuations).uniform_(u_lo, u_hi)#uniform_(u_lo, u_hi)

    for gamma in correlations:
        bidder1.draw_valuations_('constant_weights', gamma, common)
        bidder2.draw_valuations_('constant_weights', gamma, common)

        corr = np.corrcoef(torch.stack((common.flatten(),
                                        bidder1.valuations.flatten(),
                                        bidder2.valuations.flatten())
                           ).cpu())
        
        assert abs(corr[0,1]**2 - gamma) < 0.01, \
            f'Wrong cor between common and bidder 1, got {corr[0,1]:.3f}, exp. {gamma:.3f}!'
        assert abs(corr[0,2]**2 - gamma) < 0.01, \
            f'Wrong cor between common and bidder 2, got {corr[0,2]:.3f}, exp. {gamma:.3f}!'
        assert abs(corr[1,2] - gamma) < 0.01, \
            f'Wrong cor between bidder 1 and bidder 2, got {corr[1,2]:.3f}, exp. {gamma:.3f}!'

def test_correlated_valuation_draw_Bernoulli_weights():
    """Tests whether the Bernoulli weights model returns valuations with the correct correlation.

    Note: this requires affected bidders and the common component to be drawn from the same distribution!
    """
    strat = s.TruthfulStrategy()
    bidder1 = b.Bidder.uniform(u_lo,u_hi,strat, batch_size = 2**16)
    bidder2 = b.Bidder.uniform(u_lo,u_hi,strat, batch_size = 2**16)

    correlations = torch.linspace(0, 1, 10)

    # Note: relu in valuations might break correlation, so ensure no negative common parts
    common = torch.ones_like(bidder1.valuations).uniform_(u_lo, u_hi)#uniform_(u_lo, u_hi)

    for gamma in correlations:
        bidder1.draw_valuations_('Bernoulli_weights', gamma, common)
        bidder2.draw_valuations_('Bernoulli_weights', gamma, common)

        corr = np.corrcoef(torch.stack((common.flatten(),
                                        bidder1.valuations.flatten(),
                                        bidder2.valuations.flatten())
                           ).cpu())
        
        assert abs(corr[0,1] - gamma) < 0.01, \
            f'Wrong cor between common and bidder 1, got {corr[0,1]:.3f}, exp. {gamma:.3f}!'
        assert abs(corr[0,2] - gamma) < 0.01, \
            f'Wrong cor between common and bidder 2, got {corr[0,2]:.3f}, exp. {gamma:.3f}!'
        assert abs(corr[1,2] - gamma) < 0.01, \
            f'Wrong cor between bidders 1 and 2, got {corr[1,2]:.3f}, exp. {gamma:.3f}!'



def test_parallel_closure_evaluation():
    """Parallelism of closure evaluation should work as expected."""
    pytest.skip("Test not implemented.")
