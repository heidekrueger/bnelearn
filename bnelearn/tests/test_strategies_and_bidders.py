""" This test file checks whether bidders and strategies have expected behaviour"""

import time

import pytest
import torch

import bnelearn.bidder
import bnelearn.strategy as s

device = 'cuda' if torch.cuda.is_available() else 'cpu'
u_lo = 0.
u_hi = 10.
mu = u_lo + (u_hi - u_lo)/2

def test_truthful_strategy():
    """Truthful strategy should bid truthfully"""
    strat = s.TruthfulStrategy()
    bidder = bnelearn.bidder.Bidder.uniform(0,10,strat, batch_size = 2**10)

    assert torch.equal(bidder.valuations, bidder.get_action())

def test_closure_strategy_basic():
    """Closure Strategy should return expected result"""
    closure = lambda x: x+1.

    strat = s.ClosureStrategy(closure)
    bidder = bnelearn.bidder.Bidder.uniform(0,10,strat, batch_size = 2**3)

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
        time.sleep(0.5)
        return x

    strat = s.ClosureStrategy(closure)
    bidder = bnelearn.bidder.Bidder.uniform(
        0,10,strat,
        batch_size=2**3, cache_actions=True)

    actions = bidder.get_action()

    ## Rerunning actions should not reevaluate the closure!
    tic = time.time()
    new_actions = bidder.get_action()
    toc = time.time()

    assert toc - tic < 0.5, "Call took too long, was the closure re-evaluated?"
    assert torch.equal(actions, new_actions), "Second call returned different result."

    # Rerunning with new evaluations should reevaluate the closure!

    bidder.draw_valuations_()
    tic = time.time()
    new_actions = bidder.get_action()
    toc = time.time()

    assert toc - tic > 0.5, "Call was too fast, closure can't have been reevaluated."
    assert torch.equal(bidder.valuations, new_actions), "invalid results of re-evaluation."

def test_parallel_closure_evaluation():
    """Parallelism of closure evaluation should work as expected."""
    pytest.skip("Test not implemented.")
