""" This test file checks whether bidders and strategies have expected behaviour"""

import time

import pytest
import torch

import bnelearn.bidder as b
import bnelearn.strategy as s

device = 'cuda' if torch.cuda.is_available() else 'cpu'
u_lo = 0.
u_hi = 10.
mu = u_lo + (u_hi - u_lo)/2

valuation_size = observation_size = action_size = 1
batch_size = 2**3

valuations= torch.tensor(
    range(batch_size),
    dtype=torch.float, device=device
    ).view(-1, valuation_size) / (batch_size - 1)
observations = valuations

def test_truthful_strategy():
    """Truthful strategy should bid truthfully"""
    strat = s.TruthfulStrategy()
    bidder = b.Bidder(strat, None, batch_size, valuation_size, observation_size, action_size)

    assert torch.equal(observations, bidder.get_action(observations))

def test_closure_strategy_basic():
    """Closure Strategy should return expected result"""
    closure = lambda x: x+1.

    strat = s.ClosureStrategy(closure)
    bidder = b.Bidder(strat, None, batch_size, valuation_size, observation_size, action_size)

    assert torch.equal(observations+1, bidder.get_action(observations)), \
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
    bidder = b.Bidder(
        strat, None, batch_size,
        valuation_size, observation_size, action_size,
        enable_action_caching=True)

    actions = bidder.get_action(observations)

    ## Rerunning actions should not reevaluate the closure!
    tic = time.time()
    new_actions = bidder.get_action(observations)
    toc = time.time()

    assert toc - tic < 0.25, "Call took too long, was the closure re-evaluated?"
    assert torch.equal(actions, new_actions), "Second call returned different result."

    # Calling get_action without passing observations should return the cached actions
    tic = time.time()
    new_actions = bidder.get_action(observations=None)
    toc = time.time()

    assert toc - tic < 0.25, "Call took too long, was the closure re-evaluated?"
    assert torch.equal(actions, new_actions), "Second call returned different result."

    # Rerunning with new evaluations should reevaluate the closure!

    new_observations = observations * torch.empty_like(observations).uniform_()
    tic = time.time()
    new_actions = bidder.get_action(new_observations)
    toc = time.time()

    assert toc - tic > 0.25, "Call was too fast, closure can't have been reevaluated."
    assert torch.equal(new_observations, new_actions), "invalid results of re-evaluation."


def test_action_caching_with_manual_observation_change():
    """Manually changing bidder observations should not break action caching logic."""
    def closure(x):
        return x

    strat = s.ClosureStrategy(closure)
    bidder = b.Bidder(
        strat, None, batch_size,
        valuation_size, observation_size, action_size,
        enable_action_caching=True)

    _ = bidder.get_action(observations)

    zeros = torch.zeros_like(bidder.cached_observations)
    
    bidder.cached_observations = zeros
    assert torch.allclose(bidder.get_action(observations=None), zeros), "Bidder returned incorrect actions!"

def test_parallel_closure_evaluation():
    """Parallelism of closure evaluation should work as expected."""
    pytest.skip("Test not implemented.")
