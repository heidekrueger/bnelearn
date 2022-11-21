""" This pytest test file checks whether valuation and observation samplers have the
expected behaviour"""

import pytest

import torch
import numpy as np

import bnelearn.sampler as vs

device = 'cuda' if torch.cuda.is_available() else 'cpu'

batch_size = 2**20

# for first test: outer x 1
conditional_outer_batch_size = 2**15
# for second test: few x inner
conditional_inner_batch_size = 2**18

local_min = global_min = 0.
local_mean = 0.5
global_mean = 1.
local_max = 1.
global_max = 2.0

ids, test_cases = zip(*[
    #                                          setting,         method,     gamma
    ['0_LLG_independent',                     ('LLG',           None,       0.0)],
    ['1_LLG_Bernoulli_0.5'  ,                 ('LLG',           'Bernoulli',0.5)],
    ['2_LLG_Bernoulli_perfect',               ('LLG',           'Bernoulli',1.0)],
    ['3_LLG_constant_0.5',                    ('LLG',           'constant', 0.5)],
    ['4_LLG_onstant_perfect',                 ('LLG',           'constant', 1.0)],
    ['5_LLG_independent_but_method_given',    ('LLG',           'constant', 0.0)],
    ['6_LLLLGG_independent',                  ('LLLLGG',        None,       0.0)],
    ['7_LLLLGG_Bernoulli_0.5'  ,              ('LLLLGG',        'Bernoulli',0.5)],
    ['8_LLLLGG_Bernoulli_perfect',            ('LLLLGG',        'Bernoulli',1.0)],
    ['9_LLLLGG_constant_0.5',                 ('LLLLGG',        'constant', 0.5)],
    ['10_LLLLGG_onstant_perfect',             ('LLLLGG',        'constant', 1.0)],
    ['11_LLLLGG_independent_but_method_given',('LLLLGG',        'constant', 0.0)],
])

@pytest.mark.parametrize("setting, method, gamma", test_cases, ids=ids)
def test_local_global_samplers(setting, method, gamma):
    """Test whether the LLG sampler works as expected."""

    if setting == 'LLG':
        s = vs.LLGSampler(correlation=gamma, correlation_method=method,
                          default_batch_size=batch_size, default_device=device)
    elif setting == 'LLLLGG':
        s = vs.LLLLGGSampler(correlation_locals=gamma, correlation_method_locals=method,
                             default_batch_size=batch_size, default_device=device)
    else:
        raise ValueError("invalid method")

    local_indices = [0,1] if setting == 'LLG' else [0,1,2,3]
    global_indices = [2] if setting == 'LLG' else [4,5]

    n_players = 3 if setting=='LLG' else 6
    valuation_size = 1 if setting == 'LLG' else 2


    v,o = s.draw_profiles()
    assert o.device == v.device, "Observations and Valuations should be on same device"
    assert o.device.type == device, "Standard device should be cuda, if available!"

    assert torch.equal(o, v), "observations and valuations should be identical in IPV"


    assert torch.allclose(v[:, local_indices, :].mean(dim=0) - local_mean,
                          torch.zeros([2],device=device),
                          atol= 2e-3), \
        "unexpected means for locals"

    assert torch.allclose(v[:, global_indices, :].mean(dim=0) - global_mean,
                          torch.zeros([1],device=device),
                          atol= 2e-3), \
        "unexpected means for global player"

    ## quick checks for conditional sampling, let's fix the valuation of the
    # second local player
    # this is more about testing for runtime errors, rather than correctness
    # of the conditional sampling, which has been tested in the sub-group samplers
    conditioned_player = 1
    conditioned_observation = torch.tensor([[0.5]*valuation_size], device=device)

    cv,co = s.draw_conditional_profiles(conditioned_player, conditioned_observation, batch_size)

    assert torch.equal(co, cv)

    assert cv.shape == torch.Size([1, batch_size, n_players, valuation_size]), 'invalid shape!'

    assert torch.allclose(v[:, local_indices, :].mean(dim=0) - local_mean,
                          torch.zeros([2],device=device),
                          atol= 2e-3), \
        "unexpected means for locals"

    assert torch.allclose(cv[..., conditioned_player, :] - 0.5, torch.zeros([batch_size, valuation_size], device=device))

    assert torch.allclose(v[:, global_indices, :].mean(dim=0) - global_mean,
                          torch.zeros([1],device=device),
                          atol= 2e-3), \
        "unexpected means for global player"


    ## test grids
    ## choose n_points such that we get 2 points (min/max) along each dimension,
    ## then check that they match for a local and a global player:

    ASSERTION_ERROR_UNEXPECTED_GRID = "unexpected grid for local bidder."

    n_points = 2**valuation_size
    grid_local = s.generate_valuation_grid(player_position=local_indices[0],
                                           minimum_number_of_points=n_points)

    assert torch.equal(grid_local.min(dim=0).values,
                       torch.tensor([local_min]*valuation_size, device=device)), \
                           ASSERTION_ERROR_UNEXPECTED_GRID
    assert torch.equal(grid_local.max(dim=0).values,
                       torch.tensor([local_max]*valuation_size, device=device)), \
                           ASSERTION_ERROR_UNEXPECTED_GRID

    grid_global = s.generate_valuation_grid(player_position=global_indices[0],
                                            minimum_number_of_points=n_points)

    assert torch.equal(grid_global.min(dim=0).values,
                       torch.tensor([global_min]*valuation_size, device=device)), \
                           ASSERTION_ERROR_UNEXPECTED_GRID
    assert torch.equal(grid_global.max(dim=0).values,
                       torch.tensor([global_max]*valuation_size, device=device)), \
                           ASSERTION_ERROR_UNEXPECTED_GRID
