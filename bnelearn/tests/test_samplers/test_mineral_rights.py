""" This pytest test file checks whether valuation and observation samplers have the
expected behaviour"""

from math import sqrt

import pytest
import torch

import bnelearn.valuation_sampler as vs

device = 'cuda' if torch.cuda.is_available() else 'cpu'


batch_size = 2**20

# for first test: outer x 1
conditional_outer_batch_size = 2**15
# for second test: few x inner
conditional_inner_batch_size = 2**18


ids, test_cases = zip(*[
    #                      nplayers, valuation_size, u_lo,  u_hi
    ['0_base_case',       (3,        1,              0.0,   1.0     )],
    ['1_scaled'   ,       (3,        1,              0.0,   5.0     )],
    ['2_shifted',         (3,        1,              1.0,   2.0     )],
    ['3_affine',          (3,        1,              2.0,   4.0     )],
    ['4_multi_item',      (3,        3,              0.0,   1.0     )],
    ['5_multi_item_affine',(5,       3,              2.0,   4.0     )],
])



@pytest.mark.parametrize("n_players, valuation_size, u_lo, u_hi",
                         test_cases, ids=ids)
def test_mineral_rights(n_players, valuation_size, u_lo, u_hi):
    """Test the Mineral Rights Sampler."""


    ### test with valuation size 1.
    s = vs.MineralRightsValuationObservationSampler(n_players, valuation_size, u_lo, u_hi, batch_size)


    v,o = s.draw_profiles()
    assert o.device == v.device, "Observations and Valuations should be on same device"
    assert o.device.type == device, "Standard device should be cuda, if available!"

    assert o.shape == v.shape

    # Test the value distribution
    for i in range(n_players):
        assert torch.equal(v[:,0,:], v[:,i,:]), "all players should have the same valuation!"
    
    # v are uniform distributed
    v_mean = u_lo + 0.5*(u_hi - u_lo)
    v_std = (u_hi - u_lo) * sqrt(1/12)

    # observations are 2*v*x where x is U[0,1] distributed and v,x are independent.
    # thus the mean and variance of o are given by 
    # E[o] = 2*E[v]*E[X] = 2*E[V]*0.5
    # Var(o) = var(2vx) = 4var(vx) = 4*[ E[V]²Var(X) + E[X]² Var(v) + Var(V)Var(X)]
    # = 4*[u_mean² * 1/12 + 0.25*u_std² + u_std²*1/12]
    o_std = sqrt(v_mean**2 / 3 + v_std**2 * 4/3)

    expected_valuation_mean = torch.tensor([[v_mean]*valuation_size]*n_players, device=v.device)
    expected_valuation_std = torch.tensor([[v_std]*valuation_size]*n_players, device=v.device)
    expected_observation_std =  torch.tensor([[o_std]*valuation_size]*n_players, device=o.device)

    assert torch.allclose(v.mean(dim=0), expected_valuation_mean, rtol = 0.02), \
        "unexpected valuation sample mean!"
    # mean of obs should be identical to mean of vals.
    assert torch.allclose(o.mean(dim=0), expected_valuation_mean, rtol = 0.02), \
        "unexpected observation sample mean!"

    assert torch.allclose(v.std(dim=0), expected_valuation_std, rtol = 0.02), \
        "unexpected sample mean!"

    assert torch.allclose(o.std(dim=0), expected_observation_std, rtol = 0.02), \
        "unexpected sample mean!"

    ## test manual dimensionalities and devices
    # draw explicitly on cpu
    v, o = s.draw_profiles(device = 'cpu')
    assert v.device.type == 'cpu' and o.device.type == 'cpu'

    
    # draw alternative batch size
    v,o = s.draw_profiles(batch_size = conditional_outer_batch_size)
    assert v.shape[0] == conditional_outer_batch_size and  \
        o.shape[0] == conditional_outer_batch_size, \
        "explicit batch_size was not respected!"


    # reuse these sample for some do some conditional sampling

    # Conditional test 1: outer according to true distribtuion, inner_batch_size = 1

    for i in range(n_players):
        cv, co = s.draw_conditional_profiles(i, o[:,i,:], 1)
        assert torch.allclose(o[:,i,:], co[:,i,:]), \
            "conditional sample did not respect inputs!"

        assert  torch.allclose(o.mean(dim=0), co.mean(dim=0), rtol = 0.02) \
            and torch.allclose(o.std(dim=0),  co.std(dim=0),  rtol = 0.02) \
            and torch.allclose(v.mean(dim=0), cv.mean(dim=0), rtol = 0.02) \
            and torch.allclose(v.std(dim=0),  cv.std(dim=0),  rtol = 0.02), \
            "With outer batch following the true distribtuion and inner_batch_size=1," + \
                "co, cv should follow the same distribution as o,v."

    # Conditional test 2: specific outer cases, large inner batch size
    i=0
    conditioned_observation = \
        torch.tensor([[0], # lowest possible obs
                      [2*u_hi], # highest possible obs 
                      [u_hi], # highest v (but not highest obs)
                      [u_lo], # lowest v (but not lowest obs)
                      [(u_lo + u_hi)/2]]) \
            .repeat(1, valuation_size)
    outer_batch_size = conditioned_observation.shape[0]
    
    cv, co = s.draw_conditional_profiles(i, conditioned_observation, conditional_inner_batch_size)

    co =co.view(outer_batch_size, conditional_inner_batch_size, n_players, valuation_size)
    # TODO: continue here! tomorrow edge cases fail.

    






