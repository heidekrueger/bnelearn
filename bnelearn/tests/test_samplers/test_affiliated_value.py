""" This pytest test file checks whether valuation and observation samplers have the
expected behaviour"""

from math import sqrt

import pytest
import torch

import bnelearn.sampler as vs

device = 'cuda' if torch.cuda.is_available() else 'cpu'


batch_size = 2**20

# for first test: outer x 1
conditional_outer_batch_size = 2**15
# for second test: few x inner
conditional_inner_batch_size = 2**18


ids, test_cases = zip(*[
    #                      nplayers, valuation_size, u_lo,  u_hi
    ['0_base_case',       (2,        1,              0.0,   1.0     )],
    ['1_scaled'   ,       (2,        1,              0.0,   5.0     )],
    ['2_shifted',         (2,        1,              1.0,   2.0     )],
    ['3_affine',          (2,        1,              2.0,   4.0     )],
    ['4_multi_item',      (2,        3,              0.0,   1.0     )],
    ['5_multi_player',    (3,        1,              0.0,   1.0     )],
    ['6_wild_mix',        (3,        4,              2.0,   4.0     )],
])



@pytest.mark.parametrize("n_players, valuation_size, u_lo, u_hi",
                         test_cases, ids=ids)
def test_affiliated_values(n_players, valuation_size, u_lo, u_hi):
    """Test the Affiliated Values Sampler."""


    ### test with valuation size 1.
    s = vs.AffiliatedValuationObservationSampler(n_players, valuation_size, u_lo, u_hi, batch_size)


    v,o = s.draw_profiles()
    assert o.device == v.device, "Observations and Valuations should be on same device"
    assert o.device.type == device, "Standard device should be cuda, if available!"

    assert o.shape == v.shape

    # Test the value distribution
    for i in range(n_players):
        assert torch.equal(v[:,0,:], v[:,i,:]), "all players should have the same valuation!"

    # observations o_i = z_i + s are Irwin-Hall(2) distributed for U[0,1]:

    o_mean = u_hi + u_lo
    o_std = sqrt(2/12) * (u_hi - u_lo)

    # valuations v_i = \sum_i(o_i) / n = s + (\sum_i z_i)/n.
    # the first term is uniform, the second term is 1/n * Irvin-Hall(n)
    # we thus have E[v] = E[s] + E[\sum_i z_i]
    # = (u_hi-u_lo)/2 + u_lo + (u_hi-u_lo) * n/2 * 1/n + u_lo and thus
    v_mean = u_hi + u_lo
    # and Var(V) = Var(S) + 1/n² * Var(Irvin-Hall(n)) as Z and S are independent.
    #            = 1/12 + 1/n² * (n/12) = (n+1)/(12*n)
    v_std = (u_hi - u_lo) * sqrt((n_players+1)/(12*n_players))


    expected_valuation_mean = torch.tensor([[v_mean]*valuation_size]*n_players, device=v.device)
    expected_valuation_std = torch.tensor([[v_std]*valuation_size]*n_players, device=v.device)
    expected_observation_mean =  torch.tensor([[o_mean]*valuation_size]*n_players, device=o.device)
    expected_observation_std =  torch.tensor([[o_std]*valuation_size]*n_players, device=o.device)

    assert torch.allclose(v.mean(dim=0), expected_valuation_mean, rtol = 0.02), \
        "unexpected valuation sample mean!"
    # mean of obs should be identical to mean of vals.
    assert torch.allclose(o.mean(dim=0), expected_observation_mean, rtol = 0.02), \
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
    v,o = s.draw_profiles(batch_sizes = conditional_outer_batch_size)
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

    # TODO: test edge cases.
