""" This pytest test file checks whether valuation and observation samplers have the
expected behaviour"""

import bnelearn.valuation_sampler as vs
import pytest
import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'

batch_size = 2**20

# for first test: outer x 1
conditional_outer_batch_size = 2**15
# for second test: few x inner
conditional_inner_batch_size = 2**18


ids, test_cases = zip(*[
    #                      nplayers, n_items, max_demand, u_lo,  u_hi
    ['0_base_case',       (3,        4,       4,          0.0,   1.0     )],
    ['1_scaled'   ,       (3,        4,       4,          0.0,   5.0     )],
    ['2_shifted',         (3,        4,       4,          1.0,   2.0     )],
    ['3_affine',          (3,        4,       4,          2.0,   4.0     )],
    ['4a_demand_limit3',  (3,        4,       3,          0.0,   1.0     )],
    ['4b_demand_limit1',  (3,        4,       1,          0.0,   1.0     )],
    ['4_two_items',       (3,        2,       2,          0.0,   1.0     )],
    ['5_one_item',        (3,        2,       1,          0.0,   1.0     )],
    ['6_all_mixed',       (5,        3,       2,          2.0,   4.0     )],
])


@pytest.mark.parametrize("n_players, n_items, max_demand, u_lo, u_hi",
                         test_cases, ids=ids)
def test_uniform_symmetric_ipv(n_players, n_items, max_demand, u_lo, u_hi):
    """Test the UniformSymmetricIPVSampler."""

    ### test with valuation size 1.
    s = vs.MultiUnitValuationObservationSampler(n_players, n_items, max_demand,
                                                u_lo, u_hi, batch_size, device)


    v,o = s.draw_profiles()
    assert o.device == v.device, "Observations and Valuations should be on same device"
    assert o.device.type == device, "Standard device should be cuda, if available!"

    assert torch.equal(o, v), "observations and valuations should be identical in IPV"

    ## TODO: check that
    ## * mask is adhered to in the right way
    ## * things are sorted
    ## * kth valuation should be distritubuted according to the kth order statistic of a uniform RV, i.e.
    ##     Beta(k, n_items + 1 - k)

    pytest.skip("Test not completely implemented!")
