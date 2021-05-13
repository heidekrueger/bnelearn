""" This pytest test file checks whether valuation and observation samplers have the
expected behaviour"""

from math import sqrt

import torch
import numpy as np

import bnelearn.valuation_sampler as vs

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def test_uniform_symmetric_ipv():
    """Test the UniformSymmetricIPVSampler."""

    ### test with valuation size 1.
    s = vs.MultiUnitValuationObservationSampler(n_players, n_items, max_demand,
                                                u_lo, u_hi, batch_size, device)


    v,o = s.draw_profiles()
    assert o.device == v.device, "Observations and Valuations should be on same device"
    assert o.device.type == device, "Standard device should be cuda, if available!"

    assert torch.equal(o, v), "observations and valuations should be identical in IPV"

    ## check that

    ## mask is fulfilled
    ## things are sorted
    ## kth valuation should be distritubuted according to the kth order
    ## statistic of a uniform, i.e.
    # Beta(k, n_items + 1 - k)

    assert 1==0, "TODO Continue writing tests here!"