""" This pytest test file checks whether valuation and observation samplers have the
expected behaviour"""

from math import sqrt

import torch
import numpy as np

import bnelearn.valuation_sampler as vs

device = 'cuda' if torch.cuda.is_available() else 'cpu'
n_players = 3

batch_size = 2**20
alternative_batch_size = 2**15

# parameters for tests with uniform distributions
u_lo = 0.
u_hi = 10.
u_mean = 5.
u_std = (u_hi - u_lo) * sqrt(1/12)


# parameters for tests with Normal distributions
n_mean = 10.
# we clip vals at 0, so make sure std is small to not affect sample mean!
n_std = 3.

# helper functions
def correlation(valuation_profile):
    """Pearson correlation between valuations of bidders.

    valuation_profile should be (batch x n_players x 1)
    """
    assert valuation_profile.dim() == 3, "invalid valuation profile."
    assert valuation_profile.shape[2] == 1, "correlation matrix only valid for 1-d valuations"

    return torch.from_numpy(
        np.corrcoef(valuation_profile.squeeze(-1).t().cpu())
        ).float() #numpy is float 64

def check_validity(valuation_profile, expected_shape,
                   expected_mean, expected_std, expected_correlation=None):
    """Checks whether a given batch of profiles has expected shape, mean, std
       and correlation matrix.
    """
    assert valuation_profile.dim() == 3, "invalid number of dimensions!"
    assert valuation_profile.shape == expected_shape, "invalid shape!"
    assert torch.allclose(valuation_profile.mean(dim=0), expected_mean.to(device), atol=0.02), \
        "unexpected sample mean!"
    assert torch.allclose(valuation_profile.std(dim=0), expected_std.to(device), atol=0.02), \
        "unexpected sample variance!"
    if expected_correlation is not None:
        assert torch.allclose(correlation(valuation_profile), expected_correlation.cpu(), atol=0.01), \
            "unexpected correlation between bidders!"


def test_uniform_symmetric_ipv():
    """Test the UniformSymmetricIPVSampler."""

    ### test with valuation size 1.
    s = vs.UniformSymmetricIPVSampler(u_lo, u_hi, n_players, 1, batch_size)


    v,o = s.draw_profiles()
    assert o.device == v.device, "Observations and Valuations should be on same device"
    assert o.device.type == device, "Standard device should be cuda, if available!"

    assert torch.equal(o, v), "observations and valuations should be identical in IPV"

    check_validity(v,
                   expected_shape= torch.Size([batch_size, n_players, 1]),
                   expected_mean = torch.tensor(u_mean).repeat([n_players, 1]),
                   expected_std = torch.tensor(u_std).repeat([n_players, 1]),
                   expected_correlation=torch.eye(3)
                   )

    ## sample with a different batch size
    v,o = s.draw_profiles(alternative_batch_size)
    assert v.shape == torch.Size([alternative_batch_size, n_players, 1]), \
        "failed to sample with nonstandard size."

    ## sample on cpu
    v,o = s.draw_profiles(device='cpu')
    assert v.device.type == 'cpu', "sampling didn't respect device parameter."


    ### test with valuation size 4.
    valuation_size = 4
    s = vs.UniformSymmetricIPVSampler(u_lo, u_hi, n_players, valuation_size, batch_size)
    v,o = s.draw_profiles()
    assert v.device.type == device, "Standard device should be cuda, if available!"

    check_validity(v,
                   expected_shape= torch.Size([batch_size, n_players, valuation_size]),
                   expected_mean = torch.tensor(u_mean).repeat([n_players, valuation_size]),
                   expected_std = torch.tensor(u_std).repeat([n_players, valuation_size]),
                   )

    n_grid_points = 2**valuation_size
    grid = s.generate_valuation_grid(0, 2**valuation_size)
    assert grid.size() == torch.Size([n_grid_points, valuation_size]), "Unexpected Grid"


def test_gaussian_symmetric_ipv():
    """Test the GaussianSymmetricIPVSampler."""

    # test with valuation size 1.
    s = vs.GaussianSymmetricIPVSampler(n_mean, n_std, n_players, 1, batch_size)


    v,o = s.draw_profiles()
    assert o.device == v.device, "Observations and Valuations should be on same device"
    assert v.device.type == device, "Standard device should be cuda, if available!"

    assert torch.equal(o, v), "observations and valuations should be identical in IPV"

    check_validity(v,
                   expected_shape= torch.Size([batch_size, n_players, 1]),
                   expected_mean = torch.tensor(n_mean).repeat([n_players, 1]),
                   expected_std = torch.tensor(n_std).repeat([n_players, 1]),
                   expected_correlation=torch.eye(3)
                   )

    ## sample with a different batch size
    v,o = s.draw_profiles(alternative_batch_size)
    assert v.shape == torch.Size([alternative_batch_size, n_players, 1]), \
        "failed to sample with nonstandard size."

    ## sample on cpu
    v,o = s.draw_profiles(device='cpu')
    assert v.device.type == 'cpu', "sampling didn't respect device parameter."

    ### ensure valuation clipping at zero by using std=mean
    s = vs.GaussianSymmetricIPVSampler(n_mean, n_mean, n_players, 1, batch_size)
    v,o = s.draw_profiles()
    assert torch.all(v.ge(0)), "negative draws should be clipped to zero!"

    ### test with valuation size 4.
    valuation_size = 4
    s = vs.GaussianSymmetricIPVSampler(n_mean, n_std, n_players, valuation_size, batch_size)
    v,o = s.draw_profiles()
    assert v.device.type == device, "Standard device should be cuda, if available!"

    check_validity(v,
                   expected_shape= torch.Size([batch_size, n_players, valuation_size]),
                   expected_mean = torch.tensor(n_mean).repeat([n_players, valuation_size]),
                   expected_std = torch.tensor(n_std).repeat([n_players, valuation_size]),
                   )
