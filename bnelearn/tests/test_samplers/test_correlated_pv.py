""" This pytest test file checks whether valuation and observation samplers have the
expected behaviour"""

from math import sqrt

import numpy as np
import pytest
import torch

import bnelearn.valuation_sampler as vs


device = 'cuda' if torch.cuda.is_available() else 'cpu'


batch_size = 2**20
alternative_batch_size = 2**15
conditioned_inner_batch_size = 2**17


# helper functions
def correlation(valuation_profile):
    """Pearson correlation between valuations of bidders.

    valuation_profile should be (batch x n_players x 1)
    """
    assert valuation_profile.dim() >= 3, "invalid valuation profile."
    if valuation_profile.dim() > 3:
        # collapse batch dimensions
        *batch_sizes, n_players, valuation_size = valuation_profile.shape
        valuation_profile = valuation_profile.view(-1, n_players, valuation_size)
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
    if expected_std is not None:
        assert torch.allclose(valuation_profile.std(dim=0), expected_std.to(device), atol=0.02), \
            "unexpected sample variance!"
    if expected_correlation is not None:
        for k in range(valuation_profile.shape[2]):
            assert torch.allclose(correlation(valuation_profile[:,:,[k]]),
                                  expected_correlation.cpu(), atol=0.01), \
                "unexpected correlation between bidders!"

def check_validity_of_conditional_sampler(sampler: vs.CorrelatedSymmetricUniformPVSampler,
                                          n_players, valuation_size, u_lo, u_hi):

    """Check runtime at minimum, maximum, midpoint inputs, as well as shapes,
       devices and known entries.
    """

    conditioned_observation = \
            torch.tensor([[u_lo],
                          [u_hi],
                          [u_lo + 0.5*(u_hi-u_lo)],
                          [u_lo + 0.25*(u_hi-u_lo)]]) \
                .repeat(1, valuation_size)
    
    outer_batch, observation_size = conditioned_observation.shape

    #### test conditional sampling for different players (first and last)
    for i in [0, n_players-1]:


        co, cv = sampler.draw_conditional_profiles(
            i, conditioned_observation,
            conditioned_inner_batch_size)

        assert co.device == cv.device, "Private values, obs and valuations should be identical."
        assert co.device.type == device, "Output is not on excpected standard device!"

        assert torch.equal(
            co[...,i,:],
            conditioned_observation \
                .to(sampler.default_device) \
                .view(outer_batch, 1, observation_size)
                .repeat(1, conditioned_inner_batch_size, 1)
        ), "conditioned sample must respect conditioned_observation input!"

    # When outer batch is drawn from real distribution and inner_batch size is
    # 1, then the conditional samples should have the same distribution as 'regular'
    # samples.

    # get outer obs
    _, o = sampler.draw_profiles(batch_size)

    for i in range(o.shape[1]): # draw conditionals for each player
        _, co = sampler.draw_conditional_profiles(
            conditioned_player=i,
            conditioned_observation=o[:,i,:],
            inner_batch_size=1
        )

        # with batch_size = 1 (no repitition of inputs),
        # co should follow same distribution as o
        assert torch.allclose(o.mean(dim=0), co.mean(dim=0), atol=.01)
        assert torch.allclose(o.std(dim=0), co.std(dim=0), atol=.01)
        for k in range(valuation_size):
            assert torch.allclose(
                correlation( o[:,:,[k]]),
                correlation(co[...,:,[k]]),
                atol=0.01), "Unexpected correlation matrix encountered!"



# test cases
# base case is 2p, U[0,1], correlation of 0.5
ids, test_cases = zip(*[
    #                    nplayers, valuation_size, gamma,       u_lo,  u_hi
    ['base_case',       (2,        1,              0.5,         0,     1     )],
    ['perfect_corr',    (2,        1,              1.0,         0,     1     )],
    ['independent',     (2,        1,              0.0,         0,     1     )],
    ['other_corr',      (2,        1,              0.3,         0,     1     )],
    ['three_players',   (3,        1,              0.5,         0,     1     )],
    ['two_valuations',  (2,        2,              0.5,         0,     1     )],
    ['scaled_bounds',   (2,        1,              0.5,         0,     2     )],
    ['shifted_bounds',  (2,        1,              0.5,         1,     2     )],
    ['affine_bounds',   (2,        1,              0.5,         1,     3     )],
])


@pytest.mark.parametrize("n_players, valuation_size, gamma, u_lo, u_hi",
                         test_cases, ids=ids)
def test_correlated_constant_weight_pv(n_players, valuation_size,
                                       gamma, u_lo, u_hi):
    """Functionality and correcness test of the Constant Weights sampler.
    We test
    - correctness of sample on standard device with standard batch_size
      - dimensions and output devices
      - ipv, i.e. valuations == observations
      - correctness of mean, std (where known) of marginals
      - correlation matrix
    - conditioned sampling on one player's observation
      - has correct shapes and devices
      - has correct entries for the given player
      - otherwise looks like a valid sample
    - additionally, we test dimensions and output devices for manually specified
      devises or batch_sizes.
    """


    marginal_mean = torch.tensor((u_hi - u_lo)/2 + u_lo) \
        .repeat([n_players, valuation_size])
    # In some cases, we know the variance of the marginals:
    ## when correlation is 0 or 1, marginals are U[u_lo,u_hi] distributed
    ## the variance of U[0,1] is 1/12
    marginal_std_0_or_1 = sqrt( (u_hi - u_lo) / 12)
    ## when correlation is 0.5, marginals are
    # (u_hi - u_lo)/2 * Irwin-Hall(2) + u_lo distributed.
    ## the variance of Irwin-Hall(2) is 2/12.
    marginal_std_one_half = sqrt(2/12) * (u_hi - u_lo)/ 2

    marginal_std = marginal_std_0_or_1 if gamma in [0.0, 1.0] \
        else marginal_std_one_half if gamma == 0.5 \
        else None
    if marginal_std is not None:
        marginal_std = torch.tensor(marginal_std) \
            .repeat([n_players, valuation_size])

    # 1s on diagonal, correlation on non-diagonal entries
    expected_correlation = torch.eye(n_players) * (1-gamma) + gamma


    s = vs.ConstantWeightCorrelatedSymmetricUniformPVSampler(
        n_players, valuation_size,
        gamma,
        u_lo, u_hi, batch_size)


    ### test standard profile sampling

    v,o = s.draw_profiles()
    assert o.device == v.device, "Observations and Valuations should be on same device"
    assert o.device.type == device, "Standard device should be cuda, if available!"

    assert torch.equal(o, v), "observations and valuations should be identical in IPV"

    check_validity(v,
                   expected_shape= torch.Size([batch_size, n_players, valuation_size]),
                   expected_mean = marginal_mean,
                   expected_std = marginal_std,
                   expected_correlation= expected_correlation)

    ## sample with a different batch size
    v,o = s.draw_profiles(alternative_batch_size)
    assert v.shape == torch.Size([alternative_batch_size, n_players, valuation_size]), \
        "failed to sample with nonstandard size."

    ## sample on cpu
    v,o = s.draw_profiles(device='cpu')
    assert v.device.type == 'cpu', "sampling didn't respect device parameter."

    check_validity_of_conditional_sampler(s, n_players, valuation_size, u_lo, u_hi)


@pytest.mark.parametrize("n_players, valuation_size, gamma, u_lo, u_hi",
                         test_cases, ids=ids)
def test_correlated_Bernoulli_weight_pv(n_players, valuation_size,
                                        gamma, u_lo, u_hi):
    """Functionality and correcness test of the Bernoulli Weights sampler.
    We test
    - correctness of sample on standard device with standard batch_size
      - dimensions and output devices
      - ipv, i.e. valuations == observations
      - correctness of mean, std (where known) of marginals
      - correlation matrix
    - conditioned sampling on one player's observation
      - has correct shapes and devices
      - has correct entries for the given player
      - otherwise looks like a valid sample
    - additionally, we test dimensions and output devices for manually specified
      devises or batch_sizes.
    """


    marginal_mean = torch.tensor((u_hi - u_lo)/2 + u_lo) \
        .repeat([n_players, valuation_size])

    # in the Bernoulli weights model, the marginals are themselves U[u_lo,u_hi]
    # distributed.
    marginal_std = torch.tensor(sqrt(1/12) * (u_hi-u_lo)) \
            .repeat([n_players, valuation_size])

    # 1s on diagonal, correlation on non-diagonal entries
    expected_correlation = torch.eye(n_players) * (1-gamma) + gamma


    s = vs.BernoulliWeightsCorrelatedSymmetricUniformPVSampler(
        n_players, valuation_size,
        gamma,
        u_lo, u_hi, batch_size)


    ### test standard profile sampling

    v,o = s.draw_profiles()
    assert o.device == v.device, "Observations and Valuations should be on same device"
    assert o.device.type == device, "Standard device should be cuda, if available!"

    assert torch.equal(o, v), "observations and valuations should be identical in IPV"

    check_validity(v,
                   expected_shape= torch.Size([batch_size, n_players, valuation_size]),
                   expected_mean = marginal_mean,
                   expected_std = marginal_std,
                   expected_correlation= expected_correlation)

    ## sample with a different batch size
    v,o = s.draw_profiles(alternative_batch_size)
    assert v.shape == torch.Size([alternative_batch_size, n_players, valuation_size]), \
        "failed to sample with nonstandard size."

    ## sample on cpu
    v,o = s.draw_profiles(device='cpu')
    assert v.device.type == 'cpu', "sampling didn't respect device parameter."

    check_validity_of_conditional_sampler(s, n_players, valuation_size, u_lo, u_hi)
