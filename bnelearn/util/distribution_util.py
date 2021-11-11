"""Some utilities to work with torch.Distributions."""
from math import prod, sqrt
from copy import deepcopy

import torch
from torch.distributions import Distribution, Uniform, Normal


_ERR_MSG_UNEXPECTED_DEVICE = "unexpected output device"

def copy_dist_to_device(dist: Distribution, device):
    """A quick an dirty workaround to move torch.Distributions from one device to another.

    To do so, we return a copy of the original distribution with all its tensor-valued members
    moved to the desired device.

    Note that this will only work for the most basic distributions and will likely fail for complex
    or composed distribution objects. See https://github.com/pytorch/pytorch/issues/7795 for details.
    """
    result = deepcopy(dist)
    for (k,v) in result.__dict__.items():
        if isinstance(v, torch.Tensor):
            result.__dict__[k] = v.to(device)

    # quick-check whether our conversion heuristic has worked and fail if it hasn't.
    try:
        ex_device = torch.tensor(0.0, device=device).device
        p = result.cdf(torch.tensor(0.0))
        assert p.device == ex_device, _ERR_MSG_UNEXPECTED_DEVICE
        p = result.log_prob(torch.tensor(0.0))
        assert p.device == ex_device, _ERR_MSG_UNEXPECTED_DEVICE
        p = result.sample()
        assert p.device == ex_device, _ERR_MSG_UNEXPECTED_DEVICE
    except Exception as e:
        raise NotImplementedError(f"Device conversion of {dist} failed. " + \
            "This method only works for the most basic distributions. " + \
                "You may need to create the desired distribution ad-hoc.") \
            from e

    return result


def draw_sobol(distribution: Distribution,
               batch_size: int, device, scramble = False) -> torch.Tensor:
    """Draws a batch of samples from distribution via low-descrepancy sobol sequences"""
    # determine the dimension of the sobol sequence
    event_shape = distribution._batch_shape # n_players x valuation_size
    sobol_dim = prod(event_shape)

    ## Draw uniform sobol sample
    se = torch.quasirandom.SobolEngine(dimension=sobol_dim, scramble = scramble)

    # first point is always [0.0, ... 0.0] --> ignore to avoid bias
    se.draw(1)
    # batch_size x n_players x valuation_size
    sample = se.draw(batch_size) \
        .view([batch_size, *event_shape]) \
        .to(device)

    ## Transform to sample of desired transformation
    return _transform_batch(sample, distribution)

def _transform_batch(sample: torch.Tensor, distribution: Distribution) -> torch.Tensor:
    """Transforms a batch of Standard-Uniform samples to
    a sample of `distribution`. Via the icdf method.

    Args:
        sample (torch.Tensor of size batch x *event_size)
        distribution (Distribution with batch_size equal to `sample.shape[1:]`)
    """
    device = sample.device

    if isinstance(distribution, Uniform):
        lo = distribution.low.to(device)
        hi = distribution.high.to(device)

        return sample * (hi-lo) + lo
    elif isinstance(distribution, Normal):
        loc = distribution.loc.to(device)
        scale = distribution.scale.to(device)
        return loc + scale * torch.erfinv(2*sample - 1) * sqrt(2)
