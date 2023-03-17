"""Some utilities to work with torch.Distributions."""
import torch
from copy import deepcopy

_ERR_MSG_UNEXPECTED_DEVICE = "unexpected output device"

def copy_dist_to_device(dist, device):
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
