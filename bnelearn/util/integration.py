"""Some utilities to leverage parallel computation of the types of integrals
that arise in BNEs."""
import torch

def cumulatively_integrate(f: callable, upper_bounds: torch.tensor, lower_bound: float=0.0,
                  n_evaluations: int=64):
    """Integrate the fucntion `f` on the intervals `[[lower_bound, upper_bounds[0],
    [lower_bound, upper_bounds[1], ...]` that sahre a common lower bound.
    
    This function sorts the upper bounds, decomposes the integral into those 
    between any two adjacent points in lower_bound, *upper_bounds,
    calculates each partial integral using pytorch's trapezoid rule with n_evalautions
    sampling points per interval, then stichtes the resulting masses together to
    achieve the desired output.
    Note that this way, we can use pytorch.trapz in parallel over all domains and
    integrate directly on cuda, if desired.

    Arguments:
        f: callable, function to be integrated.
        upper_bounds: torch.tensor of shape (batch_size, 1) that specifies the
            upper integration bounds.
        lower_bound: float that specifies the lower bound of all domains.
        n_evaluations: int that specifies the number of function evaluations per
            indivdidual interval.

    Returns:
        integrals: torch.tensor of shape (batch_size, 1).
    """
    upper_bounds = upper_bounds.view(-1, 1)
    device = upper_bounds.device
    batch_size = upper_bounds.shape[0]

    # sort domains
    upper_bounds_sorted, index_sorted = upper_bounds.flatten().sort()

    # grid interpolation for the N evaluation points per integral
    lower_bound = torch.cat(
        [torch.tensor([lower_bound], device=device),
        upper_bounds_sorted[:-1]])
    domains_bounds = torch.cat(
        [
            lower_bound.view(-1, 1),
            torch.zeros((batch_size, n_evaluations), device=device),
            upper_bounds_sorted.view(-1, 1)
        ],
        axis=1)
    for n in range(1, n_evaluations+1):
        domains_bounds[:, n] = domains_bounds[:, 0] \
            + (float(n) / (n_evaluations + 1.0)) * (domains_bounds[:, -1] - domains_bounds[:, 0])

    # evaluate function and integrate
    F = f(domains_bounds)
    integrals_sorted = torch.trapz(F, domains_bounds)

    # restore original order
    integrals = torch.cumsum(integrals_sorted, 0) \
        .gather(0, index_sorted.argsort()) \
        .view_as(upper_bounds)

    return integrals
