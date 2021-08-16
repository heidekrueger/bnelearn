"""Some utilities to leverage parallel computation of the types of integrals
that arise in BNEs."""
import torch

def cum_integrate(f: callable, domains: torch.tensor, lower_bound: float=0.0,
                  N: int=64):
    """Integrate the fucntion `f` on the intervals `[lower_bound, domains[0],
    [lower_bound, domains[1], ...` that sahre a common lower bound using the
    trapez integration rule `torch.trapz` and `N` evaluation points.

    Arguments:
        f: callable, function to be integrated.
        domains: torch.tensor of shape (batch_size, 1) that specifies the
            integration bounds.
            lower_bound
        lower_bound: float that specifies the lower bound of all domains.
        N: int that specifies the number of function evaluations per
            indivdidual interval.

    Returns:
        integrals: torch.tensor of shape (batch_size, 1).
    """
    domains = domains.view(-1, 1)
    device = domains.device
    batch_size = domains.shape[0]

    # sort domains
    domains_sorted, index_sorted = domains.flatten().sort()

    # grid interpolation for the N evaluation points per integral
    lower_bound = torch.cat(
        [torch.tensor([lower_bound], device=device),
        domains_sorted[:-1]])
    domains_bounds = torch.cat(
        [
            lower_bound.view(-1, 1),
            torch.zeros((batch_size, N), device=device),
            domains_sorted.view(-1, 1)
        ],
        axis=1)
    for n in range(1, N+1):
        domains_bounds[:, n] = domains_bounds[:, 0] \
            + (float(n) / (N + 1.0)) * (domains_bounds[:, -1] - domains_bounds[:, 0])

    # evaluate function and integrate
    F = f(domains_bounds)
    integrals_sorted = torch.trapz(F, domains_bounds)

    # restore original order
    integrals = torch.cumsum(integrals_sorted, 0) \
        .gather(0, index_sorted.argsort()) \
        .view_as(domains)

    return integrals
