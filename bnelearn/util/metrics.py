"""This module implements metrics that may be interesting."""

import torch
from bnelearn.strategy import Strategy


def strategy_norm(strategy1: Strategy, strategy2: Strategy, valuations: torch.Tensor, p: float=2) -> float:
    """
    Calculates the approximate Lp-norm between two strategies approximated
    via Monte-Carlo integration on a sample of valuations that have been drawn according to the prior.

    The function Lp norm is given by (\int_V |s1(v) - s2(v)|^p dv)^(1/p).
    With Monte-Carlo Integration this is approximated by
     (|V|/n * \sum_i^n(|s1(v) - s2(v)|^p) )^(1/p)  where |V| is the volume of the set V.

    Here, we ignore the volume to get something closer to root mean squared error.
    If p=Infty, this evaluates to the supremum.
    """
    b1 = strategy1.play(valuations)
    b2 = strategy2.play(valuations)

    # valuations are: [n_batch x n_bundles]

    if p == float('Inf'):
        return (b1 - b2).abs().max()

    # finite p
    n = float(valuations.shape[0])
    #usually Lp uses the volume
    #volume =  (valuations.max(dim=0)[0] - valuations.min(dim=0)[0]).prod()
    #print('vol ' + str(volume.item()))
    volume = 1.
    return torch.dist(b1, b2, p=p)*(volume/n)**(1/p)