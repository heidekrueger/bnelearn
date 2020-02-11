"""This module implements metrics that may be interesting."""

import torch
from bnelearn.strategy import Strategy


def norm_actions(b1: torch.Tensor, b2: torch.Tensor, p: float = 2) -> float:
    """
    Calculates the approximate "mean" Lp-norm between two action vectors.
    (\sum_i=1^n(1/n * |b1 - b2|^p))^(1/p)
  
    If p=Infty, this evaluates to the supremum.
    """
    assert b1.shape == b2.shape

    if p == float('Inf'):
        return (b1 - b2).abs().max()

    # finite p
    n = float(b1.shape[0])

    return torch.dist(b1, b2, p=p)*(1./n)**(1/p)

def norm_strategies(strategy1: Strategy, strategy2: Strategy, valuations: torch.Tensor, p: float=2) -> float:
    """
    Calculates the approximate "mean" Lp-norm between two strategies approximated
    via Monte-Carlo integration on a sample of valuations that have been drawn according to the prior.

    The function Lp norm is given by (\int_V |s1(v) - s2(v)|^p dv)^(1/p).
    With Monte-Carlo Integration this is approximated by
     (|V|/n * \sum_i^n(|s1(v) - s2(v)|^p) )^(1/p)  where |V| is the volume of the set V.

    Here, we ignore the volume. This givses us the RMSE for L2, supremum for Linfty, etc.
    """
    b1 = strategy1.play(valuations)
    b2 = strategy2.play(valuations)

    return norm_actions(b1, b2, p)

def norm_strategy_and_actions(strategy, actions, valuations: torch.Tensor, p: float=2) -> float:
    """Calculates the norm as above, but given one action vector and one strategy.
    The valuations must match the given actions.
    
    This helper function is useful when recalculating an action vector is prohibitive and it should be reused.

    """
    s_actions = strategy.play(valuations)

    return norm_actions(s_actions, actions, p)