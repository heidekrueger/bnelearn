"""This module contains implementations of known Bayes-Nash equilibria in several specific settings."""
from typing import Callable, List, Union

import torch
import numpy as np
from scipy import interpolate, integrate, optimize
###############################################################################
#######   Known equilibrium bid functions                                ######
###############################################################################
# Define known BNE functions top level, so they may be pickled for parallelization
# These are called millions of times, so each implementation should be
# setting specific, i.e. there should be NO setting checks at runtime.


###############################################################################
######  Single-Item IPV
###############################################################################

def bne_fpsb_ipv_symmetric_generic_prior_risk_neutral(
        valuation: Union[torch.Tensor, np.ndarray, float],
        n_players: int, prior_cdf: Callable, **kwargs) -> torch.Tensor:
    """BNE in symmetric IPV first-price sealed bid auctions with
        generic prior value distributions and quasi-linear utilities.

    Reference: Krishna (2009), Chapter 2, Proof of Proposition 2.2.
    """
    if not prior_cdf(torch.tensor(0.0)).device.type == 'cpu':
        raise ValueError("prior_cdf is required to return CPU-tensors rather than gpu tensors, " + \
            "otherwise we will encounter errors when using numerical integration via scipy together with " + \
            "torch.multiprocessing. For now, please provide a cpu-version of the prior-cdf.")
    if not isinstance(valuation, torch.Tensor):
        # For float and numpy --> convert to tensor (relevant for plotting)
        valuation = torch.tensor(valuation, dtype=torch.float)
    # For float / 0d tensors --> unsqueeze to allow list comprehension below
    if valuation.dim() == 0:
        valuation.unsqueeze_(0)
    # shorthand notation for F^(n-1)
    Fpowered = lambda v: torch.pow(prior_cdf(torch.tensor(v)), n_players - 1)
    # do the calculations
    numerator = torch.tensor(
        [integrate.quad(Fpowered, 0, v)[0] for v in valuation],
        device=valuation.device
    ).reshape(valuation.shape)
    return valuation - numerator / Fpowered(valuation)


def bne_fpsb_ipv_symmetric_uniform_prior(
    valuation: torch.Tensor, n: int, r: float, u_lo, u_hi, **kwargs) -> torch.Tensor:
    """BNE in the special case of symmetric FPSB IPV auctions where priors
    are symmetric uniform.

    Compared to the generic case above, we also know this BNE for risk-averse agents.

    Reference:

    In terms of implementation, note that this equlibrium is fully vectorized."""
    return u_lo + (valuation - u_lo) * (n - 1) / (n - 1.0 + r)


def truthful_bid(valuation: torch.Tensor, **kwargs) -> torch.Tensor:
    """Truthful bidding function: A BNE in any VCG auction."""
    return valuation


def bne_fpsb_ipv_asymmetric_uniform_overlapping_priors_risk_neutral(
        valuation: torch.Tensor or float, player_position: int,
        u_lo: List, u_hi: List) -> torch.Tensor:
    """
    BNE in asymmetric 2-player IPV first-price auctions, when both players have quasilinear-utilities
    and uniform valuation priors that share the same lower lower bound, but may differ
    in the upper bound.

    Reference: https://link.springer.com/article/10.1007/BF01271133
    """

    if not isinstance(valuation, torch.Tensor):
        valuation = torch.tensor(valuation, dtype=torch.float)
    # unsqueeze if simple float
    if valuation.dim() == 0:
        valuation.unsqueeze_(0)

    c = 1 / (u_hi[0] - u_lo[0]) ** 2 - 1 / (u_hi[1] - u_lo[0]) ** 2
    factor = 2 * player_position - 1  # -1 for 0 (weak player), +1 for 1 (strong player)
    denominator = 1.0 + torch.sqrt(1 + factor * c * (valuation - u_lo[0]) ** 2)
    bid = u_lo[0] + (valuation - u_lo[0]) / denominator
    return torch.max(bid, torch.zeros_like(bid))


def bne1_kaplan_zhamir(u_lo: List, u_hi: List):
    """
    BNE in asymmetric 2-player IPV first-price auctions, when both players have quasilinear-utilities
    and uniform priors, that do NOT share the same lower bound and are nonoverlapping.

    This setting was analyzed by Kaplan and Zhamir (2014) and was found to have multiple
    pure BNE. (At least 3).

    Reference: Equilibrium 1 of https://link.springer.com/article/10.1007/s40505-014-0049-1

    NOTE: this implementation is hard-coded for the bounds used in the paper above,
    i.e. [0,5] and [6,7], but we do not perform any checks at runtime for performance reasons!

    """
    interpol_points = 2**11

    # 1. Solve implicit bid function
    v1 = np.linspace(u_lo[0], u_hi[0], interpol_points)
    v2 = np.linspace(u_lo[1], u_hi[1], interpol_points)

    def inverse_bid_player_1(bid):
        return 36 / ((2 * bid - 6) * (1 / 5) * np.exp(9 / 4 + 6 / (6 - 2 * bid)) + 24 - 4 * bid)

    def inverse_bid_player_2(bid):
        return 6 + 36 / ((2 * bid - 6) * 20 * np.exp(-9 / 4 - 6 / (6 - 2 * bid)) - 4 * bid)

    u_lo_cut = 0
    for i in range(interpol_points):
        if v1[i] > u_lo[1] / 2:
            u_lo_cut = i
            break

    b1 = np.copy(v1) # truthful at beginning
    b1[u_lo_cut:] = np.array([optimize.broyden1(lambda x: inverse_bid_player_1(x) - v, v)
                              for v in v1[u_lo_cut:]])
    b2 = np.array([optimize.broyden1(lambda x: inverse_bid_player_2(x) - v, v)
                   for v in v2])

    opt_bid_function = [
        interpolate.interp1d(v1, b1, kind=1),
        interpolate.interp1d(v2, b2, kind=1)
    ]

    # 2. return interpolation of bid function
    def _optimal_bid(valuation: torch.Tensor or float, player_position: int):
        if not isinstance(valuation, torch.Tensor):
            valuation = torch.tensor(valuation, dtype=torch.float)
        # unsqueeze if simple float
        if valuation.dim() == 0:
            valuation.unsqueeze_(0)
        bid = torch.tensor(
            opt_bid_function[player_position](valuation.cpu().numpy()),
            device=valuation.device,
            dtype=valuation.dtype
        )
        return bid

    return _optimal_bid


def bne2_kaplan_zhamir(
        valuation: torch.Tensor or float, player_position: int,
        u_lo: List, u_hi: List):
    """
    BNE in asymmetric 2-player IPV first-price auctions, when both players have quasilinear-utilities
    and uniform priors, that do NOT share the same lower bound and are nonoverlapping.

    This setting was analyzed by Kaplan and Zhamir (2014) and was found to have multiple
    pure BNE. (At least 3).

    Reference: Equilibrium 2 of https://link.springer.com/article/10.1007/s40505-014-0049-1

    NOTE: this implementation is hard-coded for the bounds used in the paper above,
    i.e. [0,5] and [6,7], but we do not perform any checks at runtime for performance reasons!
    """
    if not isinstance(valuation, torch.Tensor):
        valuation = torch.tensor(valuation, dtype=torch.float)
    # unsqueeze if simple float
    if valuation.dim() == 0:
        valuation.unsqueeze_(0)

    if player_position == 0:
        bids = torch.zeros_like(valuation)
        bids[valuation > 4] = valuation[valuation > 4] / 2 + 2
        bids[valuation <= 4] = valuation[valuation <= 4] / 4 + 3
    else:
        bids = valuation / 2 + 1

    return bids


def bne3_kaplan_zhamir(
        valuation: torch.Tensor or float, player_position: int,
        u_lo: List, u_hi: List):
    """
    BNE in asymmetric 2-player IPV first-price auctions, when both players have quasilinear-utilities
    and uniform priors, that do NOT share the same lower bound and are nonoverlapping.

    This setting was analyzed by Kaplan and Zhamir (2014) and was found to have multiple
    pure BNE. (At least 3).

    Reference: Equilibrium 3 of https://link.springer.com/article/10.1007/s40505-014-0049-1

    NOTE: this implementation is hard-coded for the bounds used in the paper above,
    i.e. [0,5] and [6,7], but we do not perform any checks at runtime for performance reasons!
    """
    if not isinstance(valuation, torch.Tensor):
        valuation = torch.tensor(valuation, dtype=torch.float)
    # unsqueeze if simple float
    if valuation.dim() == 0:
        valuation.unsqueeze_(0)

    if player_position == 0:
        bids = valuation / 5 + 4
    else:
        bids = 5 * torch.ones_like(valuation)

    return bids

###############################################################################
######  Single-Item Non-IPV
###############################################################################


def bne_3p_mineral_rights(
        valuation: torch.Tensor, player_position: int = 0) -> torch.Tensor:
    """BNE in the 3-player 'Mineral Rights' setting.

    Reference: Krishna (2009), Example 6.1
    """
    return (2 * valuation) / (2 + valuation)


def bne_2p_affiliated_values(
        valuation: torch.Tensor, player_position: int = 0) -> torch.Tensor:
    """Symmetric BNE in the 2p affiliated values model.

    Reference: Krishna (2009), Example 6.2"""
    return (2/3) * valuation