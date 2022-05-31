"""This module contains implementations of known Bayes-Nash equilibrium bid 
    functions in several specific settings.
    Whenever possible, the bid functions are fully vectorized.

    NOTE: This module contains a mix of direct implementations of bid functions,
    and factory methods that create and return the bid function as a callable object.
"""
from typing import Callable, List, Union

import torch
import numpy as np
from scipy import interpolate, optimize
from bnelearn.util.integration import cumulatively_integrate
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
    if not isinstance(valuation, torch.Tensor):
        # For float and numpy --> convert to tensor (relevant for plotting)
        valuation = torch.tensor(valuation, dtype=torch.float)

    # shorthand notation for F^(n-1)
    cdf_powered = lambda v: torch.pow(prior_cdf(v), n_players - 1)

    # calculate numerator integrals
    numerator = cumulatively_integrate(cdf_powered, upper_bounds = valuation)

    return valuation - numerator / cdf_powered(valuation)
    
def bne_tpsb_ipv_symmetric_generic_prior_risk_neutral(valuation: torch.tensor, n_players: int):
    return (valuation * (n_players - 1))/(n_players-2)

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

def loss_equilibrium(valuation: torch.Tensor, lamb: float, **kwargs):
    return -1/(2*(lamb - 1) * valuation - 2 * lamb) * (valuation ** 2)

def bne_crowdsourcing(valuation: torch.Tensor, v1: float = 1/2, v2: float = 1/2, **kwargs):
    return torch.relu(8*valuation*(3*v1 -2) + 4*valuation.log()*(3-5*v1) + 8*(2-3*v1))

    #beta = lambda c, v1: 8*c*(3*v1 -2) + 4*np.log(c)*(3-5*v1) + 8*(2-3*v1)

    #return (-8+8*valuation-8* valuation.log()) * v1 + (16-16*valuation+12*valuation.log())*v2

    

def bne_crowdsourcing_valuations(valuation: torch.Tensor, v1: float = 1, v2 = 0, N: int = 0, player_position=0, **kwargs):
    #return torch.relu(0.63 * (valuation ** 2 - 2/3 * valuation ** 3) - 0.37 * valuation ** 2)
    #return torch.relu(v1 * 2 * ((valuation ** 2)/2 - (valuation ** 3)/3) + v2 * 2 * ((valuation ** 2)/2 - (2*valuation**3)/3))

    # old for n = 3 and general m
    # a = lambda m, v: 2/(1-m) * ((m**3)/(6*(1-m)) + (v ** 3)/(3*(1-m)) - (m*v**2)/(2*(1-m)))
    # b = lambda m, v: 2/(1-m) * (-(m**3)/(3*(1-m)) - (m ** 2)/2 - (2 * v **3)/(3*(1-m)) + (m * v ** 2)/(1-m) + (v ** 2)/2)

    # return v1 * a(m, valuation) + v2 * b(m, valuation)
    a = lambda v, N: (N-1)/N * v ** N
    b = lambda v, N: (N-1) * (((N-2) * v ** (N-1))/(N-1) + (v**N)/N - v**N)
    return torch.relu(v1 * a(valuation, N) + v2 * b(valuation, N))

def bne_all_pay(valuation: torch.Tensor, n: int = 2, **kwargs) -> torch.Tensor:
    return (n-1) * (valuation ** n)/n

def bne_all_pay_cost(valuation: torch.Tensor, n: int = 2, **kwargs):
    return torch.relu(-torch.log(valuation))

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
    b1[u_lo_cut:] = np.array([optimize.broyden1(lambda x, v=v: inverse_bid_player_1(x) - v, v)
                              for v in v1[u_lo_cut:]])
    b2 = np.array([optimize.broyden1(lambda x, v=v: inverse_bid_player_2(x) - v, v)
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

###############################################################################
#### Combinatorial Auctions
###############################################################################

## NOTE: In LLG, LLGFull and LLLLGG, the equilibria are implemented directly
## within the experiment classes.


###############################################################################
### Multi-Unit Equilibria                              ###
###############################################################################

def multiunit_bne_factory(setting, payment_rule) -> Callable or None:
    """
    Factory method that returns the known BNE strategy function for the standard multi-unit auctions
    (split-award is NOT one of the) as callable if available and None otherwise.
    """

    if payment_rule in ('vcg', 'vickrey'):
        return truthful_bid

    if setting.correlation_types not in [None, 'independent'] or \
            setting.risk != 1:
        # Aside from VCG, equilibria are only known for independent priors and
        # quasilinear/risk-neutral utilities.
        return None

    if payment_rule in ('first_price', 'discriminatory'):
        if setting.n_units == 2 and setting.n_players == 2:
            if not setting.constant_marginal_values:
                print('BNE is only approximated roughly!')
                return _bne_multiunit_discriminatory_2x2
            else:
                return _bne_multiunit_discriminatory_2x2_cmv(setting.common_prior)

    if payment_rule == 'uniform':
        if setting.n_units == 2 and setting.n_players == 2:
            return _bne_multiunit_uniform_2x2()
        if (setting.n_units == 3 and setting.n_players == 2
                and setting.item_interest_limit == 2):
            return _bne_multiunit_uniform_3x2_limit2

    return None

def _bne_multiunit_discriminatory_2x2(valuation, player_position=None):
    """BNE strategy in the multi-unit discriminatory price auction with
    2 players and 2 units"""

    def b_approx(v, s, t):
        b = torch.clone(v)
        lin_e = np.array([[1, 1, 1], [2 * t, 1, 0], [t ** 2, t, 1]])
        lin_s = np.array([0.47, s / t, s])
        x = np.linalg.solve(lin_e, lin_s)
        b[v < t] *= s / t
        b[v >= t] = x[0] * b[v >= t] ** 2 + x[1] * b[v >= t] + x[2]
        return b

    b1 = lambda v: b_approx(v, s=0.42, t=0.90)
    b2 = lambda v: b_approx(v, s=0.35, t=0.55)

    opt_bid = torch.clone(valuation)
    opt_bid[:, 0] = b1(opt_bid[:, 0])
    opt_bid[:, 1] = b2(opt_bid[:, 1])
    opt_bid = opt_bid.sort(dim=1, descending=True)[0]
    return opt_bid

def _bne_multiunit_discriminatory_2x2_cmv(prior: torch.distributions.Distribution) -> Callable:
    """BNE strategy in the multi-unit discriminatory price auction with two
    players, two units, and with constant marginal valuations.
    """
    # Simplify computation for uniform prior case
    if isinstance(prior, torch.distributions.uniform.Uniform):
        def _optimal_bid(valuation, player_position=None):
            return (valuation - prior.low) / 2.0 + prior.low
        return _optimal_bid

    raise NotImplementedError("BNE not implemented for this prior.")

def _bne_multiunit_uniform_2x2():
    """ Returns two BNE strategies List[callable] in the multi-unit uniform price auction
        with 2 players and 2 units.
    """

    def opt_bid_1(valuation, player_position=None):
        opt_bid = torch.clone(valuation)
        opt_bid[:,1] = 0
        return opt_bid

    def opt_bid_2(valuation, player_position=None):
        opt_bid = torch.ones_like(valuation)
        opt_bid[:,1] = 0
        return opt_bid

    return [opt_bid_1, opt_bid_2]

def _bne_multiunit_uniform_3x2_limit2(valuation, player_position=None):
    """ BNE strategy in the multi-unit uniform price auction with 3 units and
        2 palyers that are both only interested in winning 2 units
    """
    opt_bid = torch.clone(valuation)
    opt_bid[:, 1] = opt_bid[:, 1] ** 2
    opt_bid[:, 2] = 0
    return opt_bid

def bne_splitaward_2x2_1_factory(experiment_config, payoff_dominant: bool=True):
    """Factory method returning the BNE pooling equilibrium in the split-award
    auction with 2 players and 2 lots (as in Anton and Yao, 1992).
    
    Actually, this is a continuum of BNEs of which this function returns the
    upper bound (payoff dominat BNE) and the one at the lower bound.

    Returns:
        optimal_bid (callable): The equilibrium bid function.
    """
    efficiency_parameter = experiment_config.efficiency_parameter
    u_lo = experiment_config.u_lo[0]
    u_hi = experiment_config.u_hi[0]

    # clip bids at top
    clip_cutoff = 4 * u_hi

    def optimal_bid(valuation, player_position=None):

        device = valuation.device
        dist = torch.distributions.Uniform(torch.tensor(u_lo, device=device),
                                            torch.tensor(u_hi, device=device))
        value_cdf = dist.cdf

        sigma_bounds = torch.ones((valuation.shape[0], 2), device=device)
        sigma_bounds[:, 1] = efficiency_parameter * u_hi
        sigma_bounds[:, 0] = (1 - efficiency_parameter) * u_lo

        p_sigma = (1 - efficiency_parameter) * u_lo  # highest possible p_sigma

        def G(theta):
            return p_sigma + (p_sigma - u_hi * efficiency_parameter * value_cdf(theta)) \
                    / (1 - value_cdf(theta))

        wta_bounds = 2 * sigma_bounds
        wta_bounds[:, 0] = G(valuation[:, 1])

        action_idx = 1 if payoff_dominant else 0
        bid = torch.cat(
            (sigma_bounds[:, action_idx].view(-1, 1),
            wta_bounds[:, action_idx].view(-1, 1)),
            axis=1
        )
        bid[bid > clip_cutoff] = clip_cutoff
        return bid

    return optimal_bid

def bne_splitaward_2x2_2_factory(experiment_config):
    """Factory method returning the BNE WTA equilibrium in the split-award 
    auction with 2 players and 2 lots (as in Anton and Yao Proposition 4, 1992).

    Returns:
        optimal_bid (callable): The equilibrium bid function.
    """
    efficiency_parameter = experiment_config.efficiency_parameter
    u_lo = experiment_config.u_lo
    u_hi = experiment_config.u_hi
    n_players = experiment_config.n_players

    def optimal_bid(valuation, player_position=None):
        device = valuation.device
        valuation_cdf = torch.distributions.Uniform(
            torch.tensor(u_lo[0], device=device),
            torch.tensor(u_hi[0], device=device)).cdf

        integrand = lambda x: torch.pow(1 - valuation_cdf(x), n_players - 1)
        integral = - cumulatively_integrate(integrand, 
                                            upper_bounds = valuation[:, [1]],
                                            lower_bound=u_hi[0] - 1e-4)

        opt_bid_100 = valuation[:, [1]] + integral \
            / torch.pow(1 - valuation_cdf(valuation[:, [1]]), n_players - 1)

        opt_bid = torch.cat(
            [opt_bid_100 - efficiency_parameter * u_lo[0], opt_bid_100],
            axis=1)
        opt_bid[opt_bid < 0] = 0
        return opt_bid

    return optimal_bid

###############################################################################
