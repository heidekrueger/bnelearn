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

###########################################################################
#### Combinatorial Auctions
###########################################################################

## NOTE: In LLG and LLLLGG, the equilibria are implemented directly within
## the experiment classes.



###############################################################################
### Multi-Unit Equilibria                              ###
###############################################################################

def _multiunit_bne(setting, payment_rule) -> Callable or None:
    """
    Method that returns the known BNE strategy for the standard multi-unit auctions
    (split-award is NOT one of the) as callable if available and None otherwise.
    """

    if  float(setting.risk) != 1:
        return None  # Only know BNE for risk neutral bidders

    if payment_rule in ('vcg', 'vickrey'):
        def truthful(valuation, player_position=None):  # pylint: disable=unused-argument
            return valuation
        return truthful

    if (setting.correlation_types not in [None, 'independent'] or
            setting.risk != 1):
        return None

    if payment_rule in ('first_price', 'discriminatory'):
        if setting.n_units == 2 and setting.n_players == 2:
            if not setting.constant_marginal_values:
                print('BNE is only approximated roughly!')
                return _optimal_bid_multidiscriminatory2x2
            else:
                # TODO get valuation_cdf from experiment_config
                # return _optimal_bid_multidiscriminatory2x2CMV(valuation_cdf)
                return None

    if payment_rule == 'uniform':
        if setting.n_units == 2 and setting.n_players == 2:
            return _optimal_bid_multiuniform2x2()
        if (setting.n_units == 3 and setting.n_players == 2
                and setting.item_interest_limit == 2):
            return _optimal_bid_multiuniform3x2limit2

    return None

def _optimal_bid_multidiscriminatory2x2(valuation, player_position=None):
    """BNE strategy in the multi-unit discriminatory price auction 2 players and 2 units"""

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

def _optimal_bid_multidiscriminatory2x2CMV(valuation_cdf):
    """ BNE strategy in the multi-unit discriminatory price auction 2 players and 2 units
        with constant marginal valuations
    """
    n_players = 2

    if not valuation_cdf(torch.tensor(0.0)).device.type == 'cpu':
        raise ValueError("valuation_cdf is required to return CPU-tensors rather than gpu tensors, " + \
            "otherwise we will encounter errors when using numerical integration via scipy together with " + \
            "torch.multiprocessing. For now, please provide a cpu-version of the prior-cdf.")

    if isinstance(valuation_cdf, torch.distributions.uniform.Uniform):
        def _optimal_bid(valuation, player_position=None):
            return valuation / 2

    elif isinstance(valuation_cdf, torch.distributions.normal.Normal):

        def muda_tb_cmv_bne(
                value_pdf: callable,
                value_cdf: callable = None,
                lower_bound: int = 0,
                epsabs=1e-3
        ):
            if value_cdf is None:
                def _value_cdf(x):
                    return integrate.quad(value_pdf, lower_bound, x, epsabs=epsabs)[0]

                value_cdf = _value_cdf

            def inner(s, x):
                return integrate.quad(lambda t: value_pdf(t) / value_cdf(t),
                                      s, x, epsabs=epsabs)[0]

            def outer(x):
                return integrate.quad(lambda s: np.exp(-inner(s, x)),
                                      lower_bound, x, epsabs=epsabs)[0]

            def bidding(x):
                if not hasattr(x, '__iter__'):
                    return x - outer(x)
                else:
                    return np.array([xi - outer(xi) for xi in x])

            return bidding

        bidding = muda_tb_cmv_bne(lambda x: torch.exp(valuation_cdf.log_prob(x)).cpu().numpy(),
                                  lambda x: valuation_cdf.cdf(x).cpu().numpy())

        def _optimal_bid(valuation, player_position=None):
            opt_bid = np.zeros_like(valuation.cpu().numpy())
            for agent in range(n_players):
                opt_bid[agent] = bidding(valuation[agent, :])
            return torch.tensor(opt_bid)

    return _optimal_bid

def _optimal_bid_multiuniform2x2():
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

def _optimal_bid_multiuniform3x2limit2(valuation, player_position=None):
    """ BNE strategy in the multi-unit uniform price auction with 3 units and
        2 palyers that are both only interested in 2 units
    """
    opt_bid = torch.clone(valuation)
    opt_bid[:, 1] = opt_bid[:, 1] ** 2
    opt_bid[:, 2] = 0
    return opt_bid

def _optimal_bid_splitaward2x2_1(experiment_config, payoff_dominant: bool=True):
    """BNE pooling equilibrium in the split-award auction with 2 players and 2
    lots (as in Anton and Yao, 1992). Actually, this is a continuum of BNEs of
    which this function returns the upper bound (payoff dominat BNE) and the
    one at the lower bound.

        Returns callable.
    """
    efficiency_parameter = experiment_config.efficiency_parameter
    u_lo = experiment_config.u_lo[0]
    u_hi = experiment_config.u_hi[0]

    # cut off bids at top
    _CUT_OFF = 4 * u_hi

    def _optimal_bid(valuation, player_position=None):

        device = valuation.device
        dist = torch.distributions.Uniform(torch.tensor(u_lo, device=device),
                                            torch.tensor(u_hi, device=device))
        value_cdf = dist.cdf

        sigma_bounds = torch.ones((valuation.shape[0], 2), device=device)
        sigma_bounds[:, 1] = efficiency_parameter * u_hi
        sigma_bounds[:, 0] = (1 - efficiency_parameter) * u_lo

        _p_sigma = (1 - efficiency_parameter) * u_lo  # highest possible p_sigma

        def G(theta):
            return _p_sigma + (_p_sigma - u_hi * efficiency_parameter * value_cdf(theta)) \
                    / (1 - value_cdf(theta))

        wta_bounds = 2 * sigma_bounds
        wta_bounds[:, 0] = G(valuation[:, 1])

        action_idx = 1 if payoff_dominant else 0
        bid = torch.cat(
            (sigma_bounds[:, action_idx].view(-1, 1),
            wta_bounds[:, action_idx].view(-1, 1)),
            axis=1
        )
        bid[bid > _CUT_OFF] = _CUT_OFF
        return bid

    return _optimal_bid

def _optimal_bid_splitaward2x2_2(experiment_config):
    """BNE WTA equilibrium in the split-award auction with 2 players and
        2 lots (as in Anton and Yao Proposition 4, 1992).

        Returns callable.
    """
    print('Warning: BNE is approximated on CPU.')

    efficiency_parameter = experiment_config.efficiency_parameter
    u_lo = experiment_config.u_lo[0]
    u_hi = experiment_config.u_hi[0]
    n_players = experiment_config.n_players

    def value_cdf(value):
        value = np.array(value)
        result = (value - u_lo) / (u_hi - u_lo)
        return result.clip(0, 1)

    # CONSTANTS
    opt_bid_batch_size = 2**12
    _EPS = 1e-4

    opt_bid = np.zeros((opt_bid_batch_size, 2))

    # do one-time approximation via integration
    val_lin = np.linspace(u_lo, u_hi - _EPS, opt_bid_batch_size)

    def integral(theta):
        return np.array(
            [integrate.quad(
                lambda x: (1 - value_cdf(x))**(n_players - 1), v, u_hi,
                epsabs=_EPS
            )[0] for v in theta]
        )

    def opt_bid_100(theta):
        return theta + (integral(theta) / ((1 - value_cdf(theta))**(n_players - 1)))

    opt_bid[:, 1] = opt_bid_100(val_lin)
    opt_bid[:, 0] = opt_bid[:, 1] - efficiency_parameter * u_lo  # or more

    opt_bid_function = [
        interpolate.interp1d(val_lin, opt_bid[:, 1], fill_value='extrapolate'),
        interpolate.interp1d(val_lin, opt_bid[:, 0], fill_value='extrapolate')
    ]

    # use interpolation of opt_bid done on first batch
    def _optimal_bid(valuation, player_position=None):
        bid = torch.empty((valuation.shape[0], 2), device=valuation.device,
                          dtype=valuation.dtype)
        val = valuation[:, 1].cpu().numpy()
        bid[:, 0] = torch.tensor(opt_bid_function[1](val), device=valuation.device)
        bid[:, 1] = torch.tensor(opt_bid_function[0](val), device=valuation.device)
        bid[bid < 0] = 0
        return bid

    return _optimal_bid

###############################################################################
