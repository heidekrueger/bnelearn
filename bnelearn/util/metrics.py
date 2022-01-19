"""This module implements metrics that may be interesting."""

from typing import Tuple
from tqdm import tqdm
import torch
import matplotlib.pyplot as plt

from bnelearn.bidder import Bidder
from bnelearn.environment import AuctionEnvironment
from bnelearn.mechanism import Mechanism
from bnelearn.strategy import Strategy
from bnelearn.util.tensor_util import apply_with_dynamic_mini_batching

## defines a mapping of internal metrics and their desired tensorboard output tags
MAPPING_METRICS_TAGS = {
    'utilities':            'market/utilities',
    'efficiency':           'market/efficiency',
    'revenue':              'market/revenue',
    'PoA':                  'market/PoA',
    'update_norm':          'learner_info/update_norm',
    'gradient_norm':        'learner_info/gradient_norm',
    'util_loss_ex_ante':    'eval/util_loss_ex_ante',
    'util_loss_ex_interim': 'eval/util_loss_ex_interim',
    'estimated_relative_ex_ante_util_loss': 'eval/estimated_relative_ex_ante_util_loss',
    'utility_vs_bne':       'eval_vs_bne/utility_vs_bne',
    'epsilon_relative':     'eval_vs_bne/epsilon_relative',
    'epsilon_absolute':     'eval_vs_bne/epsilon_absolute',
    'L_2':                  'eval_vs_bne/L_2',
    'L_inf':                'eval_vs_bne/L_inf',
    'overhead_hours':       'meta/overhead_hours',

    # won't actually be logged
    'prev_params':          'learner_info/prev_params'
}

# aliases of tf tags for plotting/publications
#pylint: disable=anomalous-backslash-in-string
ALIASES_LATEX = {
    'market/efficiency':           'efficiency $\mathcal{E}$',
    'market/revenue':              'revenue $\mathcal{R}$',
    'market/utilities':            'utility $u$',

    'eval/util_loss_ex_ante':    '$\hat \ell$',
    'eval/util_loss_ex_interim': '$\hat \epsilon$',
    'eval/estimated_relative_ex_ante_util_loss': 'relative utility loss $\hat{\mathcal{L}}$',

    'eval_vs_bne/L_2':                  '$L_2$',
    'eval_vs_bne/L_inf':                '$L_\infty$',
    'eval_vs_bne/epsilon_absolute':     '$\epsilon_\text{abs}$',
    'eval_vs_bne/epsilon_relative':     '$\mathcal{L}$',
    'eval_vs_bne/utility_vs_bne':       '$\hat u(\beta_i, \beta^*_{-i})$',

    'meta/overhead_hours':       '$T$',

    'learning_info/update_norm':          '$|\Delta \theta|$'
}


def norm_actions(b1: torch.Tensor, b2: torch.Tensor, p: float = 2) -> float:
    r"""Calculates the approximate "mean" Lp-norm between two action vectors.

    .. math::
        \sum_i=1^n(1/n * |b1 - b2|^p)^{1/p}

    If p = Infty, this evaluates to the supremum.
    """
    assert b1.shape == b2.shape

    if p == float('Inf'):
        return (b1 - b2).abs().max()

    # finite p
    n = float(b1.shape[0])

    # calc. norm & detach for disregarding any gradient info
    return (torch.dist(b1, b2, p=p)*(1./n)**(1/p)).detach()

def norm_strategies(strategy1: Strategy, strategy2: Strategy, valuations: torch.Tensor, p: float=2) -> float:
    r"""Calculates the approximate "mean" Lp-norm between two strategies
    approximated via Monte-Carlo integration on a sample of valuations that
    have been drawn according to the prior.

    The function Lp norm is given by

    .. math::
        (\int_V |s1(v) - s2(v)|^p dv)^{1/p}.

    With Monte-Carlo Integration this is approximated by

    .. math::
        (|V|/n * \sum_i^n(|s1(v) - s2(v)|^p) )^{1/p}

    where |V| is the volume of the set V. Here, we ignore the volume. This
    gives us the RMSE for L2, supremum for Linfty, etc.
    """
    b1 = strategy1.play(valuations)
    b2 = strategy2.play(valuations)

    return norm_actions(b1, b2, p)

def norm_strategy_and_actions(strategy, actions, valuations: torch.Tensor, p: float=2, componentwise=False) -> torch.Tensor:
    """Calculates the norm as above, but given one action vector and one strategy.
    The valuations must match the given actions.

    This helper function is useful when recalculating an action vector is prohibitive and it should be reused.

    Args:
        strategy: Strategy
        actions: torch.Tensor
        valuations: torch.Tensor
        p: float=2
        componentwise: bool=False, only returns smallest norm of all output
            dimensions if true
    Returns:
        norm: (scalar Tensor)
    """
    s_actions = strategy.play(valuations)

    if componentwise:
        component_norms = [norm_actions(s_actions[..., d], actions[..., d], p)
                          for d in range(actions.shape[-1])]
        # select that component with the smallest norm
        return min(component_norms)
    else:
        return norm_actions(s_actions, actions, p)

def _create_grid_bid_profiles(bidder_position: int, grid: torch.Tensor, bid_profile: torch.Tensor):
    """Given an original bid profile, creates a tensor of (grid_size *
    batch_size) batches of bid profiles, where for each original batch, the
    player's bid is replaced by each possible bid in the grid.

    Args:
        bidder_position: int - the player who's bids will be replaced
        grid: FloatTensor (grid_size x n_items): tensor of possible bids to
            be evaluated
        bid_profile: FloatTensor (batch_size x n_players x n_items)
    Returns:
        bid_profile: FloatTensor (grid_size*batch_size x n_players x
            n_items)
    """
    # version with size checks: (slower)
    # batch_size, _, n_items = bid_profile.shape #batch x player x item
    # n_candidates, n_items = candidate_bids.shape # candidates x item
    #assert n_items == n_items2, "input tensors don't match"

    batch_size, _, _ = bid_profile.shape #batch x player x item
    n_candidates, _ = grid.shape # candidates x item

    bid_profile = bid_profile.repeat(n_candidates, 1, 1)
    bid_profile[:, bidder_position, :] = grid.repeat_interleave(repeats = batch_size, dim=0)

    return bid_profile #bid_eval_size*batch, 1,n_items

def ex_post_util_loss(mechanism: Mechanism, bidder_valuations: torch.Tensor, bid_profile: torch.Tensor, bidder: Bidder,
                      grid: torch.Tensor, half_precision = False):
    r"""
    # TODO: do we really need this or can we delete it in general?
    # If we decide to keep it, check implementation in detail! (Removing many many todos in the body)

    Estimates a bidder's ex post util_loss in the current bid_profile vs a potential grid,
        i.e. the potential benefit of having deviated from the current strategy, as:

    .. math::
        util\_loss = max(0, BR(v_i, b_-i) - u_i(b_i, b_-i))

    Args:
        mechanism
        player_valuations: the valuations of the player that is to be evaluated
        bid_profile: (batch_size x n_player x n_items)
        bidder: a Bidder (used to retrieve valuations and utilities)
        grid:
            option 1: 1d tensor with length grid_size
                todo for :math:`n\_items > 1`, all :math:`grid\_size^{n\_items}` combination will be used. Should be
                replaced by e.g. torch.meshgrid
            option 2: tensor with shape (grid_size, n_items)
        player_position (optional): specific position in which the player will be evaluated
            (defaults to player_position of bidder)
        half_precision: (optional, bool) Whether to use half precision tensors. default: false

    Returns:
        util_loss (batch_size)

    Useful: To get the memory used by a tensor (in MB): :math:`(tensor.element\_size() * tensor.nelement())/(1024*1024)`
    """

    player_position = bidder.player_position

    ## Use smaller dtypes to save memory
    if half_precision:
        bid_profile = bid_profile.half()
        bidder_valuations = bidder_valuations.half()
        grid = grid.half()

    #Generalize these dimensions
    batch_size, n_players, n_items = bid_profile.shape # pylint: disable=unused-variable
    grid_size = grid.shape[0] #update this
    # Create multidimensional bid tensor if required
    if n_items == 1:
        grid = grid.view(grid_size, 1).to(bid_profile.device)
    elif n_items >= 2 and len(grid.shape) == 1:
        grid = torch.combinations(grid, r=n_items, with_replacement=True).to(bid_profile.device) #grid_size**n_items x n_items
        # Stefan: this only works if both bids are over the same action space (what if one of these is the bid for a bundle?)
    grid_size, _ = grid.shape # this _new_ grid size refers to all combinations, whereas the previous one was 1D only

    ### Evaluate alternative bids on grid
    grid_bid_profile = _create_grid_bid_profiles(player_position, grid, bid_profile) # (grid_size*batch_size) x n_players x n_items
    ## Calculate allocation and payments for alternative bids given opponents bids
    allocation, payments = mechanism.play(grid_bid_profile)

    # we only need the specific player's allocation and can get rid of the rest.
    a_i = allocation[:,player_position,:]
    p_i = payments[:,player_position] # 1D tensor of length (grid * batch)

    utility_grid = bidder.get_utility(
        a_i, p_i, bidder_valuations.repeat_interleave(grid_size, dim=0)
        ).view(grid_size, batch_size)
    best_response_utility, _ = utility_grid.max(0)

    ## Evaluate actual bids
    allocation, payments = mechanism.play(bid_profile)
    a_i = allocation[:,player_position,:]
    p_i = payments[:,player_position]

    actual_utility = bidder.get_utility(a_i, p_i, bidder_valuations)

    return (best_response_utility - actual_utility).relu() # set 0 if actual bid is best (no difference in limit, but might be valuated if grid too sparse)

def ex_interim_util_loss(env: AuctionEnvironment, player_position: int,
                         agent_observations: torch.Tensor,
                         grid_size: int,
                         opponent_batch_size: int = None,
                         grid_best_response: bool = False, mute: bool = False):
    #pylint: disable = anomalous-backslash-in-string
    """Estimates a bidder's utility loss in the current state of the
    environment, i.e. the potential benefit of deviating from the current
    strategy, evaluated at each point of the agent_valuations. Therefore, we
    calculate

    .. math::
        \max_{v_i \in V_i} \max_{b_i^* \in A_i} + E_{v_{-i}|v_i} [u(v_i, b_i^*, b_{-i}(v_{-i})) - u(v_i, b_i, b_{-i}(v_{-i}))]

    We're conditioning on the agent's observation at `player_position`. That
    means, types and observations of other palyers as well as its own type have
    to be conditioned. As it's     conditioned on the observation, the agent's
    action stays the same.

    Args:
        env: bnelearn.Environment.
        player_position: int, position of the player in the environment.
        grid_size: int, stating the number of alternative actions sampled via
            env.agents[player_position].get_valuation_grid(grid_size, True).
        opponent_batch_size: int, specifing the sample size for opponents.
        grid_best_response: bool, whether or not the BRs live on the grid or
            possibly come from the actual actions (in case no better response
            was found on grid).
        mute: bool, mute stdout.

    Returns:
        utility_loss (torch.Tensor, shape: [batch_size]):  the computed
            approximate utility loss for for each input observation.
        best_response (torch.Tensor, shape: [batch_size, action_size]):
            the best response found for each input observation (This is
            either a grid point, or the actual action according to the player's
            strategy.)

    Remarks:
        Relies on availability of `draw_conditional_profiles` and
        `generate_valuation_grid` in the `env`'s ValuationObservationSampler.
    """

    mechanism = env.mechanism
    device = mechanism.device
    agent: Bidder = env.agents[player_position]
    # ensure we are not propagating any gradients (may cause memory leaks)
    agent_observations = agent_observations.detach().clone()

    agent_batch_size, _ = agent_observations.shape
    opponent_batch_size = opponent_batch_size or agent_batch_size

    ####### get actual utility #############################
    agent_action_actual = agent.get_action(agent_observations)
    utility_actual = ex_interim_utility(
        env, player_position, agent_observations, agent_action_actual,
        opponent_batch_size, device)

    ####### get best responses over grid of alternative actions #######
    action_alternatives = env.sampler.generate_action_grid(
        player_position=player_position,
        minimum_number_of_points=grid_size,
        dtype=agent_action_actual.dtype, device=agent_action_actual.device
    )
    action_size = action_alternatives.shape[-1]

    get_br_utily_and_index = lambda obs: _get_best_responses_among_alternatives(
        env, player_position, obs, action_alternatives, opponent_batch_size)
    br_utility, br_indices = apply_with_dynamic_mini_batching(
        function=get_br_utily_and_index,
        args=agent_observations, mute=mute)

    ##### calculate the loss and return best responses ###########
    utility_loss = (br_utility - utility_actual).relu_()

    # BR only on grid
    if grid_best_response:
        br_actions = action_alternatives[br_indices]
    else:
        actual_was_best = (utility_loss == 0).unsqueeze_(1).repeat(1, action_size)
        br_actions = actual_was_best * agent_action_actual + \
                 actual_was_best.logical_not() * action_alternatives[br_indices]

    return (utility_loss, br_actions)

def _get_best_responses_among_alternatives(
        env: AuctionEnvironment, player_position: int,
        agent_observations: torch.Tensor, action_alternatives: torch.Tensor,
        opponent_batch_size: int) -> Tuple[torch.Tensor, torch.IntTensor]:
    """For a batch of observations for the given player, calculates the
    ex-interim best response from a fixed set of alternatives.

    Returns:
        br_utility (torch.FloatTensor of size [agent_batch_size])
        br_indices (torch.IntTensor of size [agent_batch_size]): the indices of
            the best actions in action_alternatives
    """

    grid_size, action_size = action_alternatives.shape
    agent_batch_size, _ = agent_observations.shape
    device = env.mechanism.device

    ## grid_size x agent_batch_size x action_size
    grid_actions = action_alternatives \
        .view(grid_size, 1, action_size) \
        .repeat([1, agent_batch_size, 1])
    ## grid_size x agent_batch_size x observation_size
    grid_observations = agent_observations.repeat([grid_size, 1, 1])

    # grid_size x agent_batch_size
    grid_utilities = ex_interim_utility(
        env, player_position, grid_observations,
        grid_actions, opponent_batch_size, device
        )

    # for each agent_observation, find the best response
    # each have shape: [agent_batch_size]
    br_utility, br_indices = grid_utilities.max(dim=0)
    return br_utility, br_indices

def ex_interim_utility(
        env: AuctionEnvironment, player_position: int,
        agent_observations: torch.Tensor, agent_actions: torch.Tensor,
        opponent_batch_size: int, device) -> torch.Tensor:
    """
    Calculates the ex-interim utility of a given agent in the environment,
    given (batches of) their observations and actions.

    Can handle multiple batch dimensions for the agent.

    Args:
        env (AuctionEnvironment): The environment from which conditional type
            profiles and opponent actions will be sampled.
        player_position (int): the position of the agent to be evaluated
        agent_observations (Tensor of dim (*agent_batch_sizes x observation_size))
        agent_actions      (Tensor of dim (*agent_batch_sizes x action_size))
        opponent_batch_size (int): how many conditional valuations and opponent
            observations to sample for each agent_batch entry. The expected
            ex-interim utility will then be approximated by the sample mean
            over the opponent_batch_size dimension.
        device (device):    The output device.

    Returns:
        utility: (Tensor of dim (*agent_batch_sizes)): the resulting empirical
            ex-interim utilities.
    """
    mechanism = env.mechanism
    agent = env.agents[player_position]

    *batch_dims, _ = range(agent_actions.dim())
    *agent_batch_sizes, action_size = agent_actions.shape
    assert agent_observations.shape[:len(batch_dims)] == torch.Size(agent_batch_sizes), \
        """observations and actions must have the same batch sizes!"""
    action_dtype = agent_actions.dtype
    # draw conditional observations conditioned on `agent`'s observation:
    # co has dimension (*agent_batches , opponent_batch, n_players, observation_size)
    # each agent_observations is repeated opponent_batch_size times
    cv, co = env.draw_conditionals(
        player_position, agent_observations, opponent_batch_size, device
        )

    action_profile_actual = torch.zeros(
        *agent_batch_sizes, opponent_batch_size, env.n_players, action_size,
        dtype=action_dtype, device=device
        )

    action_profile_actual[...,:,player_position,:] = \
        agent_actions \
            .view(*agent_batch_sizes, 1, action_size) \
            .repeat(*([1]*len(agent_batch_sizes)), opponent_batch_size, 1)

    for a in env.agents:
        if a.player_position != player_position:
            action_profile_actual[..., a.player_position, :] = \
                a.strategy.play(co[..., a.player_position, :])

    # shapes: allocations: *agent_batches x opponent_batch x n_players x n_items
    #         payments:    *agent_batches x opponent_batch x n_players
    allocations, payments = mechanism.play(action_profile_actual)

    agent_allocations = allocations[..., player_position, :].type(torch.bool)
    agent_payments = payments[..., player_position]
    agent_valuations = cv[..., player_position, :]
    # shape of utility: *agent_batch_sizes x opponent_batch_size
    utility = agent.get_utility(
        agent_allocations, agent_payments, agent_valuations
        )

    # expectation over opponent batches
    utility = torch.mean(utility, axis=-1) #dim: agent_batch_size
    return utility

def verify_epsilon_bne(exp: 'Experiment', grid_size: int,
                       opponent_batch_size: int) -> torch.Tensor:
    """Calculate the epsilon for which we can guarantee an epsilon-BNE under
    the assumptions of Bosshard et al. 2017.
    """
    env = exp.env
    device = env._observations.device
    dtype = env._observations.dtype
    valuation_size = env._observations.shape[-1]
    highest_epsilon = 0.0

    # maximum over all bidders
    for player_positions in exp._model2bidder:
        player_position = player_positions[0]
        agent: Bidder = env.agents[player_position]

        # maximum over observation/valuation grid
        # (this should also include the boundaries, s.t. all parts of the step-
        # wise function are covered)
        for vertices in tqdm(
            exp.sampler.generate_cell_partition(player_position, grid_size, device=device),
            total=grid_size
            ):

            action_alternatives = env.sampler.generate_action_grid(
                player_position, minimum_number_of_points=grid_size,
                dtype=dtype, device=device
                )
            util_best_response_cell = -float('inf')
            for valuation_vertex in vertices:

                # caluclate BR utiltiy at all vertices and take the maximum
                util_best_response_vertex = _get_best_responses_among_alternatives(
                    env, player_position, valuation_vertex, action_alternatives,
                    opponent_batch_size
                    )[0]
                util_best_response_cell = max(util_best_response_cell, util_best_response_vertex)

                # interim utility at lower cell vertex
                action = agent.get_action(vertices[0])
                utility_actual = ex_interim_utility(
                    env, player_position, valuation_vertex, action,
                    opponent_batch_size, device
                    ).item()

                highest_epsilon = max(highest_epsilon, util_best_response_cell - utility_actual)

    return highest_epsilon

# def ex_interim_utility_bound(env: AuctionEnvironment, player_position: int,
#                              agent_valuation: torch.Tensor, grid_size: int,
#                              opponent_batch_size: int) -> torch.Tensor:
#     """Calculate an upper bound for the utility and specific actions.
#     According to Bosshard et al. (2020), under some assumptions (see Prop. 1),
#     the expected utility can be bounded based on estimates.
#     """
#     device = env.mechanism.device
#     agent: Bidder = env.agents[player_position]
#     assert agent_valuation.shape[0] == 1, 'Can only check one valuation at a time'

#     valuation_size = agent_valuation.shape[-1]
#     bid_size = agent.bid_size

#     grid = env.sampler.generate_action_grid(
#         player_position=player_position, minimum_number_of_points=grid_size,
#         device=device
#         )
#     grid_size = grid.shape[0]  # grid might have different sample size for higher dims

#     bid_profile = torch.empty(opponent_batch_size, grid_size, len(env.agents), bid_size,
#                               dtype=agent_valuation.dtype, device=device)
#     for opponent in env.agents:
#         if opponent.player_position != player_position:
#             bid_profile[:, :, opponent.player_position, :] = \
#                 opponent.get_action(
#                     env.sampler.draw_conditional_profiles(
#                         player_position, agent_valuation, opponent_batch_size, device
#                         )[0][:, :, opponent.player_position, :]
#                     ) \
#                     .view(opponent_batch_size, 1, bid_size) \
#                     .repeat(1, grid_size, 1)
#              # repeat opponents' actions to compete against all grid-actions of agent

#     bid_profile[:, :, player_position, :] = grid \
#         .view(1, grid_size, bid_size) \
#         .repeat(opponent_batch_size, 1, 1)
#     allocations, payments = env.mechanism.play(bid_profile)

#     # welfare for each grid action, each averaged over all opponents' actions
#     welfare = agent \
#         .get_welfare(
#             allocations[:, :, player_position, :],
#             agent_valuation \
#                 .view(1, 1, valuation_size) \
#                 .repeat(opponent_batch_size, grid_size, 1)
#             ) \
#         .view(opponent_batch_size, grid_size) \
#         .mean(axis=0)

#     # payments for each grid action averaged over opponents
#     payments = payments[:, :, [player_position]] \
#         .view(opponent_batch_size, grid_size) \
#         .mean(axis=0)

#     # compare welfare of upper bound on grid with payments on lower bound on grid
#     # TODO support more than one action dim!
#     if grid.shape[-1] > 1:
#         raise NotImplementedError()

#     utility_bound = welfare[1:] - payments[:-1]

#     # invert welfare and payments for reverse bidders
#     # TODO should be handled in bidder class perhaps
#     if isinstance(agent, ReverseBidder):
#         utility_bound *= -1
#
#    return torch.max(utility_bound)
