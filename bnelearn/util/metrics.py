"""This module implements metrics that may be interesting."""

import torch
from bnelearn.strategy import Strategy
from bnelearn.mechanism import Mechanism
from bnelearn.environment import Environment, AuctionEnvironment
from bnelearn.bidder import Bidder
from tqdm import tqdm
import warnings


def norm_actions(b1: torch.Tensor, b2: torch.Tensor, p: float = 2) -> float:
    """
    Calculates the approximate "mean" Lp-norm between two action vectors.
    (\\sum_i=1^n(1/n * |b1 - b2|^p))^(1/p)

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

    The function Lp norm is given by (\\int_V |s1(v) - s2(v)|^p dv)^(1/p).
    With Monte-Carlo Integration this is approximated by
     (|V|/n * \\sum_i^n(|s1(v) - s2(v)|^p) )^(1/p)  where |V| is the volume of the set V.

    Here, we ignore the volume. This gives us the RMSE for L2, supremum for Linfty, etc.
    """
    b1 = strategy1.play(valuations)
    b2 = strategy2.play(valuations)

    return norm_actions(b1, b2, p)

def norm_strategy_and_actions(strategy, actions, valuations: torch.Tensor, p: float=2, componentwise=False) -> float:
    """Calculates the norm as above, but given one action vector and one strategy.
    The valuations must match the given actions.

    This helper function is useful when recalculating an action vector is prohibitive and it should be reused.

    Input:
        strategy: Strategy
        actions: torch.Tensor
        valuations: torch.Tensor
        p: float=2
        componentwise: bool=False, only returns smallest norm of all output dimensions if true
    Returns:
        norm, float
    """
    s_actions = strategy.play(valuations)

    if componentwise:
        component_norm = [norm_actions(s_actions[..., d], actions[..., d], p)
                          for d in range(actions.shape[-1])]
        return min(component_norm)
    else:
        return norm_actions(s_actions, actions, p)


def _create_grid_bid_profiles(bidder_position: int, grid: torch.tensor, bid_profile: torch.tensor):
    """Given an original bid profile, creates a tensor of (grid_size * batch_size) batches of bid profiles,
       where for each original batch, the player's bid is replaced by each possible bid in the grid.

    Input:
        bidder_position: int - the player who's bids will be replaced
        grid: FloatTensor (grid_size x n_items): tensor of possible bids to be evaluated
        bid_profile: FloatTensor (batch_size x n_players x n_items)
    Returns:
        bid_profile: FloatTensor (grid_size*batch_size x n_players x n_items)
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

def ex_post_util_loss(mechanism: Mechanism, bid_profile: torch.Tensor, bidder: Bidder,
                      grid: torch.Tensor, half_precision = False, player_position: int = None):
    """
    # TODO: do we really need this or can we delete it in general?
    # If we decide to keep it, check implementation in detail! (Removing many many todos in the body)

    Estimates a bidder's ex post util_loss in the current bid_profile vs a potential grid,
        i.e. the potential benefit of having deviated from the current strategy, as:
        util_loss = max(0, BR(v_i, b_-i) - u_i(b_i, b_-i))
    Input:
        mechanism
        bid_profile: (batch_size x n_player x n_items)
        bidder: a Bidder (used to retrieve valuations and utilities)
        grid:
            option 1: 1d tensor with length grid_size
                todo for n_items > 1, all grid_size**n_items combination will be used. Should be
                replaced by e.g. torch.meshgrid
            option 2: tensor with shape (grid_size, n_items)
        player_position (optional): specific position in which the player will be evaluated
            (defaults to player_position of bidder)
        half_precision: (optional, bool) Whether to use half precision tensors. default: false
    Output:
        util_loss (batch_size)

    Useful: To get the memory used by a tensor (in MB): (tensor.element_size() * tensor.nelement())/(1024*1024)
    """

    player_position = bidder.player_position
    agent_valuation = bidder.valuations # batch_size x n_items

    ## Use smaller dtypes to save memory
    if half_precision:
        bid_profile = bid_profile.half()
        agent_valuation = agent_valuation.half()
        grid = grid.half()

    #Generalize these dimensions
    batch_size, n_players, n_items = bid_profile.shape # pylint: disable=unused-variable
    grid_size = grid.shape[0] #update this
    # Create multidimensional bid tensor if required
    if n_items == 1:
        grid = grid.view(grid_size, 1).to(bid_profile.device)
    elif n_items >= 2:
        if len(grid.shape) == 1:
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

    counterfactual_valuations = bidder.valuations.repeat(grid_size, 1) # grid*batch x n_items
    utility_grid = bidder.get_counterfactual_utility(a_i, p_i, counterfactual_valuations).view(grid_size, batch_size)
    best_response_utility, best_response = utility_grid.max(0)

    ## Evaluate actual bids
    allocation, payments = mechanism.play(bid_profile)
    a_i = allocation[:,player_position,:] # batch x n_items
    p_i = payments[:,player_position] # batch

    actual_utility = bidder.get_utility(a_i, p_i)

    return (best_response_utility - actual_utility).relu() # set 0 if actual bid is best (no difference in limit, but might be valuated if grid too sparse)


def ex_interim_util_loss(env: AuctionEnvironment, player_position: int,
                         batch_size: int, grid_size: int):
    """
    Estimates a bidder's utility loss in the current state of the environment, i.e. the
    potential benefit of deviating from the current strategy, evaluated at each point of
    the agent_valuations. therfore, we calculate
        $$\max_{v_i \in V_i} \max_{b_i^* \in A_i}
            E_{v_{-i}|v_i} [u(v_i, b_i^*, b_{-i}(v_{-i})) - u(v_i, b_i, b_{-i}(v_{-i}))]$$

    We're conditoning on the agent's observation at `player_position`. That means, types and
    observations of other palyers as well as its own type have to be conditioned. As it's
    conditioned on the observation, the agent's action stays the same.

    Input
    -----
        env: bnelearn.Environment.
        player_position: int, position of the player in the environment.
        batch_size: int, specifing the sample size for agent itself and other agents.
        grid_size: int, stating the number of alternative actions sampled via
            env.agents[player_position].get_valuation_grid(grid_size, True).
    Output
    ------
        utility_loss: torch.Tensor of shape (batch_size) describing the expected
            possible utiliy increase.

    Remark
    ------
        - Relies on the following subprocedures: `agent.get_action()`,
          `agent.get_counterfactual_utility()`, `env.draw_conditionals()`, and
          `agent.get_valuation_grid()`. Therefore, these methods need to be provided for the
          specific setting.
    """
    # pylint: disable=pointless-string-statement

    """0. SET UP"""

    mechanism = env.mechanism
    device = mechanism.device

    agent: Bidder = env.agents[player_position]

    assert batch_size <= agent.batch_size, "invalid batch size!"

    observation = agent.valuations[:batch_size, ...].detach()
    action_actual = agent.get_action()[:batch_size, ...].detach()
    n_items = observation.shape[-1]

    agent_batch_size = observation.shape[0]
    opponent_batch_size = batch_size

    # draw opponent observations conditional on `agent`'s observation:
    # dict with valuations of (batch_size * batch_size, n_items) for each opponent

    conditionals = env.draw_conditionals(
        player_position, observation, opponent_batch_size
    )

    # conditioning type on signal - not needed when they're equal
    if hasattr(agent, '_unkown_valuation'):
        agent_type = conditionals[agent.player_position]
    else:
        agent_type = observation \
            .repeat_interleave(opponent_batch_size, 0)

    """1. CALCULATE EXPECTED UTILITY FOR EACH SAMPLE WITH ACTUAL STRATEGY"""
    # actual bid profile and actual utility
    #   1st dim / batch_size: different agent valuations
    #   2nd dim / opponent_batch_size: different opponent valuations (-> different actions)
    action_profile_actual = torch.zeros(
        agent_batch_size * opponent_batch_size, env.n_players, n_items,
        dtype=action_actual.dtype, device=device
    )
    for a in env.agents:
        if a.player_position == player_position:
            action_profile_actual[:, player_position, :] = \
                action_actual.repeat_interleave(opponent_batch_size, 0) \
                    .view(agent_batch_size * opponent_batch_size, n_items)
        else:
            action_profile_actual[:, a.player_position, :] = \
                a.strategy.play(conditionals[a.player_position]) \
                    .detach().requires_grad_(False).clone() #TODO Stefan: do we need clone?

    allocation_actual, payment_actual = mechanism.play(action_profile_actual)

    # TODO: until here we can just use the real valuations in the player objects to simplify the code.

    allocation_actual = allocation_actual[:, player_position, :].type(torch.bool) \
        .view(agent_batch_size * opponent_batch_size, n_items)
    payment_actual = payment_actual[:, player_position] \
        .view(agent_batch_size * opponent_batch_size)
    utility_actual = agent.get_counterfactual_utility(
        allocation_actual, payment_actual, agent_type
    ).view(agent_batch_size, opponent_batch_size)

    # expectation over opponent batches
    utility_actual = torch.mean(utility_actual, axis=1) #dim: batch_size

    """2. CALCULATE EXPECTED UTILITY FOR EACH SAMPLE WITH ALTERNATIVE ACTIONS ON GRID"""
    action_alternative = agent.get_valuation_grid(grid_size, True)
    grid_size = action_alternative.shape[0] # grid is not always the requested size

    # calc adpative (own) batch size `mini_batch_size` based on memory estimate
    # TODO Stefan: this only runs when using GPU, will fail if cpu is active!
    assert torch.cuda.is_initialized(), "util loss implementation requires cuda GPU!"
    from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo
    nvmlInit()
    nvml_device_handle = nvmlDeviceGetHandleByIndex(torch.cuda.device(device).idx)
    mem_info = nvmlDeviceGetMemoryInfo(nvml_device_handle)
    free_mem_bytes = mem_info.free

    element_bytes = action_profile_actual.element_size()
    required_tensor_size = grid_size * opponent_batch_size * env.n_players * n_items

    mini_batch_size = batch_size
    while free_mem_bytes < mini_batch_size * element_bytes * required_tensor_size and mini_batch_size > 1:
        mini_batch_size = int(mini_batch_size / 2)

    if mini_batch_size < batch_size:
        warnings.warn("Sequential computation of utility loss: mini batches of size {}."
                      .format(mini_batch_size))
        custom_range = tqdm(range(0, batch_size, mini_batch_size))
    else:
        custom_range = range(0, batch_size, mini_batch_size)

    utility_alternative = torch.zeros_like(utility_actual)

    for b in custom_range:

        conditionals = env.draw_conditionals(
            player_position, observation[b:b+mini_batch_size, :], opponent_batch_size
        )

        if hasattr(agent, '_unkown_valuation'):
            agent_type = conditionals[agent.player_position]
        else:
            agent_type = observation[b:b+mini_batch_size, :] \
                .repeat_interleave(opponent_batch_size, 0)

        # alternative bid profile and alternative utility
        #   1st dim: different agent valuations (-> actions should be different for given vals)
        #   2nd dim: different agent actions
        #   3rd dim: differnet opponent valuations (-> different actions)
        action_profile_alternative = torch.zeros(
            mini_batch_size * grid_size * opponent_batch_size, env.n_players, n_items,
            dtype=action_actual.dtype, device=device
        )
        for a in env.agents:
            if a.player_position == player_position:
                action_profile_alternative[:, player_position, :] = \
                    action_alternative \
                        .repeat(mini_batch_size, 1) \
                        .view(mini_batch_size, grid_size, n_items) \
                        .repeat_interleave(opponent_batch_size, 1) \
                        .view(mini_batch_size * grid_size * opponent_batch_size, n_items)
            else:
                # TODO: can we get rid of clone?
                action_profile_alternative[:, a.player_position, :] = \
                    a.strategy.play(conditionals[a.player_position]) \
                        .detach().requires_grad_(False).clone() \
                        .view(mini_batch_size, opponent_batch_size, n_items) \
                        .repeat(1, grid_size, 1) \
                        .view(mini_batch_size * grid_size * opponent_batch_size, n_items)

        allocation_alternative, payment_alternative = mechanism.play(action_profile_alternative)
        allocation_alternative = allocation_alternative[:, player_position, :].type(torch.bool) \
            .view(mini_batch_size * grid_size * opponent_batch_size, n_items)
        payment_alternative = payment_alternative[:, player_position] \
            .view(mini_batch_size * grid_size * opponent_batch_size)
        utility_alternative_batch = agent.get_counterfactual_utility(
            allocation_alternative, payment_alternative,
            agent_type \
                .view(mini_batch_size, opponent_batch_size, n_items) \
                .repeat(1, grid_size, 1) \
                .view(mini_batch_size * grid_size * opponent_batch_size, n_items)
        ).view(mini_batch_size, grid_size, opponent_batch_size)

        # expectation over opponent_batch
        utility_alternative_batch = torch.mean(utility_alternative_batch, axis=2)

        # maximum expected utility over grid of alternative actions
        utility_alternative[b:b+mini_batch_size] = torch.max(utility_alternative_batch, axis=1)[0]

    """3. COMPARE UTILITY"""
    # we don't accept a negative loss when the gird is not precise enough: set to 0
    utility_loss = (utility_alternative - utility_actual).relu()

    return utility_loss
