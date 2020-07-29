"""This module implements metrics that may be interesting."""

import torch
from bnelearn.strategy import Strategy
from bnelearn.mechanism import Mechanism
from bnelearn.environment import Environment
from bnelearn.bidder import Bidder
from tqdm import tqdm


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

def norm_strategy_and_actions(strategy, actions, valuations: torch.Tensor, p: float=2) -> float:
    """Calculates the norm as above, but given one action vector and one strategy.
    The valuations must match the given actions.

    This helper function is useful when recalculating an action vector is prohibitive and it should be reused.

    """
    s_actions = strategy.play(valuations)

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


def ex_interim_util_loss_old(env: Environment, bid_profile: torch.Tensor,
                         agent: Bidder, agent_valuation: torch.Tensor,
                         grid: torch.Tensor, half_precision = False):
    """
    Estimates a bidder's util_loss/utility loss in the current bid_profile, i.e. the potential benefit of deviating from
    the current strategy, evaluated at each point of the agent_valuations.
        At each of these valuation points, the best response utility is approximated via the best utility achieved on the grid.
    Input:
        mechanism
        bid_profile: (batch_size x n_player x n_items)
        agent: specifies the agent for whom the regret is to be evaluated
        agent_valuation: (batch_size x n_items)
        grid: tensor of bids which are to evaluated
        half_precision: bool
    Output:
        util_loss: (batch_size)
        valuations: (batch_size x n_items)

    Useful: To get the memory used by a tensor (in MB): (tensor.element_size() * tensor.nelement())/(1024*1024)
    Remarks:
        - Only applicable to independent valuations, because we take the cross product over valuations.
        - Only for risk neutral bidders
    TODO:
        - Add check for risk neutral bidders.
        - Move grid_creation out of util_loss for Nils special cases
    """
    mechanism = env.mechanism
    player_position = agent.player_position
    agent_bid_actual = bid_profile[:,player_position,:]

    ## Use smaller dtypes to save memory
    if half_precision:
        bid_profile = bid_profile.half()
        agent_valuation = agent_valuation.half()
        agent_bid_actual = agent_bid_actual.half()
        grid = grid.half()
    bid_profile_origin = bid_profile

    batch_size, n_players, n_items = bid_profile.shape # pylint: disable=unused-variable
    grid_size, _ = grid.shape

    ### Evaluate alternative bids on grid
    bid_profile = _create_grid_bid_profiles(player_position, grid, bid_profile_origin) #(grid_size x n_players x n_items)

    ## Calculate allocation and payments for alternative bids given opponents bids
    allocation, payments = mechanism.play(bid_profile)

    # we only need the specific player's allocation and can get rid of the rest.
    a_i = allocation[:, player_position, :].type(torch.bool).view(grid_size * batch_size, n_items)
    p_i = payments[:, player_position].view(grid_size * batch_size) #(grid x batch)

    del allocation, payments, bid_profile
    if torch.cuda.is_available():
        torch.cuda.empty_cache() #TODO, later: find out if this actually does anything here.

    # Calculate realized valuations given allocation
    try:
        # valuation is batch x items
        v_i = agent_valuation.repeat(1, grid_size * batch_size).view(batch_size, grid_size * batch_size, n_items)
        #v_i = env.draw_conditional_valuations_(player_position, agent_valuation)
        #v_i = v_i.repeat(1, grid_size).view(batch_size, grid_size * batch_size, n_items)

        p_i = p_i.repeat(batch_size, 1)

        ## Calculate utilities
        u_i_alternative = agent.get_counterfactual_utility(a_i, p_i, v_i)
        u_i_alternative = u_i_alternative.view(batch_size, grid_size, batch_size) #(batch x grid x batch)

        # avg per bid
        u_i_alternative = torch.mean(u_i_alternative, 2) #(batch x grid)
        # max per valuations
        u_i_alternative, _ = torch.max(u_i_alternative, 1) #batch

    except RuntimeError as err:
        print("Failed computing util_loss as batch. Trying sequential valuations computation. Decrease dimensions to fix. Error:\n {0}".format(err))
        try:

            # valuations sequential
            u_i_alternative = torch.zeros(batch_size, device=p_i.device)
            for idx in tqdm(range(batch_size)):
                v_i = agent_valuation[idx].repeat(1, grid_size * batch_size).view(batch_size * grid_size, n_items)
                ## Calculate utilities
                u_i_alternative_v = agent.get_counterfactual_utility(a_i, p_i, v_i)
                u_i_alternative_v = u_i_alternative_v.view(grid_size, batch_size) #(grid x batch)
                # avg per bid
                u_i_alternative_v = torch.mean(u_i_alternative_v, 1)
                # max per valuations
                u_i_alternative[idx], _ = torch.max(u_i_alternative_v, 0)

                # clean up
                del u_i_alternative_v

        except RuntimeError as err:
            print("Failed computing util_loss as batch with sequential valuations. Decrease dimensions to fix. Error:\n {0}".format(err))
            u_i_alternative = torch.ones(batch_size, device = p_i.device) * -9999999

    # Clean up storage
    del v_i, a_i, p_i
    torch.cuda.empty_cache()

    ### Evaluate actual bids
    bid_profile = _create_grid_bid_profiles(player_position, agent_bid_actual, bid_profile_origin)

    ## Calculate allocation and payments for actual bids given opponents bids
    allocation, payments = mechanism.play(bid_profile)
    a_i = allocation[:, player_position, :].type(torch.bool).view(batch_size * batch_size, n_items)
    p_i = payments[:, player_position].view(batch_size * batch_size)

    ## Calculate realized valuations given allocation
    v_i = agent_valuation.repeat(1, batch_size).view(batch_size * batch_size, n_items)

    ## Calculate utilities
    u_i_actual = agent.get_counterfactual_utility(a_i, p_i, v_i).view(batch_size, batch_size)
    u_i_actual = torch.mean(u_i_actual, 1)

    ## average and max regret over all valuations
    util_loss = (u_i_alternative - u_i_actual).relu().clone().detach().requires_grad_(False)
    return util_loss, agent_valuation


def ex_interim_util_loss(env: Environment, player_position: int,
                         batch_size: int, grid_size: int):
    """
    Calculate
        \max_{v_i \in V_i} \max_{b_i^* \in A_i}
            E_{v_{-i}|v_i} [u(v_i, b_i^*, b_{-i}(v_{-i})) - u(v_i, b_i, b_{-i}(v_{-i}))]

    TODO: when to use repeat when repeat_interleave?
    TODO: mean and max over right axises!
    """

    """0. SET UP"""
    mechanism = env.mechanism

    agent = env.agents[player_position]
    valuation = agent.valuations[:batch_size, ...]
    action_actual = agent.get_action()[:batch_size, ...].detach().clone()
    action_alternative = agent.get_valuation_grid(grid_size, True)

    # grid is not always the requested size
    grid_size = action_alternative.shape[0]

    n_items = valuation.shape[-1]

    # dict with valuations of (batch_size * batch_size, n_items) for each opponent
    conditional_valuation_profile = env.draw_conditional_valuations_(player_position, valuation)

    """1. CALCULATE UTILITY WITH ACTUAL STRATEGY"""
    # actual bid profile and actual utility
    #   1st dim: different agent valuations
    #   2nd dim: differnet opponent valuations (-> different actions)
    action_profile_actual = torch.zeros(batch_size * batch_size, env.n_players, n_items,
                                        dtype=valuation.dtype, device=mechanism.device)
    for a in env.agents:
        if a.player_position == player_position:
            action_profile_actual[:, player_position, :] = action_actual.repeat_interleave(batch_size, 0)
        else:
            action_profile_actual[:, a.player_position, :] = \
                a.strategy.play(
                    conditional_valuation_profile[a.player_position] # (batch_size * batch_size, n_items)
                )

    # TODO Nils: why do we have NaNs?
    action_profile_actual[action_profile_actual != action_profile_actual] = 0

    allocation_actual, payment_actual = mechanism.play(action_profile_actual)
    allocation_actual = allocation_actual[:, player_position, :].type(torch.bool) # (batch_size, batch_size, n_items)
    payment_actual = payment_actual[:, player_position] # (batch_size, batch_size)
    utility_actual = agent.get_counterfactual_utility(
        allocation_actual, payment_actual, valuation.repeat_interleave(batch_size, 0)
    ).view(batch_size, batch_size)

    utility_actual = torch.mean(utility_actual, axis=1) # expectation over opponents

    """2. CALCULATE UTILITY WITH ALTERNATIVE ACTIONS ON GRID"""
    # alternative bid profile and alternative utility
    #   1st dim: different agent valuations (-> actions should be different for given valuation, )
    #   2nd dim: different agent actions
    #   3rd dim: differnet opponent valuations (-> different actions)
    action_profile_alternative = torch.zeros(batch_size * grid_size * batch_size, env.n_players, n_items,
                                             dtype=valuation.dtype, device=mechanism.device)
    for a in env.agents:
        if a.player_position == player_position:
            action_profile_alternative[:, player_position, :] = \
                action_alternative.repeat_interleave(batch_size, 0).repeat(batch_size, 1)
        else:
            action_profile_alternative[:, a.player_position, :] = \
                a.strategy.play(conditional_valuation_profile[a.player_position]).repeat(grid_size, 1)

    # TODO Nils: why do we have NaNs?
    action_profile_alternative[action_profile_alternative != action_profile_alternative] = 0

    allocation_alternative, payment_alternative = mechanism.play(action_profile_alternative)
    allocation_alternative = allocation_alternative[:, player_position, :].type(torch.bool) # (batch_size * grid_size * batch_size, n_items)
    payment_alternative = payment_alternative[:, player_position] # (batch_size * grid_size * batch_size)
    utility_alternative = agent.get_counterfactual_utility(
        allocation_alternative, payment_alternative,
         valuation.repeat_interleave(batch_size * grid_size, 0)
    ).view(batch_size, grid_size, batch_size)

    utility_alternative = torch.mean(utility_alternative, axis=2) # expectation over opponents
    utility_alternative = torch.max(utility_alternative, axis=1)[0] # maximum expected utility over alternative actions

    """3. COMPARE UTILITY"""
    utility_loss = utility_alternative - utility_actual

    return utility_loss.clone().detach().requires_grad_(False)
