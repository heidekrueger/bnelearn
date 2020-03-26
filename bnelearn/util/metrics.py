"""This module implements metrics that may be interesting."""

import torch
from bnelearn.strategy import Strategy
from bnelearn.mechanism import Mechanism
from bnelearn.bidder import Bidder


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

def ex_post_regret(mechanism: Mechanism, bid_profile: torch.Tensor, bidder: Bidder,
           grid: torch.Tensor, half_precision = False, player_position: int = None):
    """
    Estimates a bidder's ex post regret in the current bid_profile vs a potential grid,
        i.e. the potential benefit of having deviated from the current strategy, as:
        regret = max(0, BR(v_i, b_-i) - u_i(b_i, b_-i))
    Input:
        mechanism
        bid_profile: (batch_size x n_player x n_items)
        bidder: a Bidder (used to retrieve valuations and utilities)
        grid: #TODO: currently (1d with length grid_size #Currently, for n_items == 2, all grid_size**2 combination will be used. Should be replaced by e.g. torch.meshgrid
        player_position (optional): specific position in which the player will be evaluated (defaults to player_position of bidder)
        half_precision: (optional, bool) Whether to use half precision tensors. default: false
    Output:
        regret (batch_size)
    #TODO: move grid_creation out of regret

    Useful: To get the memory used by a tensor (in MB): (tensor.element_size() * tensor.nelement())/(1024*1024)
    """

    player_position = bidder.player_position
    agent_valuation = bidder.valuations # batch_size x n_items

    ## Use smaller dtypes to save memory
    if half_precision:
        bid_profile = bid_profile.half()
        agent_valuation = agent_valuation.half()
        agent_bid_actual = agent_bid_actual.half()
        grid = grid.half()

    # TODO: Generalize these dimensions
    batch_size, n_players, n_items = bid_profile.shape # pylint: disable=unused-variable
    grid_size = grid.shape[0] #TODO: update this
    # Create multidimensional bid tensor if required
    if n_items == 1:
        grid = grid.view(grid_size, 1).to(bid_profile.device)
    elif n_items == 2:
        grid = torch.combinations(grid, with_replacement=True).to(bid_profile.device) #grid_size**2 x 2
            #TODO Stefan: this only works if both bids are over the same action space (what if one of these is the bid for a bundle?)
    elif n_items > 2:
        raise NotImplementedError("Regret for >2 items not implemented yet!")
    grid_size, _ = grid.shape #TODO this _new_ grid size refers to all combinations, whereas the previous one was 1D only


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


def ex_interim_regret(mechanism: Mechanism, bid_profile: torch.Tensor,
                      player_position: int, agent_valuation: torch.Tensor,
                      grid: torch.Tensor, half_precision = False):
    #TODO: 1. Implement individual evaluation batch und bid size -> large batch for training, smaller for eval
    #TODO: 2. Implement logging for evaluations ins tensor and for printing
    #TODO: 3. Implement printing plotting of evaluation
    """
    Estimates a bidder's regret in the current bid_profile, i.e. the potential benefit of deviating from the current strategy, as:
        regret(v_i) = Max_(b_i)[ E_(b_(-i))[u(v_i,b_i,b_(-i))] ] #TODO Stefan: shouldn't there be a  - u(v_i, b) here?
        regret_max = Max_(v_i)[ regret(v_i) ]
        regret_expected = E_(v_i)[ regret(v_i) ]
    Input:
        mechanism
        bid_profile: (batch_size x n_player x n_items)
        player_position: specifies the agent for whom the regret is to be evaluated
        agent_valuation: (batch_size x n_items)
        grid: #TODO: currently (1d with length grid_size #Currently, for n_items == 2, all grid_size**2 combination will be used. Should be replaced by e.g. torch.meshgrid
    Output:
        regret (grid_size) (?) #TODO Stefan: If bid is multidimensional, shouldn't this be bid_size ** n_items? 
    #TODO: move grid_creation out of regret

    TODO: Only applicable to independent valuations. (Stefan: because we take the cross product over valuations)
    TODO: Only for risk neutral bidders. Add check.
    Useful: To get the memory used by a tensor (in MB): (tensor.element_size() * tensor.nelement())/(1024*1024)
    """
    agent_bid_actual = bid_profile[:,player_position,:]
    ## Use smaller dtypes to save memory
    if half_precision:
        bid_profile = bid_profile.half()
        agent_valuation = agent_valuation.half()
        agent_bid_actual = agent_bid_actual.half()
        grid = grid.half()
    bid_profile_origin = bid_profile

    # TODO: Generalize these dimensions
    batch_size, n_players, n_items = bid_profile.shape # pylint: disable=unused-variable
    grid_size = grid.shape[0] #TODO: update this
    # Create multidimensional bid tensor if required
    if n_items == 1:
        grid = grid.view(grid_size, 1).to(bid_profile.device)
    elif n_items == 2:
        grid = torch.combinations(grid, with_replacement=True).to(bid_profile.device) #grid_size**2 x 2
            #TODO Stefan: this only works if both bids are over the same action space (what if one of these is the bid for a bundle?)
    elif n_items > 2:
        raise NotImplementedError("Regret for >2 items not implemented yet!")
    grid_size, _ = grid.shape #TODO this _new_ grid size refers to all combinations, whereas the previous one was 1D only


    ### Evaluate alternative bids on grid
    bid_profile = _create_grid_bid_profiles(player_position, grid, bid_profile_origin) # grid_size x n_players x n_items 

    ## Calculate allocation and payments for alternative bids given opponents bids
    allocation, payments = mechanism.play(bid_profile)

    # we only need the specific player's allocation and can get rid of the rest.

    a_i = allocation[:,player_position,:].view(grid_size, batch_size, n_items).type(torch.bool) #TODO Stefan: bool will not work for multi-unit auctions, there we need int!
    p_i = payments[:,player_position].view(grid_size, batch_size) #grid * batch

    #return ((a_i * agent_valuation).sum(2) - p_i).max(0)[0]

    del allocation, payments, bid_profile
    if torch.cuda.is_available():
        torch.cuda.empty_cache() #TODO: find out if this actually does anything here.
    # Calculate realized valuations given allocation
    try:
        # valuation is batch x items
        v_i = agent_valuation.repeat(1,grid_size * batch_size).view(batch_size, grid_size, batch_size, n_items)
        # Semantic of the next line: (calculated directly, less readible but faster and robust to different types (e.g. bools/int))
        #v_i = torch.einsum('hijk,ijk->hij', v_i, a_i) 
        v_i = v_i.mul_(a_i).sum(3) # allocated value. batch x bid_eval x batch (why 2 batch_sizes?)
        ## Calculate utilities
        u_i_alternative = v_i - p_i.repeat(batch_size,1,1) #(batch x grid x batch)
        # avg per bid
        u_i_alternative = torch.mean(u_i_alternative,2) #batch x grid
        # max per valuations
        u_i_alternative, _ = torch.max(u_i_alternative,1) #batch
    except RuntimeError as err:
        print("Failed computing regret as batch. Trying sequential valuations computation. Decrease dimensions to fix. Error:\n {0}".format(err))
        try:
            # valuations sequential
            u_i_alternative = torch.zeros(batch_size, device = p_i.device)
            for v in range(batch_size):
                v_i = agent_valuation[v].repeat(1,grid_size * batch_size).view(grid_size, batch_size, n_items)
                #for bid in agent bid
                v_i = torch.einsum('ijk,ijk->ijk', v_i, a_i).sum(2)
                ## Calculate utilities
                u_i_alternative_v = v_i - p_i
                # avg per bid
                u_i_alternative_v = torch.mean(u_i_alternative_v,1)
                # max per valuations
                u_i_alternative[v], _ = torch.max(u_i_alternative_v,0)
                tmp = int(batch_size/100)
                if v % tmp == 0:
                    print('{} %'.format(v*100/batch_size))

                # clean up
                del u_i_alternative_v
        except RuntimeError as err:
            print("Failed computing regret as batch with sequential valuations. Decrease dimensions to fix. Error:\n {0}".format(err))
            u_i_alternative = torch.ones(batch_size, device = p_i.device) * -9999999

    # Clean up storage
    del v_i, a_i, p_i
    torch.cuda.empty_cache()

    ### Evaluate actual bids
    ## Merge actual bids into opponnents bids (bid_no_i)
    bid_profile = _create_grid_bid_profiles(player_position, agent_bid_actual, bid_profile_origin)

    ## Calculate allocation and payments for actual bids given opponents bids
    allocation, payments = mechanism.play(bid_profile)
    a_i = allocation[:,player_position,:].view(batch_size, batch_size, n_items)
    p_i = payments[:,player_position].view(batch_size, batch_size, 1).sum(2)

    ## Calculate realized valuations given allocation
    v_i = agent_valuation.view(batch_size,1,n_items).repeat(1, batch_size, 1)
    v_i = torch.einsum('ijk,ijk->ijk', v_i, a_i).sum(2)

    ## Calculate utilities
    u_i_actual = v_i - p_i
    # avg per bid and valuation
    u_i_actual = torch.mean(u_i_actual,1)
    ## average and max regret over all valuations
    regret = (u_i_alternative - u_i_actual).relu()

    # Explicitaly cleanup TODO:?
    return regret
