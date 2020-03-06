"""This module implements metrics that may be interesting."""

import torch
from bnelearn.strategy import Strategy
from bnelearn.mechanism import Mechanism


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

def _create_bid_profile(agent_position: int, player_bids: torch.tensor, original_bid_profile: torch.tensor):
    """Creates a bid profile to be used in regret estimation below.
    """
    batch_size, n_players, n_items = original_bid_profile.shape
    bid_eval_size, _ = player_bids.shape #Stefan: I assume second dim is also n_items? what exactly is the meaning of bid_eval_size?
    ## Merge bid_i into opponnents bids (bid_no_i)
    # bids_(-i)
    bid_no_i_left = original_bid_profile[:, [i for i in range(n_players) if i<agent_position], :] # batch x (some_players) x items
    bid_no_i_right = original_bid_profile[:, [i for i in range(n_players) if i>agent_position], :] # batch x (some_players) x items
    # bids_i x batch_size
    bid_i = player_bids.repeat(1,batch_size).view(bid_eval_size*batch_size,1,n_items)
    # bid_size x bids_(-i)
    bid_no_i_left = bid_no_i_left.repeat(bid_eval_size, 1, 1) # Stefan: why 1 when above its n_itmes?
    bid_no_i_right = bid_no_i_right.repeat(bid_eval_size, 1, 1) #Stefan: same
    #TODO: In place combination or splitting and sequential
    return torch.cat([bid_no_i_left,bid_i,bid_no_i_right],1) #Stefan: total: bid_eval_size*batch, 1,n_items?

def regret(mechanism: Mechanism, bid_profile: torch.Tensor, agent_position: int, agent_valuation: torch.Tensor,
           agent_bid_actual: torch.Tensor, agent_bid_eval: torch.Tensor, half_precision = False):
    #TODO: 1. Implement individual evaluation batch und bid size -> large batch for training, smaller for eval
    #TODO: 2. Implement logging for evaluations ins tensor and for printing
    #TODO: 3. Implement printing plotting of evaluation
    """
    Estimates the potential benefit of deviating from the current strategy, as:
        regret(v_i) = Max_(b_i)[ E_(b_(-i))[u(v_i,b_i,b_(-i))] ] #TODO Stefan: shouldn't there be a  - u(v_i, b) here?
        regret_max = Max_(v_i)[ regret(v_i) ]
        regret_expected = E_(v_i)[ regret(v_i) ]
    The current bidder is always considered with index = 0 #TODO Stefan: Why? what about asymmetric environments?
    Input:
        mechanism
        bid_profile: (batch_size x n_player x n_items)
        agent_position: specifies the agent for whom the regret is to be evaluated
        agent_valuation: (batch_size x n_items)
        agent_bid_actual: (batch_size x n_items) #TODO Stefan: isn't this in bid_profile? which one?
        agent_bid_eval: (bid_size x n_items) #TODO Stefan: defines the grid of possible actions to be evaluated for the agent
    Output:
        regret (bid_size) (?) #TODO Stefan: If bid is multidimensional, shouldn't this be bid_size ** n_items?

    TODO: Only applicable to independent valuations. Add check. #TODO Stefan: why? where is this required?
    TODO: Only for risk neutral bidders. Add check.
    Useful: To get the memory used by a tensor (in MB): (tensor.element_size() * tensor.nelement())/(1024*1024)
    """

    # TODO: Generalize these dimensions
    batch_size, n_players, n_items = bid_profile.shape # pylint: disable=unused-variable
    # Create multidimensional bid tensor if required
    if n_items == 1:
        agent_bid_eval = agent_bid_eval.view(agent_bid_eval.shape[0], 1).to(bid_profile.device)
    elif n_items == 2:
        agent_bid_eval = torch.combinations(agent_bid_eval, with_replacement=True).to(bid_profile.device) #bid_size**2 x 2 (?)
            #TODO Stefan: this only works if both bids are over the same action space right? (what if one of these is the bid for a bundle?)
    elif n_items > 2:
        raise NotImplementedError("Regret for >2 items not implemented yet!")
    bid_eval_size, _ = agent_bid_eval.shape

    # TODO: Stefan: Why not keep bid_size and batch_size as separate dims? I assume because mechanism won't allow it?

    ## Use smaller dtypes to save memory
    if half_precision:
        bid_profile = bid_profile.half()
        agent_valuation = agent_valuation.half()
        agent_bid_actual = agent_bid_actual.half()
        agent_bid_eval = agent_bid_eval.half()
    bid_profile_origin = bid_profile

    ### Evaluate alternative bids
    ## Merge alternative bids into opponnents bids (bid_no_i)
    bid_profile = _create_bid_profile(agent_position, agent_bid_eval, bid_profile_origin) # bid_eval_size x n_player x n_item 

    ## Calculate allocation and payments for alternative bids given opponents bids
    allocation, payments = mechanism.play(bid_profile)
    a_i = allocation[:,agent_position,:].view(bid_eval_size, batch_size, n_items).type(torch.bool)
    p_i = payments[:,agent_position].view(bid_eval_size, batch_size, 1).sum(2)

    del allocation, payments, bid_profile
    torch.cuda.empty_cache()
    # Calculate realized valuations given allocation
    try:
        v_i = agent_valuation.repeat(1,bid_eval_size * batch_size).view(batch_size, bid_eval_size, batch_size, n_items)
        v_i = torch.einsum('hijk,ijk->hijk', v_i, a_i).sum(3) # allocated value. batch x bid_eval x batch (why 2 batch_sizes?)
        ## Calculate utilities
        u_i_alternative = v_i - p_i.repeat(batch_size,1,1)
        # avg per bid
        u_i_alternative = torch.mean(u_i_alternative,2)
        # max per valuations
        u_i_alternative, _ = torch.max(u_i_alternative,1)
    except RuntimeError as err:
        print("Failed computing regret as batch. Trying sequential valuations computation. Decrease dimensions to fix. Error:\n {0}".format(err))
        try:
            # valuations sequential
            u_i_alternative = torch.zeros(batch_size, device = p_i.device)
            for v in range(batch_size):
                v_i = agent_valuation[v].repeat(1,bid_eval_size * batch_size).view(bid_eval_size, batch_size, n_items)
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
    del v_i
    torch.cuda.empty_cache()

    ### Evaluate actual bids
    ## Merge actual bids into opponnents bids (bid_no_i)
    bid_profile = _create_bid_profile(agent_position, agent_bid_actual, bid_profile_origin)

    ## Calculate allocation and payments for actual bids given opponents bids
    allocation, payments = mechanism.play(bid_profile)
    a_i = allocation[:,agent_position,:].view(batch_size, batch_size, n_items)
    p_i = payments[:,agent_position].view(batch_size, batch_size, 1).sum(2)

    ## Calculate realized valuations given allocation
    v_i = agent_valuation.view(batch_size,1,n_items).repeat(1, batch_size, 1)
    v_i = torch.einsum('ijk,ijk->ijk', v_i, a_i).sum(2)

    ## Calculate utilities
    u_i_actual = v_i - p_i
    # avg per bid and valuation
    u_i_actual = torch.mean(u_i_actual,1)
    ## average and max regret over all valuations
    regret = u_i_alternative - u_i_actual

    # Explicitaly cleanup TODO:?
    return regret