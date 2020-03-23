from typing import Tuple
import warnings

# pylint: disable=E1102
import torch

from .mechanism import Mechanism


def _remove_invalid_bids(bids: torch.Tensor) -> torch.Tensor:
    """
    Helper function for cleaning bids in multi-unit auctions.
    For multi-unit actions bids per must be in decreasing for each bidder.
    If agents' bids fail to fulfill this property, this function sets their bid
    to zero, which will result in no items allocated and thus zero payoff.

    Parameters
    ----------
    bids: torch.Tensor
        of bids with dimensions (batch_size, n_players, n_items); first entry of
        n_items dim corrsponds to bid of first unit, second entry to bid of second
        unit, etc.

    Returns
    -------
    cleaned_bids: torch.Tensor (batch_size, n_players, n_items)
        same dimension as bids, with zero entries whenever a bidder bid
        nondecreasing in a batch.
    """
    cleaned_bids = bids.clone()

    diff = bids.sort(dim=2, descending=True)[0] - bids
    diff = torch.abs(diff).sum(dim=2) != 0  # boolean, batch_size x n_players
    if diff.any():
        warnings.warn('bids which were not in dcreasing order have been ignored!')
    cleaned_bids[diff] = 0.0

    return cleaned_bids


def _get_multiunit_allocation(
        bids: torch.Tensor,
        random_tie_break: bool = True,
        accept_zero_bids: bool = False,
    ) -> torch.Tensor:
    """For bids (batch x player x item) in descending order for each batch/player,
       returns efficient allocation (0/1, batch x player x item)

       This function assumes that validity checks have already been performed.
    """
    # for readability
    batch_dim, player_dim, item_dim = 0, 1, 2 #pylint: disable=unused-variable
    batch_size, n_players, n_items = bids.shape

    allocations = torch.zeros(batch_size, n_players*n_items, device=bids.device)

    if random_tie_break: # randomly change order of bidders
        idx = torch.randn((batch_size, n_players*n_items), device=bids.device) \
              .sort()[1] + (n_players*n_items) \
              * torch.arange(batch_size, device=bids.device).reshape(-1, 1)
        bids_flat = bids.view(-1)[idx]
    else:
        bids_flat = bids.reshape(batch_size, n_players*n_items)

    _, sorted_idx = torch.sort(bids_flat, descending=True)
    allocations.scatter_(player_dim, sorted_idx[:,:n_items], 1)

    if random_tie_break: # restore bidder order
        idx_rev = idx.sort()[1] + (n_players*n_items) \
                  * torch.arange(batch_size, device=bids.device).reshape(-1, 1)
        allocations = allocations.view(-1)[idx_rev.view(-1)]

    # sorting is needed, since tie break could end up in favour of lower valued item
    allocations = allocations.reshape_as(bids).sort(descending=True, dim=item_dim)[0]

    if not accept_zero_bids:
        allocations.masked_fill_(mask=bids==0, value=0)

    return allocations


class MultiItemDiscriminatoryAuction(Mechanism):
    """ Multi item discriminatory auction.
        Items are allocated to the highest n_item bids, winners pay as bid.

        Bids of each bidder must be in decreasing
        order, otherwise the mechanism does not accept these bids and allocates no units
        to this bidder.
    """

    def run(self, bids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Runs a (batch of) multi item discriminatory auction(s). Invalid bids (i.e. in
        increasing order) will be ignored (-> no allocation to that bidder), s.t.
        the bidder might be able to ´learn´ the right behavior.

        Parameters
        ----------
        bids: torch.Tensor
            of bids with dimensions (batch_size, n_players, n_items); first entry of
            n_items dim corrsponds to bid of first unit, second entry to bid of second
            unit, etc. (how much one add. unit is ´valued´)

        Returns
        -------
        (allocation, payments): Tuple[torch.Tensor, torch.Tensor]
            allocation: tensor of dimension (n_batches x n_players x n_items),
                1 indicating item is allocated to corresponding player
                in that batch, 0 otherwise
            payments: tensor of dimension (n_batches x n_players);
                total payment from player to auctioneer for her
                allocation in that batch.
        """
        assert bids.dim() == 3, "Bid tensor must be 3d (batch x players x items)"
        assert (bids >= 0).all().item(), "All bids must be nonnegative."

        # move bids to gpu/cpu if necessary
        bids = bids.to(self.device)

        # only accept decreasing bids
        # assert torch.equal(bids.sort(dim=item_dim, descending=True)[0], bids), \
        #     "Bids must be in decreasing order"
        bids = _remove_invalid_bids(bids)

        # Alternative w/o loops
        allocations = _get_multiunit_allocation(bids)

        payments = torch.sum(allocations * bids, dim=2)  # sum over items

        return (allocations, payments)  # payments: batches x players, allocation: batch x players x items


class MultiItemUniformPriceAuction(Mechanism):
    """ In a uniform-price auction, all units are sold at a “market-clearing” price
        such that the total amount demanded is equal to the total amount supplied.
        We adopt the rule that the market-clearing price is the same as the highest
        losing bid.

        Bids of each bidder must be in decreasing order, otherwise the mechanism
        does not accept these bids and allocates no units to this bidder.
    """

    def run(self, bids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Runs a (batch of) Multi Item Uniform-Price Auction(s). Invalid bids (i.e. in
        increasing order) will be ignored (-> no allocation to that bidder), s.t.
        the bidder might be able to ´learn´ the right behavior.

        (allocation, payments): Tuple[torch.Tensor, torch.Tensor]
            allocation: tensor of dimension (n_batches x n_players x n_items),
                1 indicating item is allocated to corresponding player
                in that batch, 0 otherwise
            payments: tensor of dimension (n_batches x n_players),
                total payment from player to auctioneer for her
                allocation in that batch.
        """
        assert bids.dim() == 3, "Bid tensor must be 3d (batch x players x items)"
        assert (bids >= 0).all().item(), "All bids must be nonnegative."

        # name dimensions for readibility
        batch_dim, player_dim, item_dim = 0, 1, 2  # pylint: disable=unused-variable
        batch_size, n_players, n_items = bids.shape

        # move bids to gpu/cpu if necessary
        bids = bids.to(self.device)

        # only accept decreasing bids
        # assert torch.equal(bids.sort(dim=item_dim, descending=True)[0], bids), \
        #     "Bids must be in decreasing order"
        bids = _remove_invalid_bids(bids)

        # allocate return variables (flat at this stage)
        allocations = _get_multiunit_allocation(bids)

        # pricing
        payments = torch.zeros(batch_size, n_players * n_items, device=self.device)
        bids_flat = bids.reshape(batch_size, n_players * n_items)
        _, sorted_idx = torch.sort(bids_flat, descending=True)
        payments.scatter_(1, sorted_idx[:, n_items:n_items + 1], 1)
        payments = torch.t(bids_flat[payments.bool()].repeat(n_players, 1)) \
                   * torch.sum(allocations, dim=item_dim)

        return (allocations, payments)  # payments: batches x players, allocation: batch x players x items


class MultiItemVickreyAuction(Mechanism):
    """ In a Vickrey auction, a bidder who wins k units pays the k highest
        losing bids of the other bidders.

        Bids of each bidder must be in decreasing order, otherwise the
        mechanism does not accept these bids and allocates no units to this
        bidder.
    """

    def run(self, bids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Runs a (batch of) Multi Item Vickrey Auction(s). Invalid bids (i.e. in
        increasing order) will be ignored (-> no allocation to that bidder), s.t.
        the bidder might be able to ´learn´ the right behavior.

        Parameters
        ----------
        bids: torch.Tensor
            of bids with dimensions (batch_size, n_players, n_items)

        Returns
        -------
        (allocation, payments): Tuple[torch.Tensor, torch.Tensor]
            allocation: tensor of dimension (n_batches x n_players x n_items),
                1 indicating item is allocated to corresponding player
                in that batch, 0 otherwise
            payments: tensor of dimension (n_batches x n_players),
                total payment from player to auctioneer for her
                allocation in that batch.
        """
        assert bids.dim() == 3, "Bid tensor must be 3d (batch x players x items)"
        assert (bids >= 0).all().item(), "All bids must be nonnegative."

        # name dimensions for readibility
        batch_dim, player_dim, item_dim = 0, 1, 2  # pylint: disable=unused-variable
        batch_size, n_players, n_items = bids.shape

        # move bids to gpu/cpu if necessary
        bids = bids.to(self.device)

        # only accept decreasing bids
        # assert torch.equal(bids.sort(dim=item_dim, descending=True)[0], bids), \
        #     "Bids must be in decreasing order"
        bids = _remove_invalid_bids(bids)

        # allocate return variables
        allocations = _get_multiunit_allocation(bids)

        # allocations
        bids_flat = bids.reshape(batch_size, n_players * n_items)
        sorted_bids, sorted_idx = torch.sort(bids_flat, descending=True)

        # priceing TODO: optimize this, possibly with torch.topk?
        agent_ids = torch.arange(0, n_players, device=self.device) \
            .repeat(batch_size, n_items, 1).transpose_(1, 2)
        highest_loosing_player = agent_ids.reshape(batch_size, n_players * n_items) \
                                     .gather(dim=1, index=sorted_idx)[:, n_items:2 * n_items] \
            .repeat_interleave(n_players * torch.ones(batch_size, device=self.device).long(), dim=0) \
            .reshape((batch_size, n_players, n_items))
        highest_losing_prices = sorted_bids[:, n_items:2 * n_items] \
            .repeat_interleave(n_players * torch.ones(batch_size, device=self.device).long(), dim=0) \
            .reshape_as(bids).masked_fill_((highest_loosing_player == agent_ids) \
                                           .reshape_as(bids), 0).sort(descending=True)[0]
        payments = (allocations * highest_losing_prices).sum(item_dim)

        return (allocations, payments)  # payments: batches x players, allocation: batch x players x items

def batched_index_select(input, dim, index):
    """
    Extends the torch ´index_select´ function to be used for multiple batches
    at once.

    author:
        dashesy @ https://discuss.pytorch.org/t/batched-index-select/9115/11

    args:
        input: Tensor which is to be indexed
        dim: Dimension
        index: Index tensor which proviedes the seleting and ordering.

    returns/yields:
        Indexed tensor
    """
    for ii in range(1, len(input.shape)):
        if ii != dim:
            index = index.unsqueeze(ii)
    expanse = list(input.shape)
    expanse[0] = -1
    expanse[dim] = -1
    index = index.expand(expanse)
    return torch.gather(input, dim, index)

class FPSBSplitAwardAuction(Mechanism):
    """
    First-price sealed-bid split-award auction: Multiple agents bidding for either 100%
    of the share or 50%.

    We define a bids as ´torch.Tensor´ with dimensions (batch_size, n_players, n_bids=2),
    where the first bid is for the 100% share and the second for the 50% share.

    """

    def _solve_allocation_problem(self, bids: torch.Tensor,
                                  random_tie_break: bool = True):
        """
        Computes allocation and welfare

        Parameters
        ----------
        bids: torch.Tensor
            of bids with dimensions (batch_size, n_players, n_bids=2), values = [0,Inf],
            the first bid is for the 100% share and the second for the 50% share

        Returns
        -------
        allocation: torch.Tensor(batch_size, b_bundles=2), values = {0,1}
        """

        assert bids.dim() == 3, "Bid tensor must be 3d (batch x players x items)"
        assert (bids >= 0).all().item(), "All bids must be nonnegative."

        n_batch, n_players, n_bundles = bids.shape

        assert n_bundles == 2, "This auction type only allows for two bids per agent."

        winning_bundles = torch.zeros_like(bids)

        if random_tie_break: # randomly change order of bidders
            idx = torch.randn((n_batch, n_players), device=bids.device).sort(dim=1)[1]
            bids = batched_index_select(bids, 1, idx)

        best_100_bids, best_100_indices = bids[:,:,0].min(dim=1)
        best_50_bids, best_50_indices = bids[:,:,1].topk(2, largest=False, dim=1)

        sum_of_two_best_50_bids = best_50_bids.sum(dim=1)
        bid_100_won = best_100_bids < sum_of_two_best_50_bids # tie break: in favor of 50/50

        batch_arange = torch.arange(0, n_batch, device=bids.device)
        winning_bundles[
                batch_arange,
                best_100_indices,
                torch.zeros_like(best_100_indices)
            ] = 1
        winning_bundles[
                batch_arange,
                best_50_indices[:,0],
                torch.ones_like(best_100_indices)
            ] = 1
        winning_bundles[
                batch_arange,
                best_50_indices[:,1],
                torch.ones_like(best_100_indices)
            ] = 1

        winning_bundles[bid_100_won,:,1] = 0
        winning_bundles[~bid_100_won,:,0] = 0

        if random_tie_break: # restore bidder order
            idx_rev = idx.sort(dim=1)[1]
            winning_bundles = batched_index_select(winning_bundles, 1, idx_rev)
            bids = batched_index_select(bids, 1, idx_rev) # are bids even needed later on? 

        return winning_bundles

    def _calculate_payments_first_price(self, bids: torch.Tensor, allocations: torch.Tensor):
        """
        Computes first prices

        Parameters
        ----------
        bids: torch.Tensor
            of bids with dimensions (batch_size, n_bidders, n_bids=2), values = [0,Inf]
        allocations: torch.Tensor(batch_size, b_bundles=2), values = {0,1}

        Returns
        -------
        payments: torch.Tensor(batch_size, n_bidders), values = [0, Inf]
        """
        return torch.sum(allocations * bids, dim=2)

    def run(self, bids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Performs a specific auction

        Parameters
        ----------
        bids: torch.Tensor
            of bids with dimensions (batch_size, n_players, 2) [0,Inf]

        Returns
        -------
        allocation: torch.Tensor(batch_size, n_bidders, 2)
        payments: torch.Tensor(batch_size, n_bidders)
        """

        allocation = self._solve_allocation_problem(bids)
        payments = self._calculate_payments_first_price(bids, allocation)

        return (allocation, payments)
