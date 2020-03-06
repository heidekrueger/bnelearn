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


def _get_multiunit_allocation(bids: torch.Tensor) -> torch.Tensor:
    """For bids (batch x player x item) in descending order for each batch/player,
       returns efficient allocation (0/1, batch x player x item)

       This function assumes that validity checks have already been performed.
    """
    # for readability
    batch_dim, player_dim, item_dim = 0, 1, 2  # pylint: disable=unused-variable
    batch_size, n_players, n_items = bids.shape

    allocations = torch.zeros(batch_size, n_players * n_items, device=bids.device)
    bids_flat = bids.reshape(batch_size, n_players * n_items)
    _, sorted_idx = torch.sort(bids_flat, descending=True)
    allocations.scatter_(player_dim, sorted_idx[:, :n_items], 1)
    allocations = allocations.reshape_as(bids)
    allocations.masked_fill_(mask=bids == 0, value=0)

    # Equivalent but slow: for loops
    # # add fictitious negative bids (for case in which one bidder wins all items -> IndexError)
    # allocations = torch.zeros(batch_size, n_players, n_items, device=self.device)
    # bids_extend = -1 * torch.ones(batch_size, n_players, n_items+1, device=self.device)
    # bids_extend[:,:,:-1] = bids
    # for batch in range(batch_size):
    #     current_bids = bids_extend.clone().detach()[batch,:,0]
    #     current_bids_indices = [0] * n_players
    #     for _ in range(n_items):
    #         winner = current_bids.argmax()
    #         allocations[batch,winner,current_bids_indices[winner]] = 1
    #         current_bids_indices[winner] += 1
    #         current_bids[winner] = bids_extend.clone().detach()[batch,winner,current_bids_indices[winner]]
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

        # Simple but slow: for loops
        # # add fictitious negative bids (for case in which one bidder wins all items -> IndexError)
        # bids_extend = -1 * torch.ones(batch_size, n_players, n_items+1, device=self.device)
        # bids_extend[:,:,:-1] = bids
        # # allocate return variables
        # allocations = torch.zeros(batch_size, n_players, n_items, device=self.device)
        # payments = torch.zeros(batch_size, n_players, device=self.device)
        # for batch in range(batch_size):
        #     current_bids = bids_extend.clone().detach()[batch,:,0]
        #     current_bids_indices = [0] * n_players
        #     for _ in range(n_items):
        #         winner = current_bids.argmax()
        #         allocations[batch,winner,current_bids_indices[winner]] = 1
        #         current_bids_indices[winner] += 1
        #         current_bids[winner] = bids_extend.clone().detach()[batch,winner,current_bids_indices[winner]]
        #     market_clearing_price = current_bids.max()
        #     payments[batch,:] = market_clearing_price * torch.sum(allocations[batch,::], dim=item_dim-1)

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

        # # add fictitious negative bids (for case in which one bidder wins all items -> IndexError)
        # bids_extend = -1 * torch.ones(batch_size, n_players, n_items+1, device=self.device)
        # bids_extend[:,:,:-1] = bids
        # payments = torch.zeros(batch_size, n_players, device=self.device)
        # for batch in range(batch_size):
        #     current_bids = bids_extend.clone().detach()[batch,:,0]
        #     current_bids_indices = torch.tensor([0] * n_players, device=self.device)
        #     for _ in range(n_items):
        #         winner = current_bids.argmax()
        #         allocations[batch,winner,current_bids_indices[winner]] = 1
        #         current_bids_indices[winner] += 1
        #         current_bids[winner] = bids_extend.clone().detach()[batch,winner,current_bids_indices[winner]]
        #     won_items_per_agent = torch.sum(allocations[batch,::], dim=item_dim-1)
        #     for agent in range(n_players):
        #         mask = [True] * n_players
        #         mask[agent] = False
        #         highest_losing_prices_indices = current_bids_indices.clone().detach()[mask]
        #         highest_losing_prices = current_bids.clone().detach()[mask]
        #         for _ in range(int(won_items_per_agent[agent])):
        #             highest_losing_price_agent = int(highest_losing_prices.argmax())
        #             payments[batch,agent] += highest_losing_prices[highest_losing_price_agent]
        #             highest_losing_prices_indices[highest_losing_price_agent] += 1
        #             highest_losing_prices = \
        #                 bids_extend.clone().detach()[batch,mask,highest_losing_prices_indices]

        return (allocations, payments)  # payments: batches x players, allocation: batch x players x items
