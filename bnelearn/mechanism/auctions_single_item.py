from typing import Tuple

# pylint: disable=E1102
import torch

from .mechanism import Mechanism
from ..util.tensor_util import batched_index_select


class VickreyAuction(Mechanism):
    "Vickrey / Second Price Sealed Bid Auctions"

    def __init__(self, random_tie_break: bool=False, **kwargs):
        self.random_tie_break = random_tie_break
        super().__init__(**kwargs)

    def run(self, bids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Runs a (batch of) Vickrey/Second Price Sealed Bid Auctions.

        This function is meant for single-item auctions.
        If a bid tensor for multiple items is submitted, each item is auctioned
        independently of one another.

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
            payments:   tensor of dimension (n_batches x n_players)
                        Total payment from player to auctioneer for her
                        allocation in that batch.
        """

        assert bids.dim() == 3, "Bid tensor must be 3d (batch x players x items)"
        assert (bids >= 0).all().item(), "All bids must be nonnegative."

        # move bids to gpu/cpu if necessary
        bids = bids.to(self.device)

        # name dimensions for readibility
        # pylint: disable=unused-variable
        batch_dim, player_dim, item_dim = 0, 1, 2
        batch_size, n_players, n_items = bids.shape

        if self.random_tie_break: # randomly change order of bidders
            idx = torch.randn((batch_size, n_players), device=bids.device).sort(dim=1)[1]
            bids = batched_index_select(bids, 1, idx)

        # allocate return variables
        payments_per_item = torch.zeros(batch_size, n_players, n_items, device=self.device)
        allocations = torch.zeros(batch_size, n_players, n_items, device=self.device)

        highest_bids, winning_bidders = bids.max(dim=player_dim,
                                                 keepdim=True)  # shape of each: [batch_size, 1, n_items]

        # getting the second prices --> price is the lowest of the two highest bids
        top2_bids, _ = bids.topk(2, dim=player_dim, sorted=False)
        second_prices, _ = top2_bids.min(player_dim, keepdim=True)

        payments_per_item.scatter_(player_dim, winning_bidders, second_prices)
        payments = payments_per_item.sum(item_dim)
        allocations.scatter_(player_dim, winning_bidders, 1)
        # Don't allocate items that have a winnign bid of zero.
        allocations.masked_fill_(mask=payments_per_item == 0, value=0)

        if self.random_tie_break: # restore bidder order
            idx_rev = idx.sort(dim=1)[1]
            allocations = batched_index_select(allocations, 1, idx_rev)
            payments = batched_index_select(payments, 1, idx_rev)

            # also revert the order of bids if they're used later on
            bids = batched_index_select(bids, 1, idx_rev)

        return (allocations, payments)  # payments: batches x players, allocation: batch x players x items


class FirstPriceSealedBidAuction(Mechanism):
    """First Price Sealed Bid auction"""

    # TODO: If multiple players submit the highest bid, the implementation chooses the first rather than at random
    def run(self, bids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Runs a (batch of) First Price Sealed Bid Auction.

        This function is meant for single-item auctions.
        If a bid tensor for multiple items is submitted, each item is auctioned
        independently of one another.

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
            payments:   tensor of dimension (n_batches x n_players)
                        Total payment from player to auctioneer for her
                        allocation in that batch.
        """
        assert bids.dim() == 3, "Bid tensor must be 3d (batch x players x items)"
        assert (bids >= 0).all().item(), "All bids must be nonnegative."

        # move bids to gpu/cpu if necessary
        bids = bids.to(self.device)

        # name dimensions for readibility
        batch_dim, player_dim, item_dim = 0, 1, 2  # pylint: disable=unused-variable
        batch_size, n_players, n_items = bids.shape

        # allocate return variables
        payments_per_item = torch.zeros(batch_size, n_players, n_items, device=self.device)
        allocations = torch.zeros(batch_size, n_players, n_items, device=self.device)

        highest_bids, winning_bidders = bids.max(dim=player_dim, keepdim=True)  # both shapes: [batch_size, 1, n_items]

        # replaced by equivalent, faster torch.scatter operation, see below,
        # but keeping nested-loop code for readability
        # note: code in comment references bids.max with keepdim=False.
        ##for batch in range(batch_size):
        ##    for j in range(n_items):
        ##        hb = highest_bidders[batch, j]
        ##        payments_per_item[batch][ highest_bidders[batch, j] ][j] = highest_bids[batch, j]
        ##        allocation[batch][ highest_bidders[batch, j] ][j] = 1
        # The above can be written as the following one-liner:
        payments_per_item.scatter_(player_dim, winning_bidders, highest_bids)
        payments = payments_per_item.sum(item_dim)
        allocations.scatter_(player_dim, winning_bidders, 1)
        # Don't allocate items that have a winnign bid of zero.
        allocations.masked_fill_(mask=payments_per_item == 0, value=0)

        return (allocations, payments)  # payments: batches x players, allocation: batch x players x items


class ThirdPriceSealedBidAuction(Mechanism):
    def run(self, bids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Runs a (batch of) Third Price Sealed Bid Auctions.

        This function is meant for single-item auctions.
        If a bid tensor for multiple items is submitted, each item is auctioned
        independently of one another.

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
            payments:   tensor of dimension (n_batches x n_players)
                        Total payment from player to auctioneer for her
                        allocation in that batch.
        """
        assert torch.min((bids > 0).sum(1)) >= 3, "Auction format needs at least three participants (with positive bid)"
        assert bids.dim() == 3, "Bid tensor must be 3d (batch x players x items)"
        assert (bids >= 0).all().item(), "All bids must be nonnegative."

        # move bids to gpu/cpu if necessary
        bids = bids.to(self.device)

        # name dimensions for readibility
        # pylint: disable=unused-variable
        batch_dim, player_dim, item_dim = 0, 1, 2
        batch_size, n_players, n_items = bids.shape

        # allocate return variables
        payments_per_item = torch.zeros(batch_size, n_players, n_items, device=self.device)
        allocations = torch.zeros(batch_size, n_players, n_items, device=self.device)

        highest_bids, winning_bidders = bids.max(dim=player_dim,
                                                 keepdim=True)  # shape of each: [batch_size, 1, n_items]

        # getting the third prices --> price is the lowest of the three highest bids
        top3_bids, _ = bids.topk(3, dim=player_dim, sorted=False)
        third_prices, _ = top3_bids.min(player_dim, keepdim=True)

        payments_per_item.scatter_(player_dim, winning_bidders, third_prices)
        payments = payments_per_item.sum(item_dim)
        allocations.scatter_(player_dim, winning_bidders, 1)
        # Don't allocate items that have a winnign bid of zero.
        allocations.masked_fill_(mask=payments_per_item == 0, value=0)

        return (allocations, payments)  # payments: batches x players, allocation: batch x players x items


class SingleItemAllPayAuction(Mechanism):

    def run(self, bids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
       
        """
        
        Runs a (batch of) the standard version of the all pay auction.

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
            payments:   tensor of dimension (n_batches x n_players)
                        Total payment from player to auctioneer for her
                        allocation in that batch.
        """

        assert bids.dim() == 3, "Bid tensor must be 3d (batch x players x items)"
        assert (bids >= 0).all().item(), "All bids must be nonnegative."

        # move bids to gpu/cpu if necessary
        bids = bids.to(self.device)

        # name dimensions for readibility
        # pylint: disable=unused-variable
        batch_dim, player_dim, item_dim = 0, 1, 2
        batch_size, n_players, n_items = bids.shape

        # allocate return variables
        payments = bids.reshape(batch_size, n_players) # pay as bid
        allocations = torch.zeros(batch_size, n_players, n_items, device=self.device)

        # Assign item to the bidder with the highest bid, in case of a tie assign it randomly to one of the winning bidderss
        highest_bids, winning_bidders = bids.max(dim=player_dim, keepdim=True) 
        allocations.scatter_(player_dim, winning_bidders, 1)
        # Don't allocate items that have a winnign bid of zero.
        payments_per_item = payments.reshape((payments.shape[0], payments.shape[1], 1))
        allocations.masked_fill_(mask=payments_per_item == 0, value=0)

        return allocations, payments
