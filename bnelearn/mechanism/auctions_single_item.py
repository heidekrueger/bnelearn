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

    # pylint: disable=arguments-differ
    def run(self, bids: torch.Tensor, smooth_market: bool=False) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Runs a (batch of) Vickrey/Second Price Sealed Bid Auctions.

        This function is meant for single-item auctions.
        If a bid tensor for multiple items is submitted, each item is auctioned
        independently of one another.

        Parameters
        ----------
        bids: torch.Tensor
            of bids with dimensions (batch_size, n_players, n_items)
        smooth_market: Smoothens allocations and payments s.t. the ex-post
            utility is continuous again. This introduces a bias though.
            PG then is applicable.

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

        assert bids.dim() >= 3, "Bid tensor must be at least 3d (*batch_dims x players x items)"
        assert (bids >= 0).all().item(), "All bids must be nonnegative."

        device = bids.device

        # name dimensions
        *batch_dims, player_dim, item_dim = range(bids.dim())  # pylint: disable=unused-variable
        *batch_sizes, n_players, n_items = bids.shape

        if self.random_tie_break: # randomly change order of bidders
            idx = torch.randn((*batch_sizes, n_players), device=bids.device).sort(dim=1)[1]
            bids = batched_index_select(bids, 1, idx)

        # calculate payments
        payments_per_item = torch.zeros(*batch_sizes, n_players, n_items, device=device)
        highest_bids, winning_bidders = bids.max(dim=player_dim,
                                                 keepdim=True)  # shape of each: [batch_size, 1, n_items]

        # getting the second prices --> price is the lowest of the two highest bids
        top2_bids, _ = bids.topk(2, dim=player_dim, sorted=False)
        second_prices, _ = top2_bids.min(player_dim, keepdim=True)

        payments_per_item.scatter_(player_dim, winning_bidders, second_prices)
        payments = payments_per_item.sum(item_dim)

        if not smooth_market:
            allocations = torch.zeros(*batch_sizes, n_players, n_items, device=device)
            allocations.scatter_(player_dim, winning_bidders, 1)

            # Don't allocate items that have a winning bid of zero.
            allocations.masked_fill_(mask=payments_per_item==0, value=0)

        else:
            softmax = torch.nn.Softmax(dim=player_dim)
            allocations = softmax(bids / self.smoothing_temperature)

            # redistribute original payments proportional to allocation smoothing
            total_payments = second_prices.view(*batch_sizes, 1, n_items).repeat(1, n_players, 1)
            payments = (allocations * total_payments).sum(axis=item_dim)

        if self.random_tie_break: # restore bidder order
            idx_rev = idx.sort(dim=1)[1]
            allocations = batched_index_select(allocations, 1, idx_rev)
            payments = batched_index_select(payments, 1, idx_rev)

            # also revert the order of bids if they're used later on
            bids = batched_index_select(bids, 1, idx_rev)

        return (allocations, payments)  # payments: batches x players, allocation: batch x players x items


class FirstPriceSealedBidAuction(Mechanism):
    """First Price Sealed Bid auction"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    # TODO: If multiple players submit the highest bid, the implementation chooses the first rather than at random
    # pylint: disable=arguments-differ
    def run(self, bids: torch.Tensor, smooth_market: bool=False) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Runs a (batch of) First Price Sealed Bid Auction.

        This function is meant for single-item auctions.
        If a bid tensor for multiple items is submitted, each item is auctioned
        independently of one another.

        Parameters
        ----------
        bids: torch.Tensor
            of bids with dimensions (*batch_sizes, n_players, n_items)

        Returns
        -------
        (allocation, payments): Tuple[torch.Tensor, torch.Tensor]
            allocation: tensor of dimension (*batch_sizes x n_players x n_items),
                        1 indicating item is allocated to corresponding player
                        in that batch, 0 otherwise
            payments:   tensor of dimension (*batch_sizes x n_players)
                        Total payment from player to auctioneer for her
                        allocation in that batch.
        """
        assert bids.dim() >= 3, "Bid tensor must be at least 3d (*batch_dims x players x items)"
        assert (bids >= 0).all().item(), "All bids must be nonnegative."

        device = bids.device

        # name dimensions
        *batch_dims, player_dim, item_dim = range(bids.dim())  # pylint: disable=unused-variable
        *batch_sizes, n_players, n_items = bids.shape

        # allocate return variables
        payments_per_item = torch.zeros(*batch_sizes, n_players, n_items, device=device)
        allocations = torch.zeros(*batch_sizes, n_players, n_items, device=device)

        highest_bids, winning_bidders = bids.max(dim=player_dim, keepdim=True)  # both shapes: [batch_sizes, 1, n_items]
        payments_per_item.scatter_(player_dim, winning_bidders, highest_bids)
        payments = payments_per_item.sum(item_dim)

        if not smooth_market:
            allocations.scatter_(player_dim, winning_bidders, 1)

            # Don't allocate items that have a winning bid of zero.
            allocations.masked_fill_(mask=payments_per_item == 0, value=0)

        else:
            softmax = torch.nn.Softmax(dim=player_dim)
            allocations = softmax(bids / self.smoothing_temperature)

            # redistribute original payments proportional to allocation smoothing
            total_payments = highest_bids.view(*batch_sizes, 1, n_items).repeat(1, n_players, 1)
            payments = (allocations * total_payments).sum(axis=item_dim)

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

        assert bids.dim() >= 3, "Bid tensor must be at least 3d (*batch_dims x players x items)"
        assert (bids >= 0).all().item(), "All bids must be nonnegative."

        device = bids.device

        # name dimensions
        *batch_dims, player_dim, item_dim = range(bids.dim())  # pylint: disable=unused-variable
        *batch_sizes, n_players, n_items = bids.shape

        assert torch.min((bids > 0).sum(player_dim)) >= 3, "Auction format needs at least three participants (with positive bid)"

        # allocate return variables
        payments_per_item = torch.zeros(*batch_sizes, n_players, n_items, device=device)
        allocations = torch.zeros(*batch_sizes, n_players, n_items, device=device)

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


class AllPayAuction(Mechanism):

    def __init__(self, cuda: bool):
        super().__init__(cuda=cuda)

    def run(self, bids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Runs a (batch of) All-Pay Auctions.

        This function is meant for single-item auctions.
        If a bid tensor for multiple items is submitted, each item is auctioned
        independently of one another.

        Parameters
        ----------
        bids: torch.Tensor
            of bids with dimensions (*batch_sizes, n_players, n_items)

        Returns
        -------
        (allocation, payments): Tuple[torch.Tensor, torch.Tensor]
            allocation: tensor of dimension (*batch_sizes x n_players x n_items),
                        1 indicating item is allocated to corresponding player
                        in that batch, 0 otherwise
            payments:   tensor of dimension (*batch_sizes x n_players)
                        Total payment from player to auctioneer for her
                        allocation in that batch.
        """

        assert bids.dim() >= 3, "Bid tensor must be 3d (batch x players x items)"
        assert (bids >= 0).all().item(), "All bids must be nonnegative."

        device = bids.device

        # name dimensions for readibility
        *batch_dims, player_dim, item_dim = range(bids.dim()) 
        *batch_sizes, n_players, n_items = bids.shape
 
        # allocate return variables
        allocations = torch.zeros(*batch_sizes, n_players, n_items, device=device)

        # Assign item to the bidder with the highest bid, in case of a tie assign it to the first one
        _, winning_bidders = bids.max(dim=player_dim, keepdim=True) 

        allocations.scatter_(player_dim, winning_bidders, 1)
        allocations.masked_fill_(mask=bids == 0, value=0)

        payments = bids.reshape(*batch_sizes, n_players) # pay as bid

        return allocations, payments # payments: batches x players, allocation: batch x players x items

