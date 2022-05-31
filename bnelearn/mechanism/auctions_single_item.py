from typing import List, Tuple

# pylint: disable=E1102
import torch
import math

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

        assert bids.dim() >= 3, "Bid tensor must be at least 3d (*batch_dims x players x items)"
        assert (bids >= 0).all().item(), "All bids must be nonnegative."

        # move bids to gpu/cpu if necessary
        bids = bids.to(self.device)

        # name dimensions
        *batch_dims, player_dim, item_dim = range(bids.dim())  # pylint: disable=unused-variable
        *batch_sizes, n_players, n_items = bids.shape

        if self.random_tie_break: # randomly change order of bidders
            idx = torch.randn((*batch_sizes, n_players), device=bids.device).sort(dim=1)[1]
            bids = batched_index_select(bids, 1, idx)

        # allocate return variables
        payments_per_item = torch.zeros(*batch_sizes, n_players, n_items, device=self.device)
        allocations = torch.zeros(*batch_sizes, n_players, n_items, device=self.device)

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

        return (allocations, payments, second_prices.reshape(*batch_sizes), highest_bids.reshape(*batch_sizes))  # payments: batches x players, allocation: batch x players x items


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

        # move bids to gpu/cpu if necessary
        bids = bids.to(self.device)

        # name dimensions
        *batch_dims, player_dim, item_dim = range(bids.dim())  # pylint: disable=unused-variable
        *batch_sizes, n_players, n_items = bids.shape

        # allocate return variables
        payments_per_item = torch.zeros(*batch_sizes, n_players, n_items, device=self.device)
        allocations = torch.zeros(*batch_sizes, n_players, n_items, device=self.device)

        highest_bids, winning_bidders = bids.max(dim=player_dim, keepdim=True)  # both shapes: [*batch_sizes, 1, n_items]

        # determine ref_bid 
        top2_bids, _ = bids.topk(2, dim=player_dim, sorted=False)
        second_prices, _ = top2_bids.min(player_dim, keepdim=True)
        

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

        return (allocations, payments, second_prices.reshape(*batch_sizes), highest_bids.reshape(*batch_sizes))  # payments: batches x players, allocation: batch x players x items


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

        # move bids to gpu/cpu if necessary
        bids = bids.to(self.device)

        # name dimensions
        *batch_dims, player_dim, item_dim = range(bids.dim())  # pylint: disable=unused-variable
        *batch_sizes, n_players, n_items = bids.shape

        # if torch.min((bids > 0).sum(player_dim)) < 3:
        #     print(2)

        # assert torch.min((bids > 0).sum(player_dim)) >= 3, "Auction format needs at least three participants (with positive bid)"

        # allocate return variables
        payments_per_item = torch.zeros(*batch_sizes, n_players, n_items, device=self.device)
        allocations = torch.zeros(*batch_sizes, n_players, n_items, device=self.device)

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

        return (allocations, payments, third_prices.reshape(*batch_sizes), highest_bids.reshape(*batch_sizes))  # payments: batches x players, allocation: batch x players x items

class SingleItemAllPayAuction(Mechanism):

    def __init__(self, cuda: bool, random_tie_break: bool = False):
        self.random_tie_break = random_tie_break
        super().__init__(cuda=cuda)

    def run(self, bids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Runs a (batch of) First Price Sealed Bid All-Pay Auction.

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

        # move bids to gpu/cpu if necessary
        bids = bids.to(self.device)

        # name dimensions for readibility
        # pylint: disable=unused-variable
        *batch_dims, player_dim, item_dim = range(bids.dim())  # pylint: disable=unused-variable
        *batch_sizes, n_players, n_items = bids.shape
 
        # allocate return variables
        allocations = torch.zeros(*batch_sizes, n_players, n_items, device=self.device)

        # Assign item to the bidder with the highest bid, in case of a tie assign it randomly to one of the winning bidderss
        highest_bids, winning_bidders = bids.max(dim=player_dim, keepdim=True) 

        allocations.scatter_(player_dim, winning_bidders, 1)
        allocations.masked_fill_(mask=bids == 0, value=0)

        payments = bids.reshape(*batch_sizes, n_players) # pay as bid

        # white noise
        #allocations = allocations + torch.normal(torch.zeros_like(allocations), 0.01)

        return allocations, payments # payments: batches x players, allocation: batch x players x items

class TullockContest(Mechanism):
    """Tullock Lottery"""

    def __init__(self, impact_function, cuda: bool = True, use_valuation: bool = True, cost_param: float = None, cost_type: str = None):
        super().__init__(cuda)

        self.impact_fun = impact_function
        self.use_valuation = use_valuation

        if cost_type is None:
            self.cost_function = lambda x: x
        elif cost_type == "additive":
            self.cost_function = lambda x: x + cost_param
        elif cost_type == "multiplicative":
            self.cost_function = lambda x: x * cost_param
        elif cost_type == "exponent":
            self.cost_function = lambda x: x ** cost_param
        else:
            raise ValueError("Cost function not implemented")

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

        # move bids to gpu/cpu if necessary
        bids = bids.to(self.device)

        # apply cost function
        bids = self.cost_function(bids)
        bids[bids < 0] = 0

        # name dimensions
        *batch_dims, player_dim, item_dim = range(bids.dim())  # pylint: disable=unused-variable
        *batch_sizes, n_players, n_items = bids.shape

        # temporarily reshape bids to cope with probability calculation
        #eff = bids.reshape(math.prod([*batch_sizes]), n_players, n_items)

        # allocate return variables
        #payments = bids.reshape(math.prod([*batch_sizes]), n_players) # pay as bid
        payments = bids.reshape(*batch_sizes, n_players) # pay as bid

        # transform bids according to impact function
        #eff = self.impact_fun(eff).reshape(math.prod([*batch_sizes]), n_players)
        bids = self.impact_fun(bids)

        # Calculate winning probabilities
        winning_probs = bids/bids.sum(dim=-2, keepdim=True)
        winning_probs[winning_probs.isnan()] = 1/n_players

        if self.use_valuation:
            #winner = winning_probs.multinomial(num_samples=1)

            #allocations = torch.zeros(math.prod([*batch_sizes]), n_players, device=self.device)

            #allocations.scatter_(1, winner, 1)

            # Don't allocate items that have a winnign bid of zero.
            #allocations.masked_fill_(mask=payments== 0, value=0)

            #transform payments back
            #payments = payments.reshape(*batch_sizes, n_players)


            #allocations = allocations.reshape(*batch_sizes, n_players, n_items)    

            return (winning_probs, payments)  # payments: batches x players, allocation: batch x players x items

        else:
            winning_probs = winning_probs.reshape(*batch_sizes, n_players, n_items)
            #transform payments back
            payments = payments.reshape(*batch_sizes, n_players)
            
            return (winning_probs, payments)  # payments: batches x players, allocation: batch x players x items


class CrowdsourcingContest(Mechanism):
    """Tullock Lottery"""

    def __init__(self, cuda: bool = True, deterministic = True):
        super().__init__(cuda)

        self.deterministic = deterministic

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

        # move bids to gpu/cpu if necessary
        bids = bids.to(self.device)

        # name dimensions
        *batch_dims, player_dim, item_dim = range(bids.dim())  # pylint: disable=unused-variable
        *batch_sizes, n_players, n_items = bids.shape

        if len(batch_sizes) > 1:
            print(2)

        # determine allocation
        if self.deterministic:
            _, allocations = torch.topk(bids, n_players, dim=player_dim)

            _, sorted_allocations = torch.sort(allocations, dim=player_dim)

            allocation = sorted_allocations
        else:
            raise NotImplementedError

        # apply cost function
        #bids = self.cost_function(bids)

        payments = bids.reshape(*batch_sizes, n_players) # pay as bid

        return (allocation, payments)  # payments: batches x players, allocation: batch x players x items

  