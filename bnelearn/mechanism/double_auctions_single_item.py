"""

This module implements mechanisms for double auction.
Double auctions contains buyers and sellers. 

In a given bid profile for double auctions, 
    0 : n_buyers -> indices for buyers
    n_buyers+1 : n_players -> indices for sellers

allocation for a buyer is 1 when the buyer buys an item.
payment for a buyer is the amount buyer pays for an item.

allocation for a seller is 1 when the seller sells an item.
payment for a seller is the amount seller receives for an item.

"""

from typing import Tuple

import torch

from .double_auction_mechanism import DoubleAuctionMechanism
from ..util.tensor_util import batched_index_select


class AverageAuction(DoubleAuctionMechanism):

    def run(self, bids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:

        assert bids.dim() == 3, "Bid tensor must be 3d (batch x players x items)"
        assert (bids >= 0).all().item(), "All bids must be nonnegative."

        # move bids to gpu/cpu if necessary
        bids = bids.to(self.device)

        batch_dim, player_dim, item_dim = 0, 1, 2
        bids_buyers, bids_sellers = torch.split(bids,[self.n_buyers, self.n_sellers], dim=player_dim)
        batch_size, _, n_items = bids.shape
        
        # allocate return variables

        payments_per_item_buyers = torch.zeros(batch_size, self.n_buyers, n_items, device=self.device)
        allocations_buyers = torch.zeros(batch_size, self.n_buyers, n_items, device=self.device)

        payments_per_item_sellers = torch.zeros(batch_size, self.n_sellers, n_items, device=self.device)
        allocations_sellers = torch.zeros(batch_size, self.n_sellers, n_items, device=self.device)

        allocations_buyers = torch.ge(bids_buyers,bids_sellers).type(torch.uint8)
        allocations_sellers = allocations_buyers

        payments_per_item_buyers = torch.add(bids_buyers*allocations_buyers, 
                                             bids_sellers*allocations_sellers)*0.5
        payments_per_item_sellers = payments_per_item_buyers

        allocations = torch.cat((allocations_buyers, allocations_sellers), dim=player_dim)
        payments = torch.cat((payments_per_item_buyers, payments_per_item_sellers), 
                            dim=player_dim).sum(dim=item_dim)
        
        return (allocations, payments)


class VickreyDoubleAuction(DoubleAuctionMechanism):

    def run(self, bids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:

        assert bids.dim() == 3, "Bid tensor must be 3d (batch x players x items)"
        assert (bids >= 0).all().item(), "All bids must be nonnegative."

        # move bids to gpu/cpu if necessary
        bids = bids.to(self.device)

        batch_dim, player_dim, item_dim = 0, 1, 2
        bids_buyers, bids_sellers = torch.split(bids,[self.n_buyers, self.n_sellers], dim=player_dim)
        batch_size, _, n_items = bids.shape

        # allocate return variables

        payments_per_item_buyers = torch.zeros(batch_size, self.n_buyers, n_items, device=self.device)
        allocations_buyers = torch.zeros(batch_size, self.n_buyers, n_items, device=self.device)

        payments_per_item_sellers = torch.zeros(batch_size, self.n_sellers, n_items, device=self.device)
        allocations_sellers = torch.zeros(batch_size, self.n_sellers, n_items, device=self.device)

        allocations_buyers = torch.gt(bids_buyers,bids_sellers).type(torch.uint8)
        allocations_sellers = allocations_buyers

        payments_per_item_buyers = bids_sellers*allocations_buyers
        payments_per_item_sellers = bids_buyers*allocations_sellers

        allocations = torch.cat((allocations_buyers, allocations_sellers), dim=player_dim)
        payments = torch.cat((payments_per_item_buyers, payments_per_item_sellers), 
                            dim=player_dim).sum(dim=item_dim)

        return (allocations, payments)





