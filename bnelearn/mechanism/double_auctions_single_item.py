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


class kDoubleAuction(DoubleAuctionMechanism):

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

        bids_sorted_init_buyers, indx_buyers = torch.sort(bids_buyers, dim=player_dim, descending=True)
        bids_sorted_init_sellers, indx_sellers = torch.sort(bids_sellers, dim=player_dim, descending=False)

        min_player_dim = min(self.n_buyers, self.n_sellers)

        bids_sorted_buyers = torch.narrow(bids_sorted_init_buyers, player_dim, 0, min_player_dim)
        bids_sorted_sellers = torch.narrow(bids_sorted_init_sellers, player_dim, 0, min_player_dim)

        trade = torch.ge(bids_sorted_buyers, bids_sorted_sellers).type(torch.float)
        min_trade_val, trade_indx = trade.min(dim=player_dim, keepdim=True)
        trade_indx[trade_indx > 0] -= 1  
        trade_indx[min_trade_val == 1] = min_player_dim - 1

        trade_buyers = torch.zeros(batch_size, self.n_buyers, n_items, device=self.device)
        trade_buyers[:,:min_player_dim,:] = trade

        trade_sellers = torch.zeros(batch_size, self.n_sellers, n_items, device=self.device)
        trade_sellers[:,:min_player_dim,:] = trade

        trade_price = torch.add(
                                ((self.k)*(torch.gather(bids_sorted_buyers, dim=player_dim, index=trade_indx))),
                                ((1-self.k)*(torch.gather(bids_sorted_sellers, dim=player_dim, index=trade_indx)))
                               )
        
        trade_price_buyers = trade_price*trade_buyers
        trade_price_sellers = trade_price*trade_sellers

        allocations_buyers = allocations_buyers.scatter_(dim=player_dim, index=indx_buyers, 
                                                         src=trade_buyers)
        allocations_sellers = allocations_sellers.scatter_(dim=player_dim, index=indx_sellers, 
                                                           src=trade_sellers)

        payments_per_item_buyers = payments_per_item_buyers.scatter_(dim=player_dim, index=indx_buyers, 
                                                                     src=trade_price_buyers)
    
        payments_per_item_sellers = payments_per_item_sellers.scatter_(dim=player_dim, index=indx_sellers, 
                                                                       src=trade_price_sellers)

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

        bids_sorted_init_buyers, indx_buyers = torch.sort(bids_buyers, dim=player_dim, descending=True)
        bids_sorted_init_sellers, indx_sellers = torch.sort(bids_sellers, dim=player_dim, descending=False)

        min_player_dim = min(self.n_buyers, self.n_sellers)

        bids_sorted_buyers = torch.narrow(bids_sorted_init_buyers, player_dim, 0, min_player_dim)
        bids_sorted_sellers = torch.narrow(bids_sorted_init_sellers, player_dim, 0, min_player_dim)

        trade = torch.ge(bids_sorted_buyers, bids_sorted_sellers).type(torch.float)
        min_trade_val, trade_indx = trade.min(dim=player_dim, keepdim=True)
        trade_indx[trade_indx > 0] -= 1 
        trade_indx[min_trade_val == 1] = min_player_dim - 1

        trade_buyers = torch.zeros(batch_size, self.n_buyers, n_items, device=self.device)
        trade_buyers[:,:min_player_dim,:] = trade

        trade_sellers = torch.zeros(batch_size, self.n_sellers, n_items, device=self.device)
        trade_sellers[:,:min_player_dim,:] = trade

        trade_indx_inc_buyers = trade_indx.clone().detach()
        trade_indx_inc_buyers[trade_indx_inc_buyers < (self.n_buyers - 1)] += 1

        trade_indx_inc_sellers = trade_indx.clone().detach()
        trade_indx_inc_sellers[trade_indx_inc_sellers < (self.n_sellers - 1)] += 1     

        trade_price_indx_buyers = torch.gather(bids_sorted_init_buyers, dim=player_dim, index=trade_indx)
        trade_price_indx_sellers = torch.gather(bids_sorted_init_sellers, dim=player_dim, index=trade_indx)

        trade_price_indx_inc_buyers = torch.gather(bids_sorted_init_buyers, dim=player_dim, index=trade_indx_inc_buyers)
        trade_price_indx_inc_sellers = torch.gather(bids_sorted_init_sellers, dim=player_dim, index=trade_indx_inc_sellers)

        trade_price_init_buyers = torch.max(trade_price_indx_sellers, trade_price_indx_inc_buyers)
        trade_price_init_sellers = torch.min(trade_price_indx_buyers, trade_price_indx_inc_sellers)

        trade_price_buyers = torch.where((trade_indx > 0) & (trade_indx < (self.n_buyers - 1)), 
                                         trade_price_init_buyers, trade_price_indx_sellers) * trade_buyers
        trade_price_sellers = torch.where((trade_indx > 0) & (trade_indx < (self.n_sellers - 1)), 
                                          trade_price_init_sellers, trade_price_indx_buyers) * trade_sellers

        allocations_buyers = allocations_buyers.scatter_(dim=player_dim, index=indx_buyers, 
                                                         src=trade_buyers)
        allocations_sellers = allocations_sellers.scatter_(dim=player_dim, index=indx_sellers, 
                                                           src=trade_sellers)

        payments_per_item_buyers = payments_per_item_buyers.scatter_(dim=player_dim, index=indx_buyers, 
                                                                     src=trade_price_buyers)
    
        payments_per_item_sellers = payments_per_item_sellers.scatter_(dim=player_dim, index=indx_sellers, 
                                                                       src=trade_price_sellers)


        allocations = torch.cat((allocations_buyers, allocations_sellers), dim=player_dim)
        payments = torch.cat((payments_per_item_buyers, payments_per_item_sellers), 
                            dim=player_dim).sum(dim=item_dim)
        
        return (allocations, payments)



# class McAfeeDoubleAuction(DoubleAuctionMechanism):

#     def run(self, bids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:

#         assert bids.dim() == 3, "Bid tensor must be 3d (batch x players x items)"
#         assert (bids >= 0).all().item(), "All bids must be nonnegative."

#         # move bids to gpu/cpu if necessary
#         bids = bids.to(self.device)

#         batch_dim, player_dim, item_dim = 0, 1, 2
#         bids_buyers, bids_sellers = torch.split(bids,[self.n_buyers, self.n_sellers], dim=player_dim)
#         batch_size, _, n_items = bids.shape

#         # allocate return variables

#         payments_per_item_buyers = torch.zeros(batch_size, self.n_buyers, n_items, device=self.device)
#         allocations_buyers = torch.zeros(batch_size, self.n_buyers, n_items, device=self.device)

#         payments_per_item_sellers = torch.zeros(batch_size, self.n_sellers, n_items, device=self.device)
#         allocations_sellers = torch.zeros(batch_size, self.n_sellers, n_items, device=self.device)

#         bids_sorted_init_buyers, indx_buyers = torch.sort(bids_buyers, dim=player_dim, descending=True)
#         bids_sorted_init_sellers, indx_sellers = torch.sort(bids_sellers, dim=player_dim, descending=False)

#         min_player_dim = min(self.n_buyers, self.n_sellers)

#         bids_sorted_buyers = torch.narrow(bids_sorted_init_buyers, player_dim, 0, min_player_dim)
#         bids_sorted_sellers = torch.narrow(bids_sorted_init_sellers, player_dim, 0, min_player_dim)

#         trade = torch.ge(bids_sorted_buyers, bids_sorted_sellers).type(torch.float)
#         _, trade_indx = trade.min(dim=player_dim, keepdim=True)
#         trade_indx[trade_indx > 0] -= 1 

        
#         trade_indx_dec =  trade_indx.clone().detach()
#         trade_indx_dec[trade_indx_dec > 0] -= 1     

#         trade_price_init = torch.add(
#                                  torch.gather(bids_buyers_sorted, dim=player_dim, index=trade_indx_dec),
#                                  torch.gather(bids_sellers_sorted, dim=player_dim, index=trade_indx_dec)
#                                )*0.5
        
#         trade_price = torch.where(trade_indx > 0, trade_price_buyers_init, trade_price_sellers_indx)

#         trade_price_buyers_indx = torch.gather(bids_buyers_sorted, dim=player_dim, index=trade_indx)
#         trade_price_sellers_indx = torch.gather(bids_sellers_sorted, dim=player_dim, index=trade_indx)

#         trade_price_buyers_indx_dec = torch.gather(bids_buyers_sorted, dim=player_dim, index=trade_indx_dec)
#         trade_price_sellers_indx_dec = torch.gather(bids_sellers_sorted, dim=player_dim, index=trade_indx_dec)

#         trade_price_buyers_init = torch.max(trade_price_sellers_indx, trade_price_buyers_indx_dec)
#         trade_price_sellers_init = torch.min(trade_price_buyers_indx, trade_price_sellers_indx_dec)

#         trade_price_buyers = torch.where(trade_indx > 0, trade_price_buyers_init, trade_price_sellers_indx) * trade
#         trade_price_sellers = torch.where(trade_indx > 0, trade_price_sellers_init, trade_price_buyers_indx) * trade

#         allocations_buyers = allocations_buyers.scatter_(dim=player_dim, index=indx_buyers, 
#                                                          src=trade)
#         allocations_sellers = allocations_sellers.scatter_(dim=player_dim, index=indx_sellers, 
#                                                            src=trade)

#         payments_per_item_buyers = payments_per_item_buyers.scatter_(dim=player_dim, index=indx_buyers, 
#                                                                      src=trade_price_buyers)
    
#         payments_per_item_sellers = payments_per_item_sellers.scatter_(dim=player_dim, index=indx_sellers, 
#                                                                        src=trade_price_sellers)


#         allocations = torch.cat((allocations_buyers, allocations_sellers), dim=player_dim)
#         payments = torch.cat((payments_per_item_buyers, payments_per_item_sellers), 
#                             dim=player_dim).sum(dim=item_dim)
        
#         return (allocations, payments)