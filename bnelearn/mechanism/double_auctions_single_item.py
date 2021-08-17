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


class kDoubleAuction(DoubleAuctionMechanism):

    def __init__(self, n_buyers, n_sellers, k_value, **kwargs):
        super().__init__(n_buyers, n_sellers, **kwargs)
        self.k_value = k_value

    def run(self, bids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:

        assert bids.dim() >= 3, "Bid tensor must be at least 3D (*batch x players x items)."
        assert (bids >= 0).all().item(), "All bids must be nonnegative."

        # move bids to gpu/cpu if necessary
        bids = bids.to(self.device)

        # flaten bids, if there are multiple batch dims
        bids, batch_sizes, n_items, batch_size = self._reshape_for_multiple_batch_dims(bids)
        _, player_dim, item_dim = 0, 1, 2

        bids_buyers, bids_sellers = self._split_bids_into_buyers_and_sellers(bids, player_dim)

        indx_buyers, bids_sorted_buyers = self._sort_bids_with_tracked_indices(player_dim, bids_buyers, descending=True)
        indx_sellers, bids_sorted_sellers = self._sort_bids_with_tracked_indices(player_dim, bids_sellers, descending=False)

        trading_indices = self._determine_trading_indices(bids_sorted_buyers, bids_sorted_sellers)

        trade_buyers = self._broadcast_trading_indices(n_items, self.n_buyers, batch_size, trading_indices)
        trade_sellers = self._broadcast_trading_indices(n_items, self.n_sellers, batch_size, trading_indices)

        trade_prices = self._calc_trade_prices(player_dim, bids_sorted_buyers, bids_sorted_sellers, trading_indices)

        trade_price_buyers, trade_price_sellers = self._determine_individual_trade_prices(trade_buyers, trade_sellers, trade_prices)


        allocations = self._determine_allocations(n_items, batch_size, player_dim, indx_buyers, indx_sellers, trade_buyers, trade_sellers)


        payments = self._determine_payments(n_items, batch_size, player_dim, item_dim, indx_buyers, indx_sellers, trade_price_buyers, trade_price_sellers)

        return self._combine_allocations_and_payments(batch_sizes, n_items, allocations, payments)

    def _determine_payments(self, n_items, batch_size, player_dim, item_dim, indx_buyers, indx_sellers, trade_price_buyers, trade_price_sellers):
        payments_per_item_buyers = self._scatter_values_to_indices((batch_size, self.n_buyers, n_items), indx_buyers, trade_price_buyers, player_dim)
        payments_per_item_sellers = self._scatter_values_to_indices((batch_size, self.n_sellers, n_items), indx_sellers, trade_price_sellers, player_dim)

        payments = torch.cat((payments_per_item_buyers, payments_per_item_sellers),
                             dim=player_dim).sum(dim=item_dim)
                             
        return payments

    def _determine_allocations(self, n_items, batch_size, player_dim, indx_buyers, indx_sellers, trade_buyers, trade_sellers):
        allocations_buyers = self._scatter_values_to_indices((batch_size, self.n_buyers, n_items), indx_buyers, trade_buyers, player_dim)
        allocations_sellers = self._scatter_values_to_indices((batch_size, self.n_sellers, n_items), indx_sellers, trade_sellers, player_dim)
        allocations = torch.cat((allocations_buyers, allocations_sellers), dim=player_dim)
        return allocations

    def _scatter_values_to_indices(self, tensor_shape, indices, values_to_scatter, player_dim):
        scattered_values = torch.zeros(tensor_shape, device=self.device)
        scattered_values = scattered_values.scatter_(dim=player_dim, index=indices,
                                                         src=values_to_scatter)
        return scattered_values

    def _determine_individual_trade_prices(self, trade_buyers, trade_sellers, trade_prices):
        trade_price_buyers = trade_prices*trade_buyers
        trade_price_sellers = trade_prices*trade_sellers
        return trade_price_buyers,trade_price_sellers

    def _calc_trade_prices(self, player_dim, bids_sorted_buyers, bids_sorted_sellers, trading_indices):
        trade_indx = self._determine_break_even_trading_index(player_dim, trading_indices)
        trade_price = torch.add(
                                ((self.k_value)*(torch.gather(bids_sorted_buyers, dim=player_dim, index=trade_indx))),
                                ((1-self.k_value)*(torch.gather(bids_sorted_sellers, dim=player_dim, index=trade_indx)))
                               )
                               
        return trade_price

    def _broadcast_trading_indices(self, n_items, num_players, batch_size, trading_indices):
        trade_players = torch.zeros(batch_size, num_players, n_items, device=self.device)
        trade_players[:,:self.min_player_dim,:] = trading_indices
        return trade_players

    def _determine_break_even_trading_index(self, player_dim, trade):
        min_trade_val, trade_indx = trade.min(dim=player_dim, keepdim=True)
        trade_indx[trade_indx > 0] -= 1  
        trade_indx[min_trade_val == 1] = self.min_player_dim - 1
        return trade_indx

    def _determine_trading_indices(self, bids_sorted_buyers, bids_sorted_sellers):
        """This function expects sorted tensors of buyers and sellers and determines
        which agents are going to trade depending on their bids."""
        trade = torch.ge(bids_sorted_buyers, bids_sorted_sellers).type(torch.float)
        return trade

    def _sort_bids_with_tracked_indices(self, player_dim, bids, descending: bool):
        bids_sorted_init, indx_sort = torch.sort(bids, dim=player_dim, descending=descending)
        bids_sorted = torch.narrow(bids_sorted_init, player_dim, 0, self.min_player_dim)
        return indx_sort,bids_sorted

    def _split_bids_into_buyers_and_sellers(self, bids, player_dim):
        bids_buyers, bids_sellers = torch.split(bids,[self.n_buyers, self.n_sellers], dim=player_dim)
        return bids_buyers,bids_sellers


class VickreyDoubleAuction(DoubleAuctionMechanism):

    def run(self, bids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:

        assert bids.dim() >= 3, "Bid tensor must be at least 3d (*batch x players x items)"
        assert (bids >= 0).all().item(), "All bids must be nonnegative."

        # move bids to gpu/cpu if necessary
        bids = bids.to(self.device)

        # flaten bids, if there are multiple batch dims
        bids, batch_sizes, n_items, batch_size = self._reshape_for_multiple_batch_dims(bids)
        _, player_dim, item_dim = 0, 1, 2

        bids_buyers, bids_sellers = torch.split(bids,[self.n_buyers, self.n_sellers], dim=player_dim)

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

        return self._combine_allocations_and_payments(batch_sizes, n_items, allocations, payments)



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