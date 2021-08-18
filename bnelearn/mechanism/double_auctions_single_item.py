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
import torch
from typing import Tuple
from functools import reduce
from operator import mul

from .double_auction_mechanism import DeterministicDoubleAuctionMechanism


class kDoubleAuction(DeterministicDoubleAuctionMechanism):

    def __init__(self, n_buyers, n_sellers, k_value, **kwargs):
        super().__init__(n_buyers, n_sellers, **kwargs)
        self.k_value = k_value

    def _determine_combined_and_splitted_trading_indices(self, params_dict, bids_dict, indices_dict):
        trading_indices = self._determine_trading_indices(
            bids_dict["bids_sorted_buyers"], bids_dict["bids_sorted_sellers"])

        trade_buyers = self._broadcast_trading_indices(
            params_dict["n_items"], self.n_buyers, params_dict["batch_size"], trading_indices)
        trade_sellers = self._broadcast_trading_indices(
            params_dict["n_items"], self.n_sellers, params_dict["batch_size"], trading_indices)
        indices_dict["trading_indices"] = trading_indices
        return trade_buyers,trade_sellers

    def _determine_individual_trade_prices(self, trade_buyers, trade_sellers, params_dict, bids_dict, indices_dict):
        trade_prices, trade_indx = self._calc_trade_prices(
            params_dict["player_dim"],
             bids_dict["bids_sorted_buyers"],
              bids_dict["bids_sorted_sellers"],
               indices_dict["trading_indices"])
        indices_dict["trade_indx"] = trade_indx
        trade_price_buyers = trade_prices*trade_buyers
        trade_price_sellers = trade_prices*trade_sellers
        return trade_price_buyers,trade_price_sellers

    def _calc_trade_prices(self, player_dim, bids_sorted_buyers, bids_sorted_sellers, trading_indices):
        trade_indx = self._determine_break_even_trading_index(player_dim, trading_indices)
        trade_price = torch.add(
                                ((self.k_value)*(torch.gather(bids_sorted_buyers, dim=player_dim, index=trade_indx))),
                                ((1-self.k_value)*(torch.gather(bids_sorted_sellers, dim=player_dim, index=trade_indx)))
                               )
        return trade_price, trade_indx


class VickreyDoubleAuction(DeterministicDoubleAuctionMechanism):
    def _determine_combined_and_splitted_trading_indices(self, params_dict, bids_dict, indices_dict):
        trading_indices = self._determine_trading_indices(
            bids_dict["bids_sorted_buyers"], bids_dict["bids_sorted_sellers"])

        trade_buyers = self._broadcast_trading_indices(
            params_dict["n_items"], self.n_buyers, params_dict["batch_size"], trading_indices)
        trade_sellers = self._broadcast_trading_indices(
            params_dict["n_items"], self.n_sellers, params_dict["batch_size"], trading_indices)
        indices_dict["trading_indices"] = trading_indices
        return trade_buyers,trade_sellers
    
    def _determine_individual_trade_prices(self, trade_buyers, trade_sellers, params_dict, bids_dict, indices_dict):
        trade_indx = self._determine_break_even_trading_index(params_dict["player_dim"], indices_dict["trading_indices"])
        trade_indx_inc_buyers = trade_indx.clone().detach()
        trade_indx_inc_buyers[trade_indx_inc_buyers < (self.n_buyers - 1)] += 1

        trade_indx_inc_sellers = trade_indx.clone().detach()
        trade_indx_inc_sellers[trade_indx_inc_sellers < (self.n_sellers - 1)] += 1     

        trade_price_indx_buyers = torch.gather(bids_dict["bids_sorted_init_buyers"], dim=params_dict["player_dim"], index=trade_indx)
        trade_price_indx_sellers = torch.gather(bids_dict["bids_sorted_init_sellers"], dim=params_dict["player_dim"], index=trade_indx)

        trade_price_indx_inc_buyers = torch.gather(bids_dict["bids_sorted_init_buyers"], dim=params_dict["player_dim"], index=trade_indx_inc_buyers)
        trade_price_indx_inc_sellers = torch.gather(bids_dict["bids_sorted_init_sellers"], dim=params_dict["player_dim"], index=trade_indx_inc_sellers)

        trade_price_init_buyers = torch.max(trade_price_indx_sellers, trade_price_indx_inc_buyers)
        trade_price_init_sellers = torch.min(trade_price_indx_buyers, trade_price_indx_inc_sellers)

        trade_price_buyers = torch.where((trade_indx > 0) & (trade_indx < (self.n_buyers - 1)), 
                                         trade_price_init_buyers, trade_price_indx_sellers) * trade_buyers
        trade_price_sellers = torch.where((trade_indx > 0) & (trade_indx < (self.n_sellers - 1)), 
                                          trade_price_init_sellers, trade_price_indx_buyers) * trade_sellers
                                          
        return trade_price_buyers,trade_price_sellers

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