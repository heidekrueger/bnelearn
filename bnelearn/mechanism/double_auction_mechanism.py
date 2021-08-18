# -*- coding: utf-8 -*-

"""
This module implements double auctions.
"""

from abc import ABC, abstractmethod
import torch
from .mechanism import Mechanism
from functools import reduce
from operator import mul
from typing import Tuple, Dict


class DoubleAuctionMechanism(Mechanism, ABC):
    """
    Double Auction Mechanism - Interpreted as a Bayesian game.
    A Mechanism collects bids from all players, then allocates available
    items as well as payments for each of the players.
    """

    def __init__(self, n_buyers, n_sellers, **kwargs): 
        super().__init__(**kwargs)

        # 0:n_buyers: indices for buyers
        # n_buyers+1 : n_players : indices for sellers

        self.n_buyers = n_buyers
        self.n_sellers = n_sellers
        self.min_player_dim = min(self.n_buyers, self.n_sellers)


    def play(self, action_profile):
        return self.run(bids=action_profile)

    @abstractmethod
    def run(self, bids):
        """Alias for play for double auction mechanisms"""
        raise NotImplementedError()


class DeterministicDoubleAuctionMechanism(DoubleAuctionMechanism, ABC):

    def run(self, bids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:

        assert bids.dim() >= 3, "Bid tensor must be at least 3D (*batch x players x items)."
        assert (bids >= 0).all().item(), "All bids must be nonnegative."

        # move bids to gpu/cpu if necessary
        bids = bids.to(self.device)
        *batch_sizes, _, n_items = bids.shape
        # flaten bids, if there are multiple batch dims
        bids = self._reshape_for_multiple_batch_dims(bids)

        sorted_bids_and_indices_dict, params_dict = self._determine_bid_and_indice_variants(batch_sizes, n_items, bids)

        trade_buyers, trade_sellers = self._determine_combined_and_splitted_trading_indices(
            params_dict, sorted_bids_and_indices_dict)

        trade_price_buyers, trade_price_sellers = self._determine_individual_trade_prices(
            trade_buyers, trade_sellers, params_dict, sorted_bids_and_indices_dict)

        allocations = self._determine_allocations(trade_buyers, trade_sellers, params_dict, sorted_bids_and_indices_dict)

        payments = self._determine_payments(trade_price_buyers, trade_price_sellers, params_dict, sorted_bids_and_indices_dict)

        return self._combine_allocations_and_payments(allocations, payments, params_dict)
    
    @abstractmethod
    def _determine_individual_trade_prices(self, trade_buyers, trade_sellers, params_dict, sorted_bids_and_indices_dict):
        """This returns tensors of the individual trade prices of buyers and sellers"""
        raise NotImplementedError()
    
    @abstractmethod
    def _determine_combined_and_splitted_trading_indices(self, params_dict, sorted_bids_and_indices_dict):
        """This returns tensors of the trading indices of buyers and sellers"""
        raise NotImplementedError()
    
    def _determine_bid_and_indice_variants(self, batch_sizes, n_items, bids):
        params_dict = self._create_params_dict(batch_sizes, n_items)
        bids_buyers, bids_sellers = self._split_bids_into_buyers_and_sellers(bids, params_dict["player_dim"])
        indx_buyers, bids_sorted_buyers, bids_sorted_init_buyers = self._sort_bids_with_tracked_indices(params_dict["player_dim"], bids_buyers, descending=True)
        indx_sellers, bids_sorted_sellers, bids_sorted_init_sellers = self._sort_bids_with_tracked_indices(params_dict["player_dim"], bids_sellers, descending=False)
        sorted_bids_and_indices_dict = self._create_sorted_bids_and_indices_dict(
            bids_buyers, bids_sellers,
            bids_sorted_buyers,
            bids_sorted_init_buyers,
            bids_sorted_sellers,
            bids_sorted_init_sellers,
            indx_buyers, indx_sellers)
        return sorted_bids_and_indices_dict, params_dict
    
    def _create_params_dict(self, batch_sizes, n_items) -> Dict:
        return {
            "batch_sizes": batch_sizes,
            "batch_size": reduce(mul, batch_sizes, 1),
            "player_dim": 1,
            "item_dim": 2,
            "n_items": n_items
        }
    
    def _create_sorted_bids_and_indices_dict(self, 
    bids_buyers, bids_sellers, bids_sorted_buyers, bids_sorted_init_buyers, bids_sorted_sellers,
    bids_sorted_init_sellers,  indx_buyers, indx_sellers) -> Dict:
        return {
                "bids_buyers": bids_buyers,
                "bids_sellers": bids_sellers,
                "bids_sorted_buyers": bids_sorted_buyers,
                "bids_sorted_sellers": bids_sorted_sellers,
                "bids_sorted_init_buyers": bids_sorted_init_buyers,
                "bids_sorted_init_sellers": bids_sorted_init_sellers,
                "indx_buyers": indx_buyers,
                "indx_sellers": indx_sellers,
                }

    def _reshape_for_multiple_batch_dims(self, bids):
        *batch_sizes, _, n_items = bids.shape
        batch_size = reduce(mul, batch_sizes, 1)
        bids = bids.view(batch_size, self.n_buyers+self.n_sellers, n_items)
        return bids

    def _combine_allocations_and_payments(self, allocations, payments, params_dict):
        return (allocations.view(*params_dict["batch_sizes"], self.n_buyers+self.n_sellers, params_dict["n_items"]),
                payments.view(*params_dict["batch_sizes"], self.n_buyers+self.n_sellers))
    
    def _split_bids_into_buyers_and_sellers(self, bids, player_dim):
        bids_buyers, bids_sellers = torch.split(bids,[self.n_buyers, self.n_sellers], dim=player_dim)
        return bids_buyers,bids_sellers
    
    def _determine_combined_and_splitted_trading_indices(self, n_items, batch_size, bids_sorted_buyers, bids_sorted_sellers):
        trading_indices = self._determine_trading_indices(bids_sorted_buyers, bids_sorted_sellers)

        trade_buyers = self._broadcast_trading_indices(n_items, self.n_buyers, batch_size, trading_indices)
        trade_sellers = self._broadcast_trading_indices(n_items, self.n_sellers, batch_size, trading_indices)
        return trading_indices,trade_buyers,trade_sellers
    
    def _determine_trading_indices(self, bids_sorted_buyers, bids_sorted_sellers):
        """This function expects sorted tensors of buyers and sellers and determines
        which agents are going to trade depending on their bids."""
        trade = torch.ge(bids_sorted_buyers, bids_sorted_sellers).type(torch.bool)
        return trade
    
    def _sort_bids_with_tracked_indices(self, player_dim, bids, descending: bool):
        bids_sorted_init, indx_sort = torch.sort(bids, dim=player_dim, descending=descending)
        bids_sorted = torch.narrow(bids_sorted_init, player_dim, 0, self.min_player_dim)
        return indx_sort,bids_sorted, bids_sorted_init
    
    def _broadcast_trading_indices(self, n_items, num_players, batch_size, trading_indices):
        trade_players = torch.zeros(batch_size, num_players, n_items, device=self.device)
        trade_players[:,:self.min_player_dim,:] = trading_indices
        return trade_players
    
    def _determine_break_even_trading_index(self, player_dim, trade):
        min_trade_val, trade_indx = trade.min(dim=player_dim, keepdim=True)
        trade_indx[trade_indx > 0] -= 1  
        trade_indx[min_trade_val == 1] = self.min_player_dim - 1
        return trade_indx
    
    def _determine_payments(self, trade_price_buyers, trade_price_sellers, params_dict, sorted_bids_and_indices_dict):
        payments_per_item_buyers = self._scatter_values_to_indices(
            (params_dict["batch_size"], self.n_buyers, params_dict["n_items"]), sorted_bids_and_indices_dict["indx_buyers"], trade_price_buyers, params_dict["player_dim"])
        payments_per_item_sellers = self._scatter_values_to_indices(
            (params_dict["batch_size"], self.n_sellers, params_dict["n_items"]), sorted_bids_and_indices_dict["indx_sellers"], trade_price_sellers, params_dict["player_dim"])

        payments = torch.cat((payments_per_item_buyers, payments_per_item_sellers),
                             dim=params_dict["player_dim"]).sum(dim=params_dict["item_dim"])
        return payments

    def _determine_allocations(self, trade_buyers, trade_sellers, params_dict, sorted_bids_and_indices_dict):
        allocations_buyers = self._scatter_values_to_indices(
            (params_dict["batch_size"], self.n_buyers, params_dict["n_items"]), sorted_bids_and_indices_dict["indx_buyers"], trade_buyers, params_dict["player_dim"])
        allocations_sellers = self._scatter_values_to_indices(
            (params_dict["batch_size"], self.n_sellers, params_dict["n_items"]), sorted_bids_and_indices_dict["indx_sellers"], trade_sellers, params_dict["player_dim"])
        allocations = torch.cat((allocations_buyers, allocations_sellers), dim=params_dict["player_dim"])
        return allocations
    
    def _scatter_values_to_indices(self, tensor_shape, indices, values_to_scatter, player_dim):
        scattered_values = torch.zeros(tensor_shape, device=self.device)
        scattered_values = scattered_values.scatter_(dim=player_dim, index=indices,
                                                         src=values_to_scatter)
        return scattered_values
    

