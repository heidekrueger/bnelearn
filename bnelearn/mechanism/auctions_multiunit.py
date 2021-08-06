"""Auction mechanism for multi-unit aucttions (where objects are homogeneous)."""
from typing import Tuple
import warnings
from functools import reduce
from operator import mul

# pylint: disable=E1102
import torch

from .mechanism import Mechanism
from ..util.tensor_util import batched_index_select


class MultiUnitAuction(Mechanism):
    """Class for multi-unit auctions where multiple identical items (so called
    units) are for sale. Agents thus don't care about winning item no. x or no.
    y, as they're homogeneous.
    """

    @staticmethod
    def _remove_invalid_bids(bids: torch.Tensor) -> torch.Tensor:
        """Helper function for cleaning bids in multi-unit auctions.

        For multi-unit actions bids per must be in decreasing for each bidder.
        If agents' bids fail to fulfill this property, this function sets their bid
        to zero, which will result in no items allocated and thus zero payoff.

        Args:
            bids: torch.Tensor
                of bids with dimensions (*batch_sizes, n_players, n_items); first entry of
                n_items dim corrsponds to bid of first unit, second entry to bid of second
                unit, etc.

        Returns:
            cleaned_bids: torch.Tensor (*batch_sizes, n_players, n_items)
                same dimension as bids, with zero entries whenever a bidder bid
                nondecreasing in a batch.
        """
        diff = bids.sort(dim=-1, descending=True)[0] - bids
        diff = torch.abs(diff).sum(dim=-1) != 0  # boolean, batch_sizes x n_players
        if diff.any():
            warnings.warn('Bids which were not in decreasing order have been ignored!')
        bids[diff] = 0.0

        return bids

    @staticmethod
    def _solve_allocation_problem(
            bids: torch.Tensor,
            random_tie_break: bool = False,
            accept_zero_bids: bool = False,
        ) -> torch.Tensor:
        """For bids (batch x player x item) in descending order for each batch/
        player, returns efficient allocation (0/1, batch x player x item).

        Args:
            bids (torch.Tensor) of agents bids of shape (batch, agent, unit)
            random_tie_break (bool), optinoal: wether or not to randomize the
                order of the agents (matters e.g. when all agents bid same
                amount, then the first agent would always win).
            accept_zero_bids (bool), wether or not agents can win by bidding
                zero on a unit.

        Returns:
            allocations (torch.Tensor) of zeros and ones to indicate the
                alocated units for each batch and for each bid.

        Note:
            This function assumes that validity checks have already been
            performed.

        """
        *batch_sizes, n_players, n_items = bids.shape
        total_batch_size = reduce(mul, batch_sizes, 1)


        if random_tie_break: # randomly change order of bidders
            idx = torch.randn((*batch_sizes, n_players*n_items), device=bids.device) \
                .sort()[1] + (n_players*n_items) \
                * torch.arange(*batch_sizes, device=bids.device).reshape(-1, 1)
            bids_flat = bids.view(-1)[idx]
        else:
            bids_flat = bids.reshape(total_batch_size, n_players*n_items)

        allocations = torch.zeros_like(bids_flat)
        _, sorted_idx = torch.sort(bids_flat, dim=-1, descending=True)
        allocations.scatter_(1, sorted_idx[:, :n_items], 1)

        if random_tie_break:  # restore bidder order
            idx_rev = idx.sort()[1] + (n_players*n_items) \
                    * torch.arange(*batch_sizes, device=bids.device).reshape(-1, 1)
            allocations = allocations.view(-1)[idx_rev.view(-1)]

        # sorting is needed, since tie break could end up in favour of lower valued item
        allocations = allocations.reshape_as(bids).sort(descending=True, dim=-1)[0]

        if not accept_zero_bids:
            allocations.masked_fill_(mask=bids==0, value=0)

        return allocations


class MultiUnitDiscriminatoryAuction(MultiUnitAuction):
    """Multi item discriminatory auction. Units are allocated to the highest
    n_item bids, winners pay as bid.

    Bids of each bidder must be in decreasing order, otherwise the mechanism
    does not accept these bids and allocates no units to this bidder.
    """

    def run(self, bids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Runs a (batch of) multi item discriminatory auction(s). Invalid bids
        (i.e. in increasing order) will be ignored (-> no allocation to that
        bidder), s.t. the bidder might be able to ´learn´ the right behavior.

        Args:
            bids: torch.Tensor
                of bids with dimensions (*batch_sizes, n_players, n_items);
                first entry of n_items dim corrsponds to bid of first unit,
                second entry to bid of second unit, etc.

        Returns:
            (allocation, payments): Tuple[torch.Tensor, torch.Tensor]
                allocation: tensor of dimension (n_batches x n_players x
                    n_items), 1 indicating item is allocated to corresponding
                    player in that batch, 0 otherwise.
                payments: tensor of dimension (n_batches x n_players);
                    total payment from player to auctioneer for her
                    allocation in that batch.
        """
        assert bids.dim() >= 3, "Bid tensor must be at least 3d (*batches x players x items)"
        assert (bids >= 0).all().item(), "All bids must be nonnegative."

        # move bids to gpu/cpu if necessary
        bids = bids.to(self.device)

        # only accept decreasing bids
        # assert torch.equal(bids.sort(dim=item_dim, descending=True)[0], bids), \
        #     "Bids must be in decreasing order"
        bids = self._remove_invalid_bids(bids)

        # Alternative w/o loops
        allocations = self._solve_allocation_problem(bids)

        payments = torch.sum(allocations * bids, dim=-1)  # sum over items

        return (allocations, payments)  # payments: batches x players, allocation: batch x players x items


class MultiUnitUniformPriceAuction(MultiUnitAuction):
    """ In a uniform-price auction, all units are sold at a "market-clearing"
    price such that the total amount demanded is equal to the total amount
    supplied. We adopt the rule that the market-clearing price is the same as
    the highest losing bid.

    Bids of each bidder must be in decreasing order, otherwise the mechanism
    does not accept these bids and allocates no units to this bidder.
    """

    def run(self, bids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Runs a (batch of) Multi Unit Uniform-Price Auction(s). Invalid bids
        (i.e. in increasing order) will be ignored (-> no allocation to that
        bidder), s.t. the bidder might be able to ´learn´ the right behavior.

        Args:
            bids: torch.Tensor
                of bids with dimensions (*batch_sizes, n_players, n_items);
                first entry of n_items dim corrsponds to bid of first unit,
                second entry to bid of second unit, etc.

        Returns:
            (allocation, payments): Tuple[torch.Tensor, torch.Tensor]
                allocation: tensor of dimension (n_batches x n_players x
                    n_items), 1 indicating item is allocated to corresponding
                    player in that batch, 0 otherwise.
                payments: tensor of dimension (n_batches x n_players);
                    total payment from player to auctioneer for her
                    allocation in that batch.
        """
        assert bids.dim() >= 3, "Bid tensor must be at least 3d (*batches x players x items)"
        assert (bids >= 0).all().item(), "All bids must be nonnegative."

        # name dimensions for readibility
        *batch_sizes, n_players, n_items = bids.shape
        total_batch_size = reduce(mul, batch_sizes, 1)

        # move bids to gpu/cpu if necessary
        bids = bids.to(self.device)

        # only accept decreasing bids
        # assert torch.equal(bids.sort(dim=-1, descending=True)[0], bids), \
        #     "Bids must be in decreasing order"
        bids = self._remove_invalid_bids(bids)

        # allocate return variables (flat at this stage)
        allocations = self._solve_allocation_problem(bids)

        # pricing
        payments = torch.zeros(total_batch_size, n_players * n_items, device=self.device)
        bids_flat = bids.reshape(total_batch_size, n_players * n_items)
        _, sorted_idx = torch.sort(bids_flat, dim=-1, descending=True)
        payments.scatter_(1, sorted_idx[:, n_items:n_items + 1], 1)
        payments = torch.t(
            bids_flat[payments.bool()] \
            .repeat(n_players, 1)
        ) * torch.sum(allocations.view(-1, n_players, n_items), dim=-1)

        # payments: batches x players, allocation: batch x players x items
        return (allocations, payments.view(*batch_sizes, n_players))


class MultiUnitVickreyAuction(MultiUnitAuction):
    """In a Vickrey auction, a bidder who wins k units pays the k highest
    losing bids of the other bidders.

    Bids of each bidder must be in decreasing order, otherwise the
    mechanism does not accept these bids and allocates no units to this
    bidder.
    """

    def run(self, bids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Runs a (batch of) Multi Unit Vickrey Auction(s). Invalid bids (i.e.
        in increasing order) will be ignored (-> no allocation to that bidder),
        s.t. the bidder might be able to ´learn´ the right behavior.

        Args:
            bids: torch.Tensor
                of bids with dimensions (*batch_sizes, n_players, n_items);
                first entry of n_items dim corrsponds to bid of first unit,
                second entry to bid of second unit, etc.

        Returns:
            (allocation, payments): Tuple[torch.Tensor, torch.Tensor]
                allocation: tensor of dimension (n_batches x n_players x
                    n_items), 1 indicating item is allocated to corresponding
                    player in that batch, 0 otherwise.
                payments: tensor of dimension (n_batches x n_players);
                    total payment from player to auctioneer for her
                    allocation in that batch.
        """
        assert bids.dim() >= 3, "Bid tensor must be at least 3d (*batches x players x items)"
        assert (bids >= 0).all().item(), "All bids must be nonnegative."

        # name dimensions for readibility
        *batch_sizes, n_players, n_items = bids.shape
        total_batch_size = reduce(mul, batch_sizes, 1)

        # move bids to gpu/cpu if necessary
        bids = bids.to(self.device)

        # only accept decreasing bids
        # assert torch.equal(bids.sort(dim=-1, descending=True)[0], bids), \
        #     "Bids must be in decreasing order"
        bids = self._remove_invalid_bids(bids)

        # allocate return variables
        allocations = self._solve_allocation_problem(bids)

        # allocations
        bids_flat = bids.reshape(total_batch_size, n_players * n_items)
        sorted_bids, sorted_idx = torch.sort(bids_flat, dim=-1, descending=True)

        # priceing TODO: optimize this, possibly with torch.topk?
        agent_ids = torch.arange(0, n_players, device=self.device) \
            .repeat(total_batch_size, n_items, 1).transpose_(1, 2)
        highest_loosing_player = agent_ids \
            .reshape(total_batch_size, n_players * n_items) \
            .gather(dim=1, index=sorted_idx)[:, n_items:2 * n_items] \
            .repeat_interleave(n_players * torch.ones(total_batch_size, device=self.device).long(), dim=0) \
            .reshape(total_batch_size, n_players, n_items)
        highest_losing_prices = sorted_bids[:, n_items:2 * n_items] \
            .repeat_interleave(n_players * torch.ones(total_batch_size, device=self.device).long(), dim=0) \
            .reshape_as(bids) \
            .masked_fill_((highest_loosing_player == agent_ids).reshape_as(bids), 0) \
            .sort(descending=True)[0]
        payments = (allocations * highest_losing_prices).sum(-1)

        return (allocations, payments)  # payments: batches x players, allocation: batch x players x items


class FPSBSplitAwardAuction(MultiUnitAuction):
    """First-price sealed-bid split-award auction: Multiple agents bidding for
    either 100% of the share or 50%.

    We define a bids as ´torch.Tensor´ with dimensions (*batch_sizes,
    n_players, n_bids=2), where the first bid is for the 50% share and the
    second for the 100% share.
    """

    def _solve_allocation_problem(self, bids: torch.Tensor,
                                  random_tie_break: bool = False,
                                  accept_zero_bids: bool = False):
        """Computes allocation

        Args:
            bids: torch.Tensor
                of bids with dimensions (batch_size, n_players, n_bids=2), values = [0,Inf],
                the first bid is for the 50% share and the second for the 100% share

        Returns:
            allocation: torch.Tensor, dim (batch_size, b_bundles=2), values = {0,1}
        """

        *batch_sizes, n_players, n_bundles = bids.shape
        total_batch_size = reduce(mul, batch_sizes, 1)
        bids_flat = bids.view(total_batch_size, n_players, n_bundles)

        device = bids.device

        assert n_bundles == 2, "This auction type only allows for two bids per agent."

        winning_bundles = torch.zeros_like(bids_flat)

        if random_tie_break:  # randomly change order of bidders
            idx = torch.randn((*batch_sizes, n_players), device=device).sort(dim=1)[1]
            bids_flat = batched_index_select(bids_flat, 1, idx)

        # Get highest bid for 100% lot and the two highest bids for 50% lots
        best_100_bids, best_100_indices = bids_flat[:, :, 1].min(dim=1)
        best_50_bids, best_50_indices = bids_flat[:, :, 0].topk(2, largest=False, dim=1)

        # Determine winning bids
        sum_of_two_best_50_bids = best_50_bids.sum(dim=1)
        bid_100_won = best_100_bids < sum_of_two_best_50_bids  # tie break: in favor of 50/50

        batch_arange = torch.arange(0, total_batch_size, device=device)
        zeros = torch.zeros_like(best_100_indices)
        ones = torch.ones_like(best_100_indices)

        # 100% lot bid wins
        winning_bundles[batch_arange, best_100_indices, ones] = 1

        # 50% lot bid wins of highest 50% bidder
        winning_bundles[batch_arange, best_50_indices[:, 0], zeros] = 1

        # 50% lot bid wins of second-highest 50% bidder
        winning_bundles[batch_arange, best_50_indices[:, 1], zeros] = 1

        # Make sure losers are not allocated lots
        winning_bundles[bid_100_won, :, 0] = 0
        winning_bundles[~bid_100_won, :, 1] = 0

        if random_tie_break: # restore bidder order
            idx_rev = idx.sort(dim=1)[1]
            winning_bundles = batched_index_select(winning_bundles, 1, idx_rev)
            # bids_flat = batched_index_select(bids_flat, 1, idx_rev)  # unused

        winning_bundles = winning_bundles.view_as(bids)  # reshape to original sahpe

        if not accept_zero_bids:
            winning_bundles.masked_fill_(mask=bids==0, value=0)

        return winning_bundles

    def _calculate_payments_first_price(self, bids: torch.Tensor, allocations: torch.Tensor):
        """Computes first prices.

        Args:
            bids: torch.Tensor, of dimensions (batch_size, n_bidders,
                n_bids=2), values = [0, Inf].
            allocations: torch.Tensor, dim: (batch_size, b_bundles=2), values =
                {0, 1}.

        Returns:
            payments: tensor of dimension (*n_batches x n_players); total
                payment from player to auctioneer for her allocation in that
                batch.
        """
        return torch.sum(allocations * bids, dim=-1)

    def run(self, bids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Performs a batch of split-award auction rounds.

        Args:
            bids: torch.Tensor
                of bids with dimensions (*batch_sizes, n_players, n_items=2);
                first entry of n_items dim corrsponds to bid for 50% lot,
                second entry to bid for 100% lot, etc.

        Returns:
            (allocation, payments): Tuple[torch.Tensor, torch.Tensor]
                allocation: tensor of dimension (*n_batches x n_players x 2),
                    1 indicating item is allocated to corresponding player in
                    that batch, 0 otherwise.
                payments: tensor of dimension (*n_batches x n_players);
                    total payment from player to auctioneer for her
                    allocation in that batch.
        """
        assert bids.dim() >= 3, "Bid tensor must be at least 3d (*batches x players x items)"
        assert (bids >= 0).all().item(), "All bids must be nonnegative."

        allocation = self._solve_allocation_problem(bids)
        payments = self._calculate_payments_first_price(bids, allocation)

        return (allocation, payments)
