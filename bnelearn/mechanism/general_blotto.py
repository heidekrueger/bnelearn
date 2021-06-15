import torch
import os
from typing import Tuple


from .mechanism import Mechanism


class GeneralBlotto(Mechanism):
    "The standard General Blotto game following the description of Borel (1921)"

    def run(self, bids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
       
        """
        
        Runs a (batch of) the standard version of the General Blotto game.

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

        # TODO - should we ensure that the sum of the bids, equals our budget?
        # TODO - implement the budget constraint for the bidders => implement an additional player..

        # move bids to gpu/cpu if necessary
        bids = bids.to(self.device)

        # assign budgets to relative bids
        

        # name dimensions for readibility
        # pylint: disable=unused-variable
        batch_dim, player_dim, item_dim = 0, 1, 2
        batch_size, n_players, n_items = bids.shape

        # allocate return variables
        payments = torch.zeros(batch_size, n_players, device=self.device)
        allocations = torch.ones(batch_size, n_players, n_items, device=self.device)

        _, loosing_bidders = bids.min(dim = 1, keepdims = True)

        allocations.scatter_(player_dim, loosing_bidders, 0)
        
        return allocations, payments