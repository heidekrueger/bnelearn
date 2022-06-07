from typing import Tuple

import torch

from .mechanism import Mechanism


class TullockContest(Mechanism):
    """Tullock Contest"""

    def __init__(self, impact_function, cuda: bool = True):
        super().__init__(cuda)

        self.impact_fun = impact_function

    def run(self, bids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Runs a (batch of) Tullock Contests.

        This function is meant for single-item contests.
        If a effort tensor for multiple items is submitted, each item is auctioned
        independently of one another.

        Note that if multiple contestants submit the highest effort, we choose the first one. 

        Parameters
        ----------
        bids: torch.Tensor
            of efforts with dimensions (*batch_sizes, n_players, n_items)

        Returns
        -------
        (winning_probs, payments): Tuple[torch.Tensor, torch.Tensor]
            winning_probs: tensor of dimension (*batch_sizes x n_players x n_items),
                        Winning probabilities defined by the impact function and the contest success function
            payments:   tensor of dimension (*batch_sizes x n_players)
                        Total payment from player to auctioneer for her
                        allocation in that batch.
        """
        assert bids.dim() >= 3, "Effort tensor must be at least 3d (*batch_dims x players x items)"
        assert (bids >= 0).all().item(), "All efforts must be nonnegative."

        # move bids to gpu/cpu if necessary
        bids = bids.to(self.device)

        # name dimensions
        *batch_dims, player_dim, item_dim = range(bids.dim())  # pylint: disable=unused-variable
        *batch_sizes, n_players, n_items = bids.shape

        # allocate return variables
        payments = bids.reshape(*batch_sizes, n_players) # pay as bid

        # transform bids according to impact function
        bids = self.impact_fun(bids)

        # Calculate winning probabilities
        winning_probs = bids/bids.sum(dim=-2, keepdim=True)
        winning_probs[winning_probs.isnan()] = 1/n_players # equal chances if all contestants exert zero effotr

        return (winning_probs, payments)  # payments: batches x players, allocation: batch x players x items


class CrowdsourcingContest(Mechanism):
    """Crowdsourcing Contest"""

    def __init__(self, cuda: bool = True):
        super().__init__(cuda)

    def run(self, bids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Runs a (batch of) Crowdsourcing Contests.

        If a effort tensor for multiple items is submitted, each item is auctioned
        independently of one another.

        Parameters
        ----------
        efforts: torch.Tensor
            of efforts with dimensions (*batch_sizes, n_players, n_items)

        Returns
        -------
        (allocation, payments): Tuple[torch.Tensor, torch.Tensor]
            allocation: tensor of dimension (*batch_sizes x n_players x n_items),
                        Entry indicates which prize the corresponding contestant would have won
            payments:   tensor of dimension (*batch_sizes x n_players)
                        Total payment from player to auctioneer for her
                        allocation in that batch.
        """
        assert bids.dim() >= 3, "Bid tensor must be at least 3d (*batch_dims x players x items)"
        assert (bids >= 0).all().item(), "All bids must be nonnegative."

        # move bids to gpu/cpu if necessary
        bids = bids.to(self.device)

        # name dimensions
        *batch_dims, player_dim, item_dim = range(bids.dim())  
        *batch_sizes, n_players, n_items = bids.shape

        _, allocations = torch.topk(bids, n_players, dim=player_dim)
        _, sorted_allocations = torch.sort(allocations, dim=player_dim)
        allocation = sorted_allocations

        payments = bids.reshape(*batch_sizes, n_players) # pay as bid

        return (allocation, payments)  # payments: batches x players, allocation: batch x players x items
