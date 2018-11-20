from abc import ABC, abstractmethod
from typing import Tuple

#pylint: disable=E1102
import torch

# Type declarations
Outcome = Tuple[torch.Tensor, torch.Tensor]

class Mechanism(ABC):
    """
    Abstract class.
    A Mechanism collects bids from all players, then allocates available
    items as well as payments for each of the players.
    """

    @abstractmethod
    def run(self, bids):
        pass

class TwoByTwoBimatrixGame(Mechanism):
    def __init__(self, outcomes: torch.Tensor, cuda: bool = True, names: dict = None):
        self.cuda = cuda and torch.cuda.is_available()
        self.device = 'cuda' if self.cuda else 'cpu'

        assert outcomes.shape == torch.Size([2,2,2])
        self.outcomes = outcomes.float().to(self.device)

        self.names = names

    def get_player_name(self, player_id: int):
        if self.names and "players" in self.names.keys():
            return self.names["players"][player_id]
        else:
            return player_id
    
    def get_action_name(self, action_id: int):
        if self.names and "actions" in self.names.keys():
            return self.names["actions"][action_id]
        else:
            return action_id

    def run(self, bids):
        """bids are actually indices of actions"""

        assert bids.dim() == 3, "Bid matrix must be 3d (batch x players x items)"
        assert bids.dtype == torch.int64, "actions must be integers!"

        batch_dim, player_dim, item_dim = 0, 1, 2
        batch_size, n_players, n_items = bids.shape

        assert n_items == 1, "only single action per player in this setting"
        assert n_players == 2, "only implemented for 2 players right now"

        #move to gpu/cpu if needed
        bids = bids.to(self.device)
        bids = bids.view(batch_size, n_players)

        allocations = torch.zeros(batch_size, n_players, n_items, device=self.device)
        
        ## memory allocation and Loop replaced by equivalent vectorized version below:
        ## (keep old code as comment for readibility)
        #payments = torch.zeros(batch_size, n_players, device=self.device)
        #for batch in range(batch_size):
        #    for player in range(n_players):
        #        payments[batch, player] = -self.outcomes[bids[batch,0], bids[batch,1]][player]

        # payment to "game master" is the negative outcome
        payments = -self.outcomes[bids[:,0], bids[:,1]].view(batch_size, n_players)

        return (allocations, payments)

class PrisonersDilemma(TwoByTwoBimatrixGame):
    def __init__(self, cuda: bool = True):
        super().__init__(
            outcomes = torch.tensor([[[-1, -1],[-3, 0]], [[ 0, -3],[-2,-2]]]),
            cuda = cuda,
            names = {
                "player_names": ["RowPlayer", "ColPlayer"],
                "action_names": ["Cooperate", "Defect"]
            }
        )

class BattleOfTheSexes(TwoByTwoBimatrixGame):
    def __init__(self, cuda: bool = True):
        super().__init__(
            outcomes=torch.tensor([[[3, 2],[0,0]], [[0,0],[2,3]]]),
            cuda=cuda,
            names = {
                "player_names": ["Boy", "Girl"],
                "action_names": ["Action", "Romance"]
            }
        )

class MatchingPennies(TwoByTwoBimatrixGame):
    def __init__(self, cuda: bool = True):
        super().__init__(
            outcomes=torch.tensor([[[1, -1],[-1, 1,]], [[-1, 1], [1, -1]]]),
            cuda=cuda,
            names = {
                "player_names": ["Even", "Odd"],
                "action_names": ["Heads", "Tails"]
            }
        )


class VickreyAuction(Mechanism):

    def __init__(self, cuda: bool = True):
        self.cuda = cuda and torch.cuda.is_available()
        self.device = 'cuda' if self.cuda else 'cpu'
    
    def run(self, bids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        assert bids.dim() == 3, "Bid matrix must be 3d (batch x players x items)"
        assert (bids >= 0).all().item(), "All bids must be nonnegative."

        # move bids to gpu/cpu if necessary
        bids = bids.to(self.device)

        # name dimensions for readibility
        batch_dim, player_dim, item_dim = 0, 1, 2
        batch_size, n_players, n_items = bids.shape

        # allocate return variables
        payments_per_item = torch.zeros(batch_size, n_players, n_items, device = self.device)
        allocations = torch.zeros(batch_size, n_players, n_items, device = self.device)

        highest_bids, winning_bidders = bids.max(dim = player_dim, keepdim=True) # shape of each: [batch_size, 1, n_items]
        
        # getting the second prices --> price is the lowest of the two highest bids
        top2_bids, _ = bids.topk(2, dim = player_dim, sorted=False)
        second_prices, _ = top2_bids.min(player_dim, keepdim=True)
        
        payments_per_item.scatter_(player_dim, winning_bidders, second_prices)
        payments = payments_per_item.sum(item_dim)
        allocations.scatter_(player_dim, winning_bidders, 1)
        # Don't allocate items that have a winnign bid of zero.
        allocations.masked_fill_(mask=payments_per_item == 0, value=0)

        return (allocations, payments) # payments: batches x players, allocation: batch x players x items
        


class FirstPriceSealedBidAuction(Mechanism):

    def __init__(self, cuda: bool = True):
        self.cuda = cuda and torch.cuda.is_available()
        self.device = 'cuda' if self.cuda else 'cpu'

    # TODO: If multiple players submit the highest bid, the current implementation chooses the first rather than at random
    def run(self, bids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Runs a first (batch of a) First Price Sealed Bid Auction.

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
            allocation: tensor of dimension (n_batches)
        """
        assert bids.dim() == 3, "Bid matrix must be 3d (batch x players x items)"
        assert (bids >= 0).all().item(), "All bids must be nonnegative."

        # move bids to gpu/cpu if necessary
        bids = bids.to(self.device)

        # name dimensions for readibility
        batch_dim, player_dim, item_dim = 0, 1, 2
        batch_size, n_players, n_items = bids.shape

        # allocate return variables
        payments_per_item = torch.zeros(batch_size, n_players, n_items, device = self.device)
        allocations = torch.zeros(batch_size, n_players, n_items, device = self.device)

        highest_bids, winning_bidders = bids.max(dim = player_dim, keepdim=True) # shape of each: [batch_size, 1, n_items]

        # replaced by torch scatter operation, see below
        # note: deleted code references bids.max with keepdim=False.
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

        return (allocations, payments) # payments: batches x players, allocation: batch x players x items
