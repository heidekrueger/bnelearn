# -*- coding: utf-8 -*-
from abc import ABC, abstractmethod
from typing import Tuple

#pylint: disable=E1102
import torch

# Type declarations
Outcome = Tuple[torch.Tensor, torch.Tensor]

class Game(ABC):
    """
    Base class for any kind of games
    """

    @abstractmethod
    def play(self, actions):
        # get actions from players and define outcome
        pass


class Mechanism(Game):
    """
    Auction Mechanism - Interpreted as a Bayesian game.
    A Mechanism collects bids from all players, then allocates available
    items as well as payments for each of the players.
    """
    def play(self, actions):
        # TODO: ensure `actions` are valid bids
        return self.run(bids=actions)

    @abstractmethod
    def run(self, bids):
        pass

class MatrixGame(Game, ABC):
    """A complete information Matrix game."""
    # pylint: disable=abstract-method
    def __init__(self, n_players: int, outcomes: torch.Tensor, cuda: bool = True, names: dict = None):
        self.cuda = cuda and torch.cuda.is_available()
        self.device = 'cuda' if self.cuda else 'cpu'
        self.n_players = n_players

        self.outcomes = outcomes.float().to(self.device)

        # TODO: this should actually be [n_actions_p1, n_actions_p2, ..., n_actions_p_n, n_players]
        # then play()-code can be moved here from 2x2 subclass!
        assert outcomes.shape == torch.Size([n_players, n_players, n_players]), 'invalid outcome matrix shape'
        self.names = names

    def check_input_validity(self, action_profile):
        """Assert validity of action profile

           An action profile should have shape of a mechanism (batch x players x items).
           In a matrix game it should therefore be (batch x players x 1).
           TODO: Each player's action should be a valid index for that player.
        """

        assert action_profile.dim() == 3, "Bid matrix must be 3d (batch x players x items)"
        assert action_profile.dtype == torch.int64, "actions must be integers!"

        # pylint: disable=unused-variable
        batch_dim, player_dim, item_dim = 0, 1, 2
        batch_size, n_players, n_items = action_profile.shape

        assert n_items == 1, "only single action per player in matrix game setting"
        assert n_players == self.n_players, "one action per player must be provided"

class TwoByTwoBimatrixGame(MatrixGame):
    """A Matrix game with two players and two actions each"""
    def __init__(self, outcomes: torch.Tensor, cuda: bool = True, names: dict = None):
        assert outcomes.shape == torch.Size([2,2,2])
        super().__init__(2, outcomes, cuda=cuda, names=names )

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

    def play(self, actions):
        """bids are actually indices of actions"""

        super().check_input_validity(actions)

        # pylint: disable=unused-variable
        batch_dim, player_dim, item_dim = 0, 1, 2
        batch_size, n_players, n_items = actions.shape

        #move to gpu/cpu if needed
        actions = actions.to(self.device)
        actions = actions.view(batch_size, n_players)

        # allocation is a dummy and will always be 0 --> all utility is
        # represented by negative payments
        allocations = torch.zeros(batch_size, n_players, n_items, device=self.device)

        ## memory allocation and Loop replaced by equivalent vectorized version below:
        ## (keep old code as comment for readibility)
        #payments = torch.zeros(batch_size, n_players, device=self.device)
        #for batch in range(batch_size):
        #    for player in range(n_players):
        #        payments[batch, player] = -self.outcomes[bids[batch,0], bids[batch,1]][player]

        # payment to "game master" is the negative outcome
        payments = -self.outcomes[actions[:,0], actions[:,1]].view(batch_size, n_players)

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

        assert bids.dim() == 3, "Bid tensor must be 3d (batch x players x items)"
        assert (bids >= 0).all().item(), "All bids must be nonnegative."

        # move bids to gpu/cpu if necessary
        bids = bids.to(self.device)

        # name dimensions for readibility
        # pylint: disable=unused-variable
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

        # move bids to gpu/cpu if necessary
        bids = bids.to(self.device)

        # name dimensions for readibility
        batch_dim, player_dim, item_dim = 0, 1, 2 #pylint: disable=unused-variable
        batch_size, n_players, n_items = bids.shape

        # allocate return variables
        payments_per_item = torch.zeros(batch_size, n_players, n_items, device = self.device)
        allocations = torch.zeros(batch_size, n_players, n_items, device = self.device)

        highest_bids, winning_bidders = bids.max(dim = player_dim, keepdim=True) # shape of each: [batch_size, 1, n_items]

        # replaced by equivalent torch.scatter operation, see below,
        # but keeping looped code for readability
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

        return (allocations, payments) # payments: batches x players, allocation: batch x players x items


class StaticMechanism(Mechanism):
    """ A static mechanism that can be used for testing purposes.
        Items are allocated with probability bid/10, payments are always given
        by b²/20, even when the item is not allocated.

        The expected payoff from this mechanism is thus
        b/10 * v - 0.05b²,
        The optimal strategy fo an agent with quasilinear utility is given by bidding truthfully.
    """

    def __init__(self, cuda: bool = True):
        self.cuda = cuda and torch.cuda.is_available()
        self.device = 'cuda' if self.cuda else 'cpu'

    def run(self, bids):
        assert bids.dim() == 3, "Bid tensor must be 3d (batch x players x items)"
        assert (bids >= 0).all().item(), "All bids must be nonnegative."
        bids = bids.to(self.device)

        payments = torch.mul(bids,bids).mul_(0.05)
        allocations = (bids >= torch.rand_like(bids).mul_(10)).float()

        return (allocations, payments)
