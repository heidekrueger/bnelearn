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

class MatrixGame(Game):
    """A complete information Matrix game."""
    # pylint: disable=abstract-method
    def __init__(self, n_players: int, outcomes: torch.Tensor, cuda: bool = True, names: dict = None):
        self.cuda = cuda and torch.cuda.is_available()
        self.device = 'cuda' if self.cuda else 'cpu'
        self.n_players = n_players

        # validate and set outcomes

        self.outcomes = outcomes.float().to(self.device)
        # Outcome tensor should be [n_actions_p1, n_actions_p2, ..., n_actions_p_n, n_players]
        # n_actions_p1 is implicitly defined by outcome tensor. Other dimensions should match
        assert outcomes.dim() == n_players + 1
        assert outcomes.shape[-1] == n_players

        # TODO: validate names. i.e.
        #   * if single list, check if all players have same number of actions
        #   * otherwise, should provide list of lists (names for each player). validate that each list matches length
        self.names = names

    def check_input_validity(self, action_profile):
        """Assert validity of action profile

           An action profile should have shape of a mechanism (batch x players x items).
           In a matrix game it should therefore be (batch x players x 1).
           TODO: Each player's action should be a valid index for that player.
        """

        assert action_profile.dim() == 3, "Bid matrix must be 3d (batch x players x items)"
        assert action_profile.dtype == torch.int64 and torch.all(action_profile >= 0), \
            "Actions must be integer indeces!"


        # pylint: disable=unused-variable
        batch_dim, player_dim, item_dim = 0, 1, 2
        batch_size, n_players, n_items = action_profile.shape

        assert n_items == 1, "only single action per player in matrix game setting"
        assert n_players == self.n_players, "one action per player must be provided"

        for i in range(n_players):
            assert torch.all(action_profile[:, i, :] < self.outcomes.shape[i]), \
                "Invalid action given for player {}".format(i)


    def get_player_name(self, player_id: int):
        if self.names and "players" in self.names.keys():
            return self.names["players"][player_id]
        else:
            return player_id

    def get_action_name(self, action_id: int):
        """Currently only works if all players have same action set!"""
        if self.names and "actions" in self.names.keys():
            return self.names["actions"][action_id]
        else:
            return action_id


    def play(self, action_profile, validate=True):
        """Plays the game for a given action_profile.

        Parameters
        ----------
        action_profile: torch.Tensor
            Shape: (batch_size, n_players, n_items)
            n_items should be 1 for now. (This might change in the future to represent information sets!)
            Actions should be integer indices. #TODO: Ipmlement that they can also be action names!

            Mixed strategies are NOT allowed as input, sampling should happen in the player class.

        validate: bool
            Whether to validate inputs. Default is true.
            (You might want to turn this off in settings with many many iterations)

        Returns
        -------
        (allocation, payments): Tuple[torch.Tensor, torch.Tensor]
            allocation: tensor of dimension (n_batches x n_players x n_items),
                        In this setting, there's nothing to be allocated, so it will be all zeroes.
            payments:   tensor of dimension (n_batches x n_players)
                        Negative outcome/utility for each player.
        """

        if validate:
            self.check_input_validity(action_profile)

        # pylint: disable=unused-variable
        batch_dim, player_dim, item_dim = 0, 1, 2
        batch_size, n_players, n_items = action_profile.shape

        #move to gpu/cpu if needed
        action_profile = action_profile.to(self.device)
        action_profile = action_profile.view(batch_size, n_players)

        # allocation is a dummy and will always be 0 --> all utility is
        # represented by negative payments
        allocations = torch.zeros(batch_size, n_players, n_items, device=self.device)

        ## memory allocation and Loop replaced by equivalent vectorized version below:
        ## (keep old code as comment for readibility)
        #payments = torch.zeros(batch_size, n_players, device=self.device)
        #for batch in range(batch_size):
        #    for player in range(n_players):
        #        payments[batch, player] = -self.outcomes[action[batch, player1], ... action[batch, player_n]][player]

        # payment to "game master" is the negative outcome
        payments = -self.outcomes[
            [
                action_profile[:, i] for i in range(n_players)
            ]].view(batch_size, n_players)

        return (allocations, payments)


class RockPaperScissors(MatrixGame):
    def __init__(self, cuda: bool = True):

        device = 'cuda' if cuda and torch.cuda.is_available() else 'cpu'

        outcomes = torch.tensor([
        # pylint:disable=bad-continuation
        #Col-p: Rock       Paper     Scissors     /  Row-p
            [   [ 0., 0],  [-1, 1],  [ 1,-1]   ], #  Rock
            [   [ 1.,-1],  [ 0, 0],  [-1, 1]   ], #  Paper
            [   [-1., 1],  [ 1,-1],  [ 0, 0]   ] #  Scissors
            ], device = device)

        names = {
            "player_names": ["RowPlayer", "ColPlayer"],
            "action_names": ["Rock", "Paper", "Scissors"]
            }

        super().__init__(2, outcomes, cuda=cuda, names=names)

class JordanGame(MatrixGame):
    """Jordan Anticoordination game (1993), FP does not converge. 3P version of Shapley fashion game:
        Player Actions: (Left, Right)
        P1 wants to be different from P2
        P2 wants to be different from P3
        P3 wants to be different from P1
    """
    def __init__(self, cuda: bool = True):
        device = 'cuda' if cuda and torch.cuda.is_available() else 'cpu'

        outcomes = torch.tensor([
            [   [   #LL
                    [0.0,0,0], # LLL
                    [0,1,1]    # LLR
                ], [#LR
                    [1,1,0],   # LRL
                    [1,0,1]    # LRR
            ]], [[  #RL
                    [1,0,1],   # RLL
                    [1,1,0]    # RLR
                ], [#RR
                    [0,1,1],   # RRL
                    [0,0,0]    # RRR
            ]]], device=device)

        super().__init__(n_players=3, outcomes=outcomes, cuda=cuda)

class PaulTestGame(MatrixGame):
    """A 3-p game without many symmetries used for testing n-player tensor implementations.
    Payoff: [M,R,C]
    """
    def __init__(self, cuda: bool = True):
        device = 'cuda' if cuda and torch.cuda.is_available() else 'cpu'

        outcomes = torch.tensor([
            [   [   #LL
                    [2., 2, 2],  # LLL
                    [-1,1,9]    # LLR
                ], [#LR
                    [-1, 9,1],   # LRL
                    [4, 3, 3]    # LRR
            ]], [[  #RL
                    [1, 2, 2],   # RLL
                    [-2, 1,7]    # RLR
                ], [#RR
                    [-2, 7,1],   # RRL
                    [3, 4, 4]    # RRR
            ]]], device=device)

        super().__init__(n_players=3, outcomes=outcomes, cuda=cuda)

class PrisonersDilemma(MatrixGame):
    def __init__(self, cuda: bool = True):
        super().__init__(
            n_players=2,
            outcomes = torch.tensor([[[-1, -1],[-3, 0]], [[ 0, -3],[-2,-2]]]),
            cuda = cuda,
            names = {
                "player_names": ["RowPlayer", "ColPlayer"],
                "action_names": ["Cooperate", "Defect"]
            }
        )

class BattleOfTheSexes(MatrixGame):
    def __init__(self, cuda: bool = True):
        super().__init__(
            n_players=2,
            outcomes=torch.tensor([[[3, 2],[0,0]], [[0,0],[2,3]]]),
            cuda=cuda,
            names = {
                "player_names": ["Boy", "Girl"],
                "action_names": ["Action", "Romance"]
            }
        )

class MatchingPennies(MatrixGame):
    def __init__(self, cuda: bool = True):
        super().__init__(
            n_players=2,
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
        batch_dim, player_dim, item_dim = 0, 1, 2 #pylint: disable=unused-variable

        bids = bids.to(self.device)

        payments = torch.mul(bids,bids).mul_(0.05).sum(item_dim)
        allocations = (bids >= torch.rand_like(bids).mul_(10)).float()

        return (allocations, payments)
