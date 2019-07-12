# -*- coding: utf-8 -*-

"""
This module implements games such as matrix games and auctions.
"""

from abc import ABC, abstractmethod
from typing import Tuple, List

#pylint: disable=E1102
import torch

# Type declarations
Outcome = Tuple[torch.Tensor, torch.Tensor]

class Game(ABC):
    """
    Base class for any kind of games
    """
    device: str

    @abstractmethod
    def play(self, action_profile):
        """Play the game!"""
        # get actions from players and define outcome
        raise NotImplementedError()

class Mechanism(Game):
    """
    Auction Mechanism - Interpreted as a Bayesian game.
    A Mechanism collects bids from all players, then allocates available
    items as well as payments for each of the players.
    """
    def play(self, action_profile):
        return self.run(bids=action_profile)

    @abstractmethod
    def run(self, bids):
        """Alias for play for auction mechanisms"""
        raise NotImplementedError()

class MatrixGame(Game):
    """
    A complete information Matrix game.

    TODO: missing documentation
    """
    # pylint: disable=abstract-method
    def __init__(self, n_players: int, outcomes: torch.Tensor,
                 cuda: bool = True, names: dict = None, validate_inputs: bool = True):
        self.cuda = cuda and torch.cuda.is_available()
        self.device = 'cuda' if self.cuda else 'cpu'
        self.n_players = n_players
        self.validate_inputs = validate_inputs

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

    def get_player_name(self, player_id: int):
        """Returns readable name of player if provided."""
        if self.names and "players" in self.names.keys():
            return self.names["players"][player_id]

        return player_id

    def get_action_name(self, action_id: int):
        """Currently only works if all players have same action set!"""
        if self.names and "actions" in self.names.keys():
            return self.names["actions"][action_id]

        return action_id

    def _validate_action_input(self, action_profile: torch.Tensor) -> None:
        """Assert validity of a (pure) action profile

        An action profile should have shape of a mechanism (batch x players x items).
        In a matrix game it should therefore be (batch x players x 1).
        TODO: Each player's action should be a valid index for that player.

        Parameters
        ----------
        action_profile: torch.Tensor
        An action profile tensor to be tested.

        Returns
        -------
        (nothing)

        Raises
        ------
        AssertionError on invalid input.
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


    def play(self, action_profile):
        """Plays the game for a given action_profile.

        Parameters
        ----------
        action_profile: torch.Tensor
            Shape: (batch_size, n_players, n_items)
            n_items should be 1 for now. (This might change in the future to represent information sets!)
            Actions should be integer indices. #TODO: Ipmlement that they can also be action names!

            Mixed strategies are NOT allowed as input, sampling should happen in the player class.

        Returns
        -------
        (allocation, payments): Tuple[torch.Tensor, torch.Tensor]
            allocation: tensor of dimension (n_batches x n_players x n_items),
                        In this setting, there's nothing to be allocated, so it will be all zeroes.
            payments:   tensor of dimension (n_batches x n_players)
                        Negative outcome/utility for each player.
        """
        if self.validate_inputs:
            self._validate_action_input(action_profile)

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

    def _validate_mixed_strategy_input(self, strategy_profile: List[torch.Tensor]) -> None:
        """Assert validity of strategy profile

            Parameters
            ----------
            action_profile: torch.Tensor
            An action profile tensor to be tested.

            Returns
            -------
            (nothing)

            Raises
            ------
            AssertionError on invalid input.
        """
        assert len(strategy_profile) == self.n_players, \
            "Invalid number of players in strategy profile!"

        for player, strategy in enumerate(strategy_profile):
            assert strategy.shape == torch.Size([self.outcomes.shape[player]]), \
                "Strategy contains invalid number of actions for player {}".format(player)
            # Check valid probabilities
            assert torch.equal(strategy.sum(), torch.tensor(1.0, device=self.device)), \
                "Probabilities must sum to 1 for player {}".format(player)
            assert torch.all(strategy >= 0.0), \
                "Probabilities must be positive for player {}".format(player)

    def _tensorize_strategy_profile(self, strategy_profile_list = List[torch.Tensor]):
        """Turns a list of strategies (1-d-tensors) into a n-player dimensional joint strategy profile tensor"""

        # set einsum_string depending on number of players
        einsum_strings = {
            1: 'i->i',
            2: 'i,j->ij',
            3: 'i,j,k->ijk',
            4: 'i,j,k,l->ijkl',
            5: 'i,j,k,l,m->ijklm',
            6: 'i,j,k,l,m,n->ijklmn'
        }

        try:
            einsum_string = einsum_strings[self.n_players]
        except KeyError:
            raise NotImplementedError('Playing mixed strategies is only implemented for up to 6 players!')

        return torch.einsum(einsum_string, strategy_profile_list)


    def play_mixed(self, strategy_profile: List[torch.Tensor], validate: bool = None):
        """Plays the game with mixed strategies, returning expectation of outcomes.

        This version does NOT support batches or multiple items, as (1) batches do not make
        sense in this setting since we are already returning expectations.

        Parameters
        ----------
        strategy_profile: List[torch.Tensor]
            A list of strategies for each player. Each element i should be a 1-dimensional
            torch tensor of length n_actions_pi with entries j = P(player i plays action j)

        validate: bool
            Whether to validate inputs. Defaults to setting in game class.
            (You might want to turn this off in settings with many many iterations)

        Returns
        -------
        (allocation, payments): Tuple[torch.Tensor, torch.Tensor]
            allocation: empty tensor of dimension (0) --> not used in this game
            payments:   tensor of dimension (n_players)
                        Negative expected outcome/utility for each player.
        """

        # move inputs to device if necessary
        for i, strat in enumerate(strategy_profile):
            strategy_profile[i] = strat.to(self.device)

        # validate inputs if desired
        if validate is None:
            validate = self.validate_inputs
        if validate:
            self._validate_mixed_strategy_input(strategy_profile)

        # for each action profile, get probability-weighted vector of utilities that the players
        # expect to get from this action profile under the mixed strategy profile
        utils_per_action_profile = (
            # get joint action profile distribution
            # with dim=n_players and shape [n_actions_p1, ... n_actions_p_n])
            # with joint_distribution[i, j, k, ... z] = P(a_i =i, a_j=j, ...z_z=z)
            self._tensorize_strategy_profile(strategy_profile)
            # add another dimension to the end of this tensor to make it same shape of outcome matrix
            .unsqueeze(self.n_players)
            # multiply with outcome for each action profile
            * self.outcomes
        )

        # sum over all but last dimension to get expected payoff per player
        payoffs_per_player = utils_per_action_profile.sum(list(range(self.n_players)))

        return torch.tensor([], device=self.device), -payoffs_per_player

###############################################################################
# Implementations of specific games ###########################################
##############################################################################

class RockPaperScissors(MatrixGame):
    """2 player, 3 action game rock paper scissors"""
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
        #pylint:disable=bad-continuation
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

        # pylint: disable=bad-continuation
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
    """Two player, two action Prisoner's Dilemma game.
       Has a unique pure Nash equilibrium in ap [1,1]
    """
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
    """Two player, two action Battle of the Sexes game"""
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

class BattleOfTheSexes_Mod(MatrixGame):
    """Modified Battle of the Sexes game"""
    def __init__(self, cuda: bool = True):
        super().__init__(
            n_players=2,
            outcomes=torch.tensor([
                [# Him: Stadium
                    [3,2],  # Her: Stadium
                    [0,0]], # Her: Theater
                [# Him: Theater
                    [0,0],  # Her: Stadium
                    [2,3]], # Her: Theater
                [# Him: Stadium with friend
                    [-1,1],  # Her: Stadium
                    [4,0]], # Her: Theater
                ]),
            cuda=cuda
        )

class MatchingPennies(MatrixGame):
    """Two Player, two action Matching Pennies / anticoordination game"""
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
    "Vickrey / Second Price Sealed Bid Auctions"

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

        highest_bids, winning_bidders = bids.max(dim=player_dim, keepdim=True) # shape of each: [batch_size, 1, n_items]

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
    """First Price Sealed Bid auction"""

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

        highest_bids, winning_bidders = bids.max(dim = player_dim, keepdim=True) # both shapes: [batch_size, 1, n_items]

        # replaced by equivalent, faster torch.scatter operation, see below,
        # but keeping nested-loop code for readability
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
    """ A static mechanism that can be used for testing purposes,
        in order to test functionality/efficiency of optimizers without introducing
        additional stochasticity from multi-player learning dynamics.

        In this 'single-player single-item' setting, items are allocated with probability bid/10,
        payments are always given by b²/20, even when the item is not allocated.
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
