from typing import List

# pylint: disable=E1102
import torch

from .mechanism import Game


class MatrixGame(Game):
    """
    A complete information Matrix game.

    TODO: missing documentation
    """

    # pylint: disable=abstract-method
    def __init__(self, n_players: int, outcomes: torch.Tensor,
                 cuda: bool = True, names: dict = None, validate_inputs: bool = True):
        super().__init__(cuda)
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

        # move to gpu/cpu if needed
        action_profile = action_profile.to(self.device)
        action_profile = action_profile.view(batch_size, n_players)

        # allocation is a dummy and will always be 0 --> all utility is
        # represented by negative payments
        allocations = torch.zeros(batch_size, n_players, n_items, device=self.device)

        # memory allocation and Loop replaced by equivalent vectorized version below:
        # (keep old code as comment for readibility)
        # payments = torch.zeros(batch_size, n_players, device=self.device)
        # for batch in range(batch_size):
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

    def _calculate_utilities_mixed(self, strategy_profile: List[torch.Tensor], player_position=None,
                                   validate: bool = None) -> torch.Tensor:
        """
            Internal function that is wrapped by both play_mixed and calculate_action_values.

            For a given strategy-profile and player_position, calculates that player's expected utility for
            each of their moves. (Only opponent's strategies are considered, player_i's strategy is ignored).
            If no player_position is specified, instead returns the expected utilities for all players in the complete
            strategy profile.

            # TODO: improve this documentation to make this clearer
            --------
            Args:
                strategy_profile: List of mixed strategies (i.e. probability vectors) for all players
                player_position: (optional) int
                    Position of the player of interest, or none if interested in all players

            Returns:
                if player_position (i) is given:
                    torch.Tensor of dimension n_actions_player_i of expected utilities against opponent strategy_profile
                if no player is specified:
                    torch.Tensor of dimension n_players of expected utilities in the complete strategy profile
        """

        # validate inputs if desired
        if validate is None:
            validate = self.validate_inputs
        if validate:
            self._validate_mixed_strategy_input(strategy_profile)

        # Note on implementation:
        # This is implemented via a series of tensor-vector products.
        # self.outcomes is of dim (n_a_p1, ... n_a_pn, n_players)
        # for specific player, we select that player's outcomes in the last dimension,
        #   then perform n-1 tensor-vector-products for each opponent strategy
        # for all player's, we keep the last dimension but perform n tensor-vector products
        #
        # This is implemented as follows:
        #  1  start with outcome matrix
        #  2  define (reverse) order of operations as [i, 1,2,i-1,i+1,...n]
        #  3  permute outcome matrix according to that order, as matrix-vector-matmuls always operate on last dimension
        #  4  perform the operations, starting with the last player
        #
        # For utility of all players, the procedure is the same, except that all player's utilities are kept in the
        # 1st dimension, i.e. the order is [n+1, 1, 2, ... n]

        if player_position is None:
            result = self.outcomes
            ignore_dim = self.n_players
            order = list(range(self.n_players + 1))
        else:
            result = self.outcomes.select(self.n_players, player_position)
            ignore_dim = player_position
            order = list(range(self.n_players))

        # put ignored dimension in the beginning, rest lexicographical
        order = order.pop(ignore_dim), *order
        result = result.permute(order)

        # repeatedly mutliply from the last dimension to the first
        for j in reversed(order):
            if j != ignore_dim:
                result = result.matmul(strategy_profile[j])

        return result

    def calculate_expected_action_payoffs(self, strategy_profile, player_position):
        """
        Calculates the expected utility for a player under a mixed opponent strategy
        ----------
        Args:
            strategy_profile: List of action-probability-vectors for each player. player i's strategy must be supplied
                          but is ignored.
            player_position: player of interest

        Returns:
            expected payoff per action of player i (tensor of dimension (1 x n_actions[i])
        """
        return self._calculate_utilities_mixed(strategy_profile, player_position, validate=False)

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
            (Validation overhead is ~100%, so you might want to turn this off in settings with many many iterations)

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

        payoffs_per_player = self._calculate_utilities_mixed(strategy_profile, validate=validate)

        return torch.tensor([], device=self.device), -payoffs_per_player


class RockPaperScissors(MatrixGame):
    """2 player, 3 action game rock paper scissors"""

    def __init__(self, **kwargs):
        outcomes = torch.tensor([
            # pylint:disable=bad-continuation
            # Col-p: Rock       Paper     Scissors     /  Row-p
            [[0., 0], [-1, 1], [1, -1]],  # Rock
            [[1., -1], [0, 0], [-1, 1]],  # Paper
            [[-1., 1], [1, -1], [0, 0]]  # Scissors
        ])

        names = {
            "player_names": ["RowPlayer", "ColPlayer"],
            "action_names": ["Rock", "Paper", "Scissors"]
        }

        super().__init__(2, outcomes, names=names, **kwargs)


class JordanGame(MatrixGame):
    """Jordan Anticoordination game (1993), FP does not converge. 3P version of Shapley fashion game:
        Player Actions: (Left, Right)
        P1 wants to be different from P2
        P2 wants to be different from P3
        P3 wants to be different from P1
    """

    def __init__(self, **kwargs):
        # pylint:disable=bad-continuation
        outcomes = torch.tensor([
            [[  # LL
                [0.0, 0, 0],  # LLL
                [0, 1, 1]  # LLR
            ], [  # LR
                [1, 1, 0],  # LRL
                [1, 0, 1]  # LRR
            ]], [[  # RL
                [1, 0, 1],  # RLL
                [1, 1, 0]  # RLR
            ], [  # RR
                [0, 1, 1],  # RRL
                [0, 0, 0]  # RRR
            ]]])

        super().__init__(n_players=3, outcomes=outcomes, **kwargs)


class PaulTestGame(MatrixGame):
    """A 3-p game without many symmetries used for testing n-player tensor implementations.
    Payoff: [M,R,C]
    """

    def __init__(self, **kwargs):
        # pylint: disable=bad-continuation
        outcomes = torch.tensor([
            [[  # LL
                [2., 2, 2],  # LLL
                [-1, 1, 9]  # LLR
            ], [  # LR
                [-1, 9, 1],  # LRL
                [4, 3, 3]  # LRR
            ]], [[  # RL
                [1, 2, 2],  # RLL
                [-2, 1, 7]  # RLR
            ], [  # RR
                [-2, 7, 1],  # RRL
                [3, 4, 4]  # RRR
            ]]])

        super().__init__(n_players=3, outcomes=outcomes, **kwargs)


class PrisonersDilemma(MatrixGame):
    """Two player, two action Prisoner's Dilemma game.
       Has a unique pure Nash equilibrium in ap [1,1]
    """

    def __init__(self, **kwargs):
        super().__init__(
            n_players=2,
            outcomes=torch.tensor([[[-1, -1], [-3, 0]], [[0, -3], [-2, -2]]]),
            names={
                "player_names": ["RowPlayer", "ColPlayer"],
                "action_names": ["Cooperate", "Defect"]
            },
            **kwargs
        )


class BattleOfTheSexes(MatrixGame):
    """Two player, two action Battle of the Sexes game"""

    def __init__(self, **kwargs):
        super().__init__(
            n_players=2,
            outcomes=torch.tensor([[[3, 2], [0, 0]], [[0, 0], [2, 3]]]),
            names={
                "player_names": ["Boy", "Girl"],
                "action_names": ["Action", "Romance"]
            },
            **kwargs
        )


class BattleOfTheSexesMod(MatrixGame):
    """Modified Battle of the Sexes game"""

    def __init__(self, **kwargs):
        super().__init__(
            n_players=2,
            outcomes=torch.tensor([
                [  # Him: Stadium
                    [3, 2],  # Her: Stadium
                    [0, 0]],  # Her: Theater
                [  # Him: Theater
                    [0, 0],  # Her: Stadium
                    [2, 3]],  # Her: Theater
                [  # Him: Stadium with friend
                    [-1, 1],  # Her: Stadium
                    [4, 0]],  # Her: Theater
            ]),
            **kwargs
        )


class MatchingPennies(MatrixGame):
    """Two Player, two action Matching Pennies / anticoordination game"""

    def __init__(self, **kwargs):
        super().__init__(
            n_players=2,
            outcomes=torch.tensor([[[1, -1], [-1, 1, ]], [[-1, 1], [1, -1]]]),
            names={
                "player_names": ["Even", "Odd"],
                "action_names": ["Heads", "Tails"]
            },
            **kwargs
        )
