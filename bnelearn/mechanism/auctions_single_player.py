import torch

from .mechanism import Mechanism


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

    def run(self, bids: torch.Tensor):
        assert bids.dim() == 3, "Bid tensor must be 3d (batch x players x items)"
        assert (bids >= 0).all().item(), "All bids must be nonnegative."
        batch_dim, player_dim, item_dim = 0, 1, 2  # pylint: disable=unused-variable

        payments = torch.mul(bids, bids).mul_(0.05).sum(item_dim)
        allocations = (bids >= torch.rand_like(bids).mul_(10)).float()

        return allocations, payments


class StaticFunctionMechanism(Mechanism):
    """ A static mechanism that can be used for testing purposes,
        in order to test functionality/efficiency of optimizers without introducing
        additional stochasticity from multi-player learning dynamics.
        This function more straightforward than the Static Mechanism above, which has stochasticity similar to an
        auction.

        Instead, this class returns a straight up function, designed such that vanilla PG will also work on it.

        Here, the player gets the item with probability 0.5 and pays (5-b)², i.e. it's optimal to always bid 5.
        The expected utility in optimal strategy is thus 2.5.

    """

    def run(self, bids):
        assert bids.dim() == 3, "Bid tensor must be 3d (batch x players x items)"
        assert (bids >= 0).all().item(), "All bids must be nonnegative."
        batch_dim, player_dim, item_dim = 0, 1, 2  # pylint: disable=unused-variable

        payments = torch.mul(5.0 - bids, 5.0 - bids).sum(item_dim)
        allocations = (torch.rand_like(bids) > 0.5).float()

        return allocations, payments
