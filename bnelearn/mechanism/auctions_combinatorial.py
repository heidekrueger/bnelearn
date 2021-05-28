
from typing import Tuple

# pylint: disable=E1102
import torch

from .mechanism import Mechanism


class LLGAuction(Mechanism):
    """
        Implements simple auctions in the LLG setting with 3 bidders and
        2 goods.
        Notably, this is not an implementation of a general Combinatorial auction
        and bidders do not submit full bundle (XOR) bids: Rather, it's assumed
        a priori that each bidder bids on a specific bundle:
        The first bidder will only bid on the bundle {1}, the second on {2},
        the third on {1,2}, thus actions are scalar for each bidder.

        For the LLG Domain see e.g. Ausubel & Milgrom 2006 or Bosshard et al 2017
    """

    def __init__(self, rule='first_price', cuda: bool = True):
        super().__init__(cuda)

        if rule not in ['first_price', 'vcg', 'nearest_bid', 'nearest_zero', 'proxy', 'nearest_vcg']:
            raise ValueError('Invalid Pricing rule!')
        # 'nearest_zero' and 'proxy' are aliases
        if rule == 'proxy':
            rule = 'nearest_zero'
        self.rule = rule

    def run(self, bids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Runs a (batch of) LLG Combinatorial auction(s).

        We assume n_players == 3 with 0,1 being local bidders and 3 being the global bidder.

        Parameters
        ----------
        bids: torch.Tensor
            of bids with dimensions (batch_size, n_players, 1)

        Returns
        -------
        (allocation, payments): Tuple[torch.Tensor, torch.Tensor]
            allocation: tensor of dimension (n_batches x n_players x 1),
                        1 indicating the desired bundle is allocated to corresponding player
                        in that batch, 0 otherwise.
                        (i.e. 1 for player0 means {1} allocated, for player2 means {2} allocated,
                        for player3 means {1,2} allocated.)
            payments:   tensor of dimension (n_batches x n_players)
                        Total payment from player to auctioneer for her
                        allocation in that batch.
        """
        assert bids.dim() == 3, "Bid tensor must be 3d (batch x players x 1)"
        assert (bids >= 0).all().item(), "All bids must be nonnegative."
        # name dimensions for readibility
        batch_dim, player_dim, item_dim = 0, 1, 2  # pylint: disable=unused-variable
        batch_size, n_players, n_items = bids.shape

        assert n_items == 1, "invalid bid_dimensionality in LLG setting"  # dummy item is desired bundle for each player

        # move bids to gpu/cpu if necessary, get rid of unused item_dim
        bids = bids.squeeze(item_dim).to(self.device)  # batch_size x n_players
        # individual bids as batch_size x 1 tensors:
        b_locals, b_global = bids[:, :-1], bids[:, [-1]]

        # allocate return variables
        payments = torch.zeros(batch_size, n_players, device=self.device)
        allocations = torch.zeros(batch_size, n_players, n_items, dtype=bool,
                                  device=self.device)

        # Two possible allocations
        allocation_locals = torch.ones(1, n_players, dtype=bool, device=self.device)
        allocation_locals[0, -1] = 0
        allocation_global = torch.zeros(1, n_players, dtype=bool, device=self.device)
        allocation_global[0, -1] = 1

        # 1. Determine efficient allocation
        locals_win = (b_locals.sum(axis=1, keepdim=True) > b_global).float()  # batch_size x 1
        allocations = locals_win * allocation_locals + (1 - locals_win) * allocation_global

        if self.rule == 'first_price':
            payments = allocations * bids  # batch x players
        else:  # calculate local and global winner prices separately
            payments = torch.zeros(batch_size, n_players, device=self.device)
            global_winner_prices = b_locals.sum(axis=1, keepdim=True)  # batch_size x 1
            payments[:, [-1]] = (1 - locals_win) * global_winner_prices

            local_winner_prices = torch.zeros(batch_size, n_players - 1, device=self.device)

            if self.rule in ['vcg', 'nearest_vcg']:
                # vcg prices are needed for vcg, nearest_vcg
                local_vcg_prices = torch.zeros_like(local_winner_prices)

                local_vcg_prices += (
                    b_global - b_locals.sum(axis=1, keepdim=True) + b_locals
                ).relu()

                if self.rule == 'vcg':
                    local_winner_prices = local_vcg_prices
                else:  # nearest_vcg
                    delta = (1/(n_players - 1)) * \
                        (b_global - local_vcg_prices.sum(axis=1, keepdim=True))  # batch_size x 1
                    local_winner_prices = local_vcg_prices + delta  # batch_size x 2

            elif self.rule in ['proxy', 'nearest_zero'] and n_players == 3:
                b1, b2 = b_locals[:, [0]], b_locals[:, [1]]

                # three cases when local bidders win:
                #  1. "both_strong": each local > half of global --> both play same
                #  2. / 3. one player 'weak': weak local player pays her bid, other pays enough to match global
                both_strong = ((b_global <= 2 * b1) & (b_global <= 2 * b2)).float()  # batch_size x 1
                first_weak = (2 * b1 < b_global).float()
                # (second_weak implied otherwise)
                local_prices_case_both_strong = 0.5 * torch.cat(2 * [b_global], dim=player_dim)
                local_prices_case_first_weak = torch.cat([b1, b_global - b1], dim=player_dim)
                local_prices_case_second_weak = torch.cat([b_global - b2, b2], dim=player_dim)

                local_winner_prices = both_strong * local_prices_case_both_strong + \
                                      first_weak * local_prices_case_first_weak + \
                                      (1 - both_strong - first_weak) * local_prices_case_second_weak

            elif self.rule == 'nearest_bid' and n_players == 3:
                b1, b2 = b_locals[:, [0]], b_locals[:, [1]]

                case_1_outbids = (b_global < b1 - b2).float()  # batch_size x 1
                case_2_outbids = (b_global < b2 - b1).float()  # batch_size x 1

                local_prices_case_1 = torch.cat([b_global, torch.zeros_like(b_global)], dim=player_dim)
                local_prices_case_2 = torch.cat([torch.zeros_like(b_global), b_global], dim=player_dim)

                delta = 0.5 * (b1 + b2 - b_global)
                local_prices_else = bids[:, [0, 1]] - delta

                local_winner_prices = case_1_outbids * local_prices_case_1 + \
                    case_2_outbids * local_prices_case_2 + \
                    (1 - case_1_outbids - case_2_outbids) * local_prices_else

            else:
                raise ValueError("invalid bid rule")

            payments[:, :-1] = locals_win * local_winner_prices  # TODO: do we even need this * op?

        return (allocations.unsqueeze(-1), payments)  # payments: batches x players, allocation: batch x players x items

    def get_efficiency(self, env, draw_valuations: bool = False) -> float:
        """LLG auction specific efficiency that uses fact of single-minded
        bidders.
        """
        batch_size = min(env.agents[0].valuations.shape[0], 2 ** 12)

        if draw_valuations:
            env.draw_valuations_()

        bid_profile = torch.zeros(batch_size, env.n_players, 1,
                                  device=self.device)
        for pos, bid in env._generate_agent_actions():
            bid_profile[:, pos, :] = bid[:batch_size, ...]
        allocations, _ = self.play(bid_profile)
        actual_welfare = torch.zeros(batch_size, device=self.device)
        for a in env.agents:
            actual_welfare += a.get_welfare(
                allocations[:batch_size, a.player_position],
                a.valuations[:batch_size, ...]
            )

        local_valuations = torch.zeros_like(actual_welfare)
        for a in env.agents[:-1]:
            local_valuations += a.valuations[:batch_size, ...].squeeze()
        maximum_welfare = torch.max(
            env.agents[-1].valuations[:batch_size, ...].squeeze(), local_valuations
        ).view_as(actual_welfare)

        efficiency = (actual_welfare / maximum_welfare).mean()
        return efficiency
