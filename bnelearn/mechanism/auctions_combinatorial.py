"""Auction mechanism for combinatorial auctions (where bidders are interested
in bundles of items).
"""
import os
import sys
from typing import Tuple
#from time import perf_counter as timer

# pylint: disable=E1102
import torch
import torch.nn as nn
# For qpth #pylint:disable=ungrouped-imports
from qpth.qp import QPFunction
from tqdm import tqdm
from functools import reduce
from operator import mul

# Some (but not all) of the features in this module need gurobi,
# but we still want to be able to use the other features when gurobi is not
# installed.
try:
    import gurobipy as grb
    GUROBI_AVAILABLE = True
except ImportError as e:
    GUROBI_AVAILABLE = False
    GUROBI_IMPORT_ERROR = e


from bnelearn.mechanism.data import LLGData, LLLLGGData
from bnelearn.util import mpc
from .mechanism import Mechanism


class _OptNet_for_LLLLGG(nn.Module):
    def __init__(self, device, A, beta, b, payment_vcg=None,
                 precision=torch.double, max_iter=20):
        """
        Build basic model
            s.t.
                pA >= beta
                p <= b
            ->
                G =     (-A     )
                    diag(1,...,1)
                h = (-beta)
                    (b    )

        See LLLLGGAuction._calculate_payments_nearest_vcg_core for details on variables.
        """
        # TODO Stefan/Paul: Please provide minimal docstring
        # I think there should be a clearer interface between solving and using
        # it for this specifc problem, e.g., what's the general form of the 
        # problem Anne's solver can tackle?
        self.n_batch, self.n_coalitions, self.n_player = A.shape  # pylint: disable=unused-variable

        # TODO Nils: would it make sense to have an optional consistency check
        # whether or not the dimensions match?
        super().__init__()
        self.device = device
        self.precision = precision
        self.payment_vcg = payment_vcg
        self.max_iter = max_iter

        A = torch.as_tensor(A, dtype=precision, device=self.device)
        b = torch.as_tensor(b, dtype=precision, device=self.device)
        beta = torch.as_tensor(beta, dtype=precision, device=self.device)

        self.G = torch.cat(
            (
                -A,
                torch.eye(self.n_player, dtype=precision, device=self.device) \
                    .repeat(self.n_batch, 1, 1),
                -torch.eye(self.n_player, dtype=precision, device=self.device) \
                    .repeat(self.n_batch, 1, 1)
            ), 1)
        self.h = torch.cat(
            (
                -beta,
                b,
                torch.zeros([self.n_batch, self.n_player], dtype=precision, device=self.device)
            ), 1)
        # will be set by methods
        self.e = None
        self.mu = None
        self.Q = None
        self.q = None

    def _add_objective_min_payments(self):
        """
        Add objective to minimize total payments and solve LP:
        min p x 1

        Q = (0,...,0)
        q = (1,...,1)
        """
        self.Q = torch.diag(
            torch.tensor(
                [1e-5,] * self.n_player,
                dtype=self.precision,
                device=self.device
            )
        ).repeat(self.n_batch, 1, 1)
        self.q = torch.ones([self.n_batch, self.n_player], dtype=self.precision, device=self.device)

    def _add_objective_min_vcg_distance(self, min_payments=None):
        """
        Add objective to minimize euclidean vcg distance QP:
        min (p-p_0)(p-p_0)

        Q = diag(2,...,2)
        q = -2p_0
        """
        if min_payments is not None:
            self.e = torch.ones(
                [self.n_batch, 1, self.n_player],
                dtype=self.precision,
                device=self.device
            )
            self.mu = min_payments.sum(1).reshape(self.n_batch, 1)

        self.Q = torch.diag(
            torch.tensor(
                [2, ] * self.n_player,
                dtype=self.precision,
                device=self.device
            )
        ).repeat(self.n_batch,1,1)
        self.q = -2 * torch.as_tensor(self.payment_vcg, dtype=self.precision, device=self.device)

    def forward(self, solver, input=None):
        """input is not used, as problem is fully specified
        Choose either 'mpc' or 'qpth' solver. The latter is both slower and more imprecise"""

        if solver == 'qpth':
            if self.e is None:
                self.e = torch.zeros(0, dtype=self.precision, device=self.device, requires_grad=True)
            if self.mu is None:
                self.mu = torch.zeros(0, dtype=self.precision, device=self.device, requires_grad=True)
            return QPFunction(verbose=-1, eps=1e-19, maxIter=20, notImprovedLim=10, check_Q_spd=False) \
                             (self.Q, self.q, self.G, self.h, self.e, self.mu)

        elif solver == 'mpc':
            mpc_solver=mpc.MpcSolver(max_iter=self.max_iter)
            # detach all variables to set requires_grad=False
            if self.e is not None:
                self.e_no_grad=self.e.detach()
                self.mu_no_grad=self.mu.detach()
            else:
                self.e_no_grad=None
                self.mu_no_grad=None
            x_mpc, _ = mpc_solver.solve(self.Q.detach(), self.q.detach(), self.G.detach(),
                                        self.h.detach(), self.e_no_grad, self.mu_no_grad,
                                        print_warning=False)
            return x_mpc
        else:
            raise NotImplementedError(":/")


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
        assert bids.dim() >= 3, "Bid tensor must be at least 3d (*batch_dims x players x items)"
        assert (bids >= 0).all().item(), "All bids must be nonnegative."

        # move bids to gpu/cpu if necessary
        bids = bids.to(self.device)

        # name dimensions
        *batch_dims, player_dim, item_dim = range(bids.dim())  # pylint: disable=unused-variable
        *batch_sizes, n_players, n_items = bids.shape

        assert n_items == 1, "invalid bid_dimensionality in LLG setting"  # dummy item is desired bundle for each player

        # move bids to gpu/cpu if necessary, get rid of unused item_dim
        bids = bids.squeeze(item_dim).to(self.device)  # batch_size x n_players
        # individual bids as *batch_sizes x 1 tensors:
        b_locals, b_global = bids[..., :-1], bids[..., [-1]]

        # NOTE: payments and allocations below will have the following dims and dtypes:
        # payments = torch.zeros(*batch_sizes, n_players, device=self.device)
        # allocations = torch.zeros(*batch_sizes, n_players, n_items, dtype=bool, device=self.device)

        # Two possible allocations
        allocation_locals = torch.ones(1, n_players, dtype=bool, device=self.device)
        allocation_locals[0, -1] = 0
        allocation_global = torch.zeros(1, n_players, dtype=bool, device=self.device)
        allocation_global[0, -1] = 1

        # 1. Determine efficient allocation
        locals_win = (b_locals.sum(axis=player_dim, keepdim=True) > b_global).float()  # batch_sizes x 1
        allocations = locals_win * allocation_locals + (1 - locals_win) * allocation_global

        if self.rule == 'first_price':
            payments = allocations * bids  # batch x players
        else:  # calculate local and global winner prices separately
            payments = torch.zeros(*batch_sizes, n_players, device=self.device)
            global_winner_prices = b_locals.sum(axis=player_dim, keepdim=True)  # batch_size x 1
            payments[..., [-1]] = (1 - locals_win) * global_winner_prices

            local_winner_prices = torch.zeros(*batch_sizes, n_players - 1, device=self.device)

            if self.rule in ['vcg', 'nearest_vcg']:
                # vcg prices are needed for vcg, nearest_vcg
                local_vcg_prices = torch.zeros_like(local_winner_prices)

                local_vcg_prices += (
                    b_global - b_locals.sum(axis=player_dim, keepdim=True) + b_locals
                ).relu()

                if self.rule == 'vcg':
                    local_winner_prices = local_vcg_prices
                else:  # nearest_vcg
                    delta = (1/(n_players - 1)) * \
                        (b_global - local_vcg_prices.sum(axis=player_dim, keepdim=True))  # *batch_sizes x 1
                    local_winner_prices = local_vcg_prices + delta  # batch_size x 2

            elif self.rule in ['proxy', 'nearest_zero'] and n_players == 3:
                b1, b2 = b_locals[..., [0]], b_locals[..., [1]]

                # three cases when local bidders win:
                #  1. "both_strong": each local > half of global --> both play same
                #  2. / 3. one player 'weak': weak local player pays her bid, other pays enough to match global
                both_strong = ((b_global <= 2 * b1) & (b_global <= 2 * b2)).float()  # *batch_sizes x 1
                first_weak = (2 * b1 < b_global).float()
                # (second_weak implied otherwise)
                local_prices_case_both_strong = 0.5 * torch.cat(2 * [b_global], dim=player_dim)
                local_prices_case_first_weak = torch.cat([b1, b_global - b1], dim=player_dim)
                local_prices_case_second_weak = torch.cat([b_global - b2, b2], dim=player_dim)

                local_winner_prices = both_strong * local_prices_case_both_strong + \
                                      first_weak * local_prices_case_first_weak + \
                                      (1 - both_strong - first_weak) * local_prices_case_second_weak

            elif self.rule == 'nearest_bid' and n_players == 3:
                b1, b2 = b_locals[..., [0]], b_locals[..., [1]]

                case_1_outbids = (b_global < b1 - b2).float()  # batch_size x 1
                case_2_outbids = (b_global < b2 - b1).float()  # batch_size x 1

                local_prices_case_1 = torch.cat([b_global, torch.zeros_like(b_global)], dim=player_dim)
                local_prices_case_2 = torch.cat([torch.zeros_like(b_global), b_global], dim=player_dim)

                delta = 0.5 * (b1 + b2 - b_global)
                local_prices_else = bids[..., [0, 1]] - delta

                local_winner_prices = case_1_outbids * local_prices_case_1 + \
                    case_2_outbids * local_prices_case_2 + \
                    (1 - case_1_outbids - case_2_outbids) * local_prices_else

            else:
                raise ValueError("invalid bid rule")

            payments[..., :-1] = locals_win * local_winner_prices  # TODO: do we even need this * op?

        return (allocations.unsqueeze(-1), payments)  # payments: batches x players, allocation: batch x players x items


class LLGFullAuction(Mechanism):
    """Implements auctions in the LLG setting with 3 bidders and 2 goods.

    Here, bidders do submit full bundle (XOR) bids. For this specific LLG
    domain see Beck & Ott 2013.

    Item dim 0 corresponds to item A, dim 1 to item B and dim 2 to the bundle
    of both.

    """
    def __init__(self, rule='first_price', cuda: bool=True):
        super().__init__(cuda)

        if rule not in ['first_price', 'vcg', 'nearest_vcg', 'mrcs_favored']:
            raise ValueError('Invalid Pricing rule!')
        self.rule = rule

        self.subsolutions = torch.tensor(
            LLGData.legal_allocations_sparse,
            device=self.device
        )
        self.n_subsolutions = self.subsolutions[-1][0] + 1
        self.solver_max_iter = 20

    def run(self, bids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Runs a batch of LLG Combinatorial auctions.

        We assume n_players == 3 with 0, 1 being local bidders and 3 being the
        global bidder.

        Args:
            bids (:obj:`torch.Tensor`): of bids with dimensions (*batch_sizes,
                n_players, 3).

        Returns:
            (allocation, payments) (:obj:`tuple` of :obj:`torch.Tensor`):
                allocation: tensor of dimension (*batche_sizes x n_players x 3)
                payments: tensor of dimension (*batch_sizes x n_players)

        """
        assert bids.dim() >= 3, "Bid tensor must be at least 3d (*batches x players x 3)"
        assert (bids >= 0).all().item(), "All bids must be nonnegative."

        # name dimensions for readibility
        # pylint: disable=unused-variable
        batch_dim, player_dim, item_dim = 0, 1, 2
        *batch_sizes, n_players, n_items = bids.shape

        assert n_players == 3, "invalid n_players in full LLG setting"
        assert n_items == 3, "invalid bid_dimensionality in full LLG setting"

        # move bids to gpu/cpu if necessary
        bids = bids.to(self.device)
        bids_flat = bids.view(-1, n_players, n_items)

        # 1. Determine allocations
        if self.rule == 'mrcs_favored':
            # # Don't allow higher bids on single-items than bundle
            # # (-> guarantees the existence of equilibria in undominated strategies)
            # err = torch.logical_or(bids[:, :, 0] > bids[:, :, 2],
            #                        bids[:, :, 1] > bids[:, :, 2])
            # bids[err, :] = 0  # don't accept any of the bids of the violating bidders
            allocations = self._solve_allocation_problem(
                bids_flat, dont_allocate_to_zero_bid=False
            )
        else:
            allocations = self._solve_allocation_problem(bids_flat)

        # 2. Determine payments
        if self.rule == 'first_price':
            payments = self._calculate_payments_first_price(bids_flat, allocations)

        elif self.rule == 'vcg':
            payments = self._calculate_payments_vcg(bids_flat, allocations)

        elif self.rule == 'nearest_vcg':
            payments = self._calculate_payments_core(bids_flat, allocations)

        elif self.rule == 'mrcs_favored':
            payments = self._calculate_payments_core(
                bids_flat, allocations, core_selection='mrcs_favored'
            )

        else:
            raise NotImplementedError()

        # allocations: batch x players x items, payments: batches x players
        return (allocations.view_as(bids), payments.view(*batch_sizes, n_players))

    def _solve_allocation_problem(
            self,
            bids: torch.Tensor,
            dont_allocate_to_zero_bid: bool = True
        ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute allocation and welfare

        Args:
            bids: torch.Tensor of bids with dimensions (batch_size, n_players,
                n_bids), values = [0, Inf].
            dont_allocate_to_zero_bid: bool, whether to allocate items to zero
                bids or not.

        Returns:
            allocation: tensor of dimension (n_batches x n_players x 3)
                values = {0, 1}.

        """
        allocations = torch.zeros_like(bids, dtype=torch.int8)

        max_individually = bids[:, :, :2].max(axis=1)
        max_bundle = bids[:, :, 2].max(dim=1)

        # tie-brake #1: prefer assignment with the maximal number of bidders
        individually = \
            max_individually.values.sum(dim=1) >= max_bundle.values

        # assign individual items
        allocations_individual = max_individually.indices[individually, :]
        # item A
        allocations[individually, allocations_individual[:, 0], 0] = 1
        # item B
        allocations[individually, allocations_individual[:, 1], 1] = 1

        # assign bundle
        individually = torch.logical_not(individually)
        allocations_bundle = max_bundle.indices[individually]
        allocations[individually, allocations_bundle, 2] = 1

        # tie-break #2: choose assignments in which bidder 1 wins package A.
        if self.rule == 'mrcs_favored':
            # only handle the relevant tie-break of zero bids
            mask = bids.sum(axis=[1, 2]) == 0
            allocations[mask, 0, 0] = 1
            allocations[mask, 1:, 0] = 0

        if dont_allocate_to_zero_bid:
            allocations *= bids > 0

        return allocations.view_as(bids)

    def _calculate_payments_first_price(
            self,
            bids: torch.Tensor,
            allocations: torch.Tensor
        ) -> torch.Tensor:
        """Compute first prices

        Args:
            bids: torch.Tensor of bids with dimensions (batch_size,
                n_players, n_bids), values in [0,Inf].
            allocations: torch.Tensor of dim (batch_size, b_bundles),
                values = {0, 1}.

        Returns:
            payments: torch.Tensor, dim (batch_size, n_bidders).

        """
        return (allocations * bids).sum(dim=2)

    def _calculate_payments_vcg(
            self,
            bids: torch.Tensor,
            allocations: torch.Tensor
        ) -> torch.Tensor:
        """Computes VCG prices

        Args:
            bids: torch.Tensor, dims (batch_size, n_players, n_bids),
                values = [0, Inf].
            allocations: torch.Tensor, dims (batch_size, b_bundles),
                values = {0, 1}.

        Returns:
            payments: torch.Tensor, dim (batch_size, n_bidders),
                values = [0, Inf].

        """
        n_batch, n_players, _ = bids.shape
        vcg_payments = torch.zeros(n_batch, n_players, device=self.device)
        for player_position in range(n_players):
            bids_reduced = bids.clone()
            bids_reduced[:, player_position] = 0
            optimal_welfare_wo_current = self._calculate_welfare(
                valuations=bids_reduced,
                allocations=self._solve_allocation_problem(bids_reduced),
                # exclude=[player_position] -> should be zero anyway
            )
            actual_welfare_wo_current = self._calculate_welfare(
                valuations=bids, allocations=allocations, exclude=[player_position])
            vcg_payments[:, player_position] = optimal_welfare_wo_current \
                - actual_welfare_wo_current
        return vcg_payments

    def _calculate_payments_core(
            self,
            bids: torch.Tensor,
            allocations: torch.Tensor,
            core_selection: str='nearest_vcg'
        ) -> torch.Tensor:
        n_batch, n_player, n_bundle = bids.shape

        # Generate dense tensor of subsolutions
        subsolutions_dense = torch.sparse.FloatTensor(
            self.subsolutions.t(),
            torch.ones(len(self.subsolutions), device=self.device),
            torch.Size(
                [self.n_subsolutions, n_player * n_bundle],
                device=self.device
            )
        ).to_dense()

        # Compute beta
        coalition_willing_to_pay = torch.mm(
            bids.view(n_batch, n_player * n_bundle),
            subsolutions_dense.t()
        )

        # For b_j(S_j) we need to consider the actual winning bid of j.
        # Therefore, we adjust the coalition and set 1 for each bundle of j
        winning_and_in_coalition = torch.einsum(
            'ij,kjl->kijl',
            subsolutions_dense.view(self.n_subsolutions, n_player, n_bundle) \
                .bool().any(axis=2).float(),
            allocations.view(n_batch, n_player, n_bundle)
        ).view(n_batch, self.n_subsolutions, n_player * n_bundle)

        coalition_already_getting = torch.bmm(
            bids.view(n_batch, 1, n_player * n_bundle),
            winning_and_in_coalition.permute(0, 2, 1)
        ).reshape(n_batch, self.n_subsolutions)

        beta = coalition_willing_to_pay - coalition_already_getting

        # Fixing numerical imprecision
        beta[beta < 1e-6] = 0

        assert beta.shape == (n_batch, self.n_subsolutions), \
            "beta has the wrong shape"

        A = allocations.view(n_batch, 1, n_player * n_bundle) \
            - winning_and_in_coalition
        A = A.view(n_batch, self.n_subsolutions, n_player, n_bundle) \
            .bool().any(axis=3)

        # Computing b
        b = torch.sum(allocations.view(n_batch, n_player, n_bundle) \
            * bids.view(n_batch, n_player, n_bundle), dim=2)

        # Calculate VCG payments
        payments_vcg = self._calculate_payments_vcg(
            bids=bids, allocations=allocations,
        ).clone()

        # Payment rule from Ott & Beck
        if core_selection == 'mrcs_favored':
            # Force agent 1 to have VCG prices: plug her prices into constraints
            beta -= torch.einsum('ij,i->ij', A[:, :, 1], payments_vcg[:, 1])
            A = A[:, :, [0, 2]]
            b = b[:, [0, 2]]

            payment = torch.zeros(n_batch, n_player, device=self.device)
            payment[:, 1] = payments_vcg[:, 1].clone()

            tight = A.sum(axis=2)  # (batch x coaltions) sum of winners of these two
            non_zero_mask = tight.sum(axis=1) > 0

            try:
                # For some reason `torch.max` can't handle empty tensors:
                # https://github.com/pytorch/pytorch/issues/34907

                # Constant constraints (p1 or p2 equals 0)
                c1 = A[non_zero_mask, :, 0] == 1
                c3 = A[non_zero_mask, :, 1] == 1
                beta_temp = beta[non_zero_mask, :].clone()
                beta_temp[torch.logical_or(torch.logical_not(c1), c3)] = 0
                payment[non_zero_mask, 0] = torch.max(beta_temp, axis=1).values
                beta_temp = beta[non_zero_mask, :].clone()
                beta_temp[torch.logical_or(c1, torch.logical_not(c3))] = 0
                payment[non_zero_mask, 2] = torch.max(beta_temp, axis=1).values

                # Diagonal constraints (p1 and p2 do not equal 0)
                diag_batch_mask = torch.any(tight==2, axis=1)
                p_alter = torch.max(beta, axis=1).values / 2
                diag_tight_mask = torch.logical_and(
                    p_alter > payment[:, 0],
                    p_alter > payment[:, 2]
                )
                mask = torch.logical_and(diag_batch_mask, diag_tight_mask)
                payment[mask, 0] = p_alter[mask]
                payment[mask, 2] = p_alter[mask]
            except RuntimeError:
                pass

        else:
            payment = self._run_batch_core_solver(
                A=A, beta=beta, payments_vcg=payments_vcg, b=b,
                min_distance_to_vcg=core_selection=='nearest_vcg'
            )
            payment = payment.view(n_batch, n_player).float()

        return payment

    def _run_batch_core_solver(self, A, beta, payments_vcg, b,
                               min_distance_to_vcg=True):
        model = _OptNet_for_LLLLGG(self.device, A, beta, b, payments_vcg,
                                   max_iter=self.solver_max_iter)
        model._add_objective_min_payments()  # pylint: disable=protected-access
        if min_distance_to_vcg:
            mu = model('mpc')
            model._add_objective_min_vcg_distance(mu)  # pylint: disable=protected-access
        return model('mpc')

    @staticmethod
    # pylint: disable=dangerous-default-value
    def _calculate_welfare(
            valuations: torch.tensor,
            allocations: torch.tensor,
            exclude: list=[]
        ) -> torch.tensor:
        """Calculate total welfare of players excluding a given set.

        Arguments:
            valuations: torch.tensor.
            allocations: torch.tensor.
            exclude: list=None.

        Returns:
            welfare: torch.Tensor, dims (batch_size), values = [0, Inf].

        """
        _, player_dim, item_dim = 0, 1, 2
        # welfare per batch and per player (reduced all items)
        welfare = (valuations * allocations).sum(axis=item_dim)
        # exclude players and sum over remaining
        welfare[:, exclude] = 0
        return welfare.sum(axis=player_dim)


class LLLLGGAuction(Mechanism):
    """
    Inspired by implementation of Seuken Paper (Bosshard et al. (2019), https://arxiv.org/abs/1812.01955).
    Hard coded possible solutions for faster batch computations.

    Args:
        rule: pricing rule
        core_solver: which solver to use, only relevant if pricing rule is a core rule
        parallel: number of processors for parallelization in gurobi (only)
    """

    def __init__(self, rule='first_price', core_solver='NoCore', parallel: int = 1, cuda: bool = True):
        super().__init__(cuda)

        if rule not in ['nearest_vcg', 'vcg', 'first_price']:
            raise ValueError('Invalid pricing rule.')

        if rule == 'nearest_vcg':
            if core_solver not in ['gurobi', 'cvxpy', 'qpth', 'mpc']:
                raise ValueError('Invalid solver.')
        if core_solver == 'gurobi':
            assert GUROBI_AVAILABLE, "You have selected the gurobi solver, but gurobipy is not installed!"
        self.rule = rule


        self.n_items = 8
        self.n_bidders = 6
        # number of bundles that each bidder is interested in
        self.action_size = 2
        # total number of bundles
        self.n_bundles = LLLLGGData.n_bundles # = 12
        assert self.n_bundles == self.n_bidders * self.action_size
        self.n_legal_allocations = LLLLGGData.n_legal_allocations # = 66

        self.core_solver = core_solver
        self.parallel = parallel

        # When using cpu-multiprocessing for the solver, self cannot have 
        # members allocated on cuda, or multiprocessing will fail.
        # In that case, we initiate members on 'cpu' even when `self.device=='cuda'`.
        # This will cost us a few copy operations, but we'll be bottlenecked by 
        # the solver anyway.
        self._solver_device = self.device
        if (parallel > 1 and core_solver == 'gurobi'):
            self._solver_device = 'cpu'

        # all feasible allocations as a dense tensor
        self.legal_allocations = LLLLGGData.legal_allocations_dense(device=self._solver_device)
        assert len(self.legal_allocations) == self.n_legal_allocations
        # subset of all feasible allocations that might be efficient (i.e. bidder optimal)
        self.candidate_solutions = LLLLGGData.efficient_allocations_dense(device=self._solver_device)
        self.player_bundles = LLLLGGData.player_bundles(device=self._solver_device)
        assert self.player_bundles.shape == torch.Size([self.n_bidders, self.action_size])

    def __mute(self):
        """suppresses stdout output from multiprocessing workers
        (e.g. avoid gurobi startup licence message clutter)"""
        sys.stdout = open(os.devnull, 'w')

    def _solve_allocation_problem(self, bids: torch.Tensor, dont_allocate_to_zero_bid=True):
        """
        Computes allocation and welfare.

        To do so, we enumerate all (possibly efficient) candidate solutions and find 
        the one with highest utility.

        Args:
            bids: torch.Tensor
                of bids with dimensions (batch_size, n_players=6, n_bids=2), values = [0,Inf]
            solutions: torch.Tensor
                of possible allocations.

        Returns:
            allocation: torch.Tensor, dims (batch_size, b_bundles = 18), values = {0,1}
            welfare: torch.Tensor, dims (batch_size), values = [0, Inf]

        """
        #candidate_solutions might be on solver device that is different from self_device
        solutions = self.candidate_solutions.to(self.device)

        *batch_sizes, n_players, n_bundles = bids.shape
        bids_flat = bids.view(reduce(mul, batch_sizes, 1), n_players * n_bundles)
        solutions_welfare = torch.mm(bids_flat, torch.transpose(solutions, 0, 1))
        welfare, solution = torch.max(solutions_welfare, dim=1)  # maximizes over all possible allocations
        winning_bundles = solutions.index_select(0, solution)
        if dont_allocate_to_zero_bid:
            winning_bundles = winning_bundles * (bids_flat > 0)

        return winning_bundles, welfare

    def _calculate_payments_first_price(self, bids: torch.Tensor, allocation: torch.Tensor):
        """
        Computes first prices

        Args:
            bids: torch.Tensor
                of bids with dimensions (batch_size, n_players=6, nbids=2), values in [0,Inf]
            allocation: torch.Tensor of dim (batch_size, b_bundles = 18), values = {0,1}

        Returns:
            payments: torch.Tensor, dim (batch_size, n_bidders)
        """
        n_batch, n_players, n_bundles = bids.shape
        return (allocation.view(n_batch, n_players, n_bundles) * bids).sum(dim=2)

    def _calculate_payments_vcg(self, bids: torch.Tensor, allocation: torch.Tensor, welfare: torch.Tensor):
        """
        Computes VCG prices

        Args:
            bids: torch.Tensor, dims (batch_size, n_players=6, n_bids=2), values = [0,Inf]
            allocation: torch.Tensor, dims (batch_size, b_bundles = 18), values = {0,1}
            welfare: torch.Tensor, dims (batch_size), values = [0, Inf]

        Returns:
            payments: torch.Tensor, dim (batch_size, n_bidders), values = [0, Inf]

        """
        player_bundles = self.player_bundles.to(self.device)

        n_batch, n_players, n_bundles = bids.shape
        bids_flat = bids.view(n_batch, n_players * n_bundles)
        vcg_payments = torch.zeros(n_batch, n_players, device=self.device)
        val = torch.zeros(n_batch, n_players, device=self.device)
        for bidder in range(n_players):
            bids_clone = bids.clone()
            bids_clone[:, bidder] = 0
            bidder_bundles = player_bundles[bidder]
            val[:, bidder] = torch.sum(
                bids_flat.index_select(1, bidder_bundles) * allocation.index_select(1, bidder_bundles),
                dim=1, keepdim=True).view(-1)
            vcg_payments[:, bidder] = val[:, bidder] - (welfare - self._solve_allocation_problem(bids_clone)[1])

        return vcg_payments

    def _calculate_payments_nearest_vcg_core(self, bids: torch.Tensor, allocation: torch.Tensor, welfare: torch.Tensor):
        """
        [Nearest VCG core payments by Day and Crampton (2012)]
        (ftp://www.cramton.umd.edu/papers2005-2009/day-cramton-core-payments-for-combinatorial-auctions.pdf)

        Instead of computing all possible coalitions, or the most blocking
        respectively, we iterate through all possible subsolutions, containing
        all possible coalitions. We minimize the prices and solve the LP:
            mu = min p1
                s.t.
                pA >= beta
                p <= b

        and after, we minimize the deviation from VCG and solve the QP:
            min (p-p_0)(p-p_0)
                s.t.
                pA >= beta
                p <= b
                p1 == mu
        ------
        p_0: (parameter) VCG payments
        p: (variable) Core Payments
        ---
        beta: (parameter) coalition's willingness to pay
        beta = welfare(coalition) - sum_(j \\in coalition){b_j(S_j)}
            \\forall coalitions in subsolutions
        with b_j(S_j) being the bid of the actual allocation (their willingness
        to pay for what they already get).
        ---
        A: (parameter) winning and not in coalition (1, else 0)
        b: (parameter) bid of winning bidders (0 if non winning)
        """
        n_batch, n_player, n_bundle = bids.shape
        
        # subsolutions might be on solver_device rather than self.device!
        subsolutions = self.legal_allocations.to(self.device)
        n_subsolutions = self.n_legal_allocations # = 66
        # Compute beta
        coalition_willing_to_pay = torch.mm(
            bids.view(n_batch, n_player * n_bundle),
            subsolutions.t())

        # For b_j(S_j) we need to consider the actual winning bid of j.
        # Therefore, we adjust the coalition and set 1 for each bundle of j
        winning_and_in_coalition = torch.einsum(
            'ij,kjl->kijl',
            subsolutions.view(n_subsolutions, n_player, n_bundle).sum(dim=2),
            allocation.view(n_batch, n_player, n_bundle)).view(n_batch, n_subsolutions, n_player * n_bundle)

        coalition_already_getting = torch.bmm(
            bids.view(n_batch, 1, n_player * n_bundle),
            winning_and_in_coalition.permute(0, 2, 1)).reshape(n_batch, n_subsolutions)

        beta = coalition_willing_to_pay - coalition_already_getting
        # Fixing numerical imprecision (as occured before!)
        beta[beta < 1e-6] = 0

        assert beta.shape == (n_batch, n_subsolutions), "beta has the wrong shape"

        A = allocation.view(n_batch, 1, n_player * n_bundle) - winning_and_in_coalition
        A = A.view(n_batch, n_subsolutions, n_player, n_bundle).sum(dim=3)

        # Computing b
        b = torch.sum(allocation.view(n_batch, n_player, n_bundle) * bids.view(n_batch, n_player, n_bundle), dim=2)
        payments_vcg = self._calculate_payments_vcg(bids, allocation, welfare)

        # Reduce problem and only keep highest value for a coalition
        A, beta = self._reduce_nearest_vcg_remove_duplicates(A, beta)
        # Not efficient. Takes longer than the speedup it results in
        #A, beta = self._reduce_nearest_vcg_remove_zeros(A, beta)

        # Choose core solver
        if self.core_solver == 'gurobi':
            payment = self._run_batch_nearest_vcg_core_gurobi(A, beta, payments_vcg, b)
        elif self.core_solver == 'cvxpy':
            payment = self._run_batch_nearest_vcg_core_cvxpy(A, beta, payments_vcg, b)
        elif self.core_solver == 'qpth' or self.core_solver == 'mpc':
            payment = self._run_batch_nearest_vcg_core_qpth_mpc(A, beta, payments_vcg, b,self.core_solver).squeeze()
        else:
            raise NotImplementedError(":/")
        return payment

    def _reduce_nearest_vcg_remove_duplicates(self, A, beta):
        """
        For each coalition keep only the instance with the highest bid (beta)
        """
        #start_time = timer()
        n_batch, n_coalition, _ = A.shape

        ## Phase 1: For each coalition duplicates, find the max bid
        # Get identical coalitions s.t. dimension are kept over all batches
        A_unique, A_unique_idx = A.unique(sorted=False,dim=1,return_inverse=True)
        # Sort coalition bids decreasing per batch
        beta_sort, beta_sort_idx = beta.squeeze().sort(descending=True)
        # Sort the A unique indexing (matching unique to original) decreasing by beta -> coalitions with highest bid up
        # And: Add very small number increasing by index to the A_unique_sort to prevent random sorting and keep order in beta
        A_unique_idx_sorted_by_beta = A_unique_idx[beta_sort_idx] + torch.linspace(0.00001,0.9,n_coalition,device=self.device)
        # Sort A_unique_idx_sorted_by_beta and beta_sort by groups now in increasing order
        A_unique_idx_sorted_complete, A_unique_idx_sorted_complete_idx = A_unique_idx_sorted_by_beta.view(n_batch,n_coalition).sort(dim = 1, descending=False)
        A_unique_idx_sorted_complete = A_unique_idx_sorted_complete.type(torch.int)
        beta_sort_complete = torch.gather(beta_sort.view(n_batch,n_coalition),1,A_unique_idx_sorted_complete_idx.view(n_batch,n_coalition))
        
        ## Phase 2: Keep only the coalition duplicate with max bid
        # Create tensor to select only the first of a group
        tmp_select_first = torch.zeros((n_batch,n_coalition), dtype=int, device=self.device)
        tmp_select_first[:,0] = -1
        tmp_select_first[:,1:] = A_unique_idx_sorted_complete[:,0:(n_coalition-1)]
        tmp_select_first = (A_unique_idx_sorted_complete - tmp_select_first) \
            .to(dtype=torch.bool, device=self.device)

        ## Phase 3: Select only the highest betas for the groups in A unique
        beta_final = torch.masked_select(beta_sort_complete,tmp_select_first).view(n_batch,max(tmp_select_first.sum(1)))

        #print("Removed {} redundand constraints in {:0.2f} seconds".format((A.shape[1]-A_unique.shape[1]), (timer() - start_time)))
        return A_unique, beta_final

    def _reduce_nearest_vcg_remove_zeros(self, A, beta):
        """
        Remove coalitions that pay no extra (beta <= 0)
        """
        #start_time = timer()
        n_batch, n_coalition, n_player = A.shape

        min_true = min((beta <= 0).sum(1))
        remove = torch.topk(beta.to(torch.float32), min_true, dim = 1, sorted=False, largest=False).indices
        keep = torch.ones((n_batch, n_coalition), device = self.device, dtype=bool).scatter_(1,remove,False)
        keep2 = torch.stack([keep]*n_player,2)
        beta_non_zero = beta.masked_select(keep).view(n_batch, n_coalition-min_true)
        A_non_zero = A.masked_select(keep2).view(n_batch, n_coalition-min_true, n_player)

        #print("Removed {} zero constraints in {:0.2f} seconds".format(min_true,(timer() - start_time)))
        return A_non_zero, beta_non_zero

    def _run_batch_nearest_vcg_core_gurobi(self, A, beta, payments_vcg, b):
        n_batch, n_coalitions, n_player = A.shape
        # Change this to 2**x to solve larger problems at once (optimal ~x=4?)
        gurobi_mini_batch_size = min(n_batch, 2 ** 3)
        n_mini_batch = (int)(n_batch / gurobi_mini_batch_size)
        pool_size = min(self.parallel, n_batch)

        assert n_batch % gurobi_mini_batch_size == 0, \
            "gurobi_mini_batch_size must be picked such that n_batch can be divided without rest"
        A = A.reshape(n_mini_batch, gurobi_mini_batch_size, n_coalitions, n_player)
        beta = beta.reshape(n_mini_batch, gurobi_mini_batch_size, n_coalitions)
        payments_vcg = payments_vcg.reshape(n_mini_batch, gurobi_mini_batch_size, n_player)
        b = b.reshape(n_mini_batch, gurobi_mini_batch_size, n_player)

        # parallel version
        if pool_size > 1:
            iterator_A = A.detach().cpu()
            iterator_beta = beta.detach().cpu()
            iterator_payment_vcg = payments_vcg.detach().cpu()
            iterator_b = b.detach().cpu()

            iterable_input = zip(iterator_A,
                                 iterator_beta, iterator_payment_vcg, iterator_b)

            chunksize = 1
            n_chunks = n_mini_batch / chunksize
            torch.multiprocessing.set_sharing_strategy('file_system')
            with torch.multiprocessing.Pool(pool_size, initializer=self.__mute) as p:  #
                result = list(tqdm(
                    p.imap(self._run_single_mini_batch_nearest_vcg_core_gurobi, iterable_input, chunksize=chunksize),
                    total=n_chunks, unit='chunks',
                    desc='Solving mechanism for batch_size {} with {} processes, chunk size of {}'.format(
                        n_chunks, pool_size, chunksize)
                ))
                p.close()
                p.join()
            payment = torch.cat(result)
        else:
            iterator_A = A.detach()
            iterator_beta = beta.detach()
            iterator_payment_vcg = payments_vcg.detach()
            iterator_b = b.detach()

            iterable_input = zip(iterator_A, iterator_beta, iterator_payment_vcg, iterator_b)

            payment = torch.cat(list(map(self._run_single_mini_batch_nearest_vcg_core_gurobi, iterable_input)))
        return payment

    def _run_single_mini_batch_nearest_vcg_core_gurobi(self, param, min_core_payments=True):
        # Set this true to make sure payments are minimized (correct day and crampton (2012) way)
        A = param[0]
        beta = param[1]
        payments_vcg = param[2]
        b = param[3]

        n_mini_batch, _, n_player = A.shape
        # init model
        # TODO: Speedup the model creation! Can we reuse a model?
        m = self._setup_init_model(A, beta, b, n_mini_batch, n_player)

        if min_core_payments:
            # setup and solve minimizing payments
            m.setParam('FeasibilityTol', 1e-9)
            m.setParam('MIPGap', 1e-9)
            m.setParam('OutputFlag', 0)
            m, mu = self._add_objective_min_payments_and_solve(m, n_mini_batch, n_player, False)
            # add minimal payments constraint to minimizing vcg distance problem
            m = self._add_constraint_min_payments(m, mu, n_mini_batch, n_player)

        # setup and solve minimizing vcg distance
        m.setParam('FeasibilityTol', 1e-9)
        m.setParam('MIPGap', 1e-9)
        payments = self._add_objective_min_vcg_distance_and_solve(m, payments_vcg, n_mini_batch, n_player, False)

        return payments

    def _setup_init_model(self, A, beta, b, n_mini_batch, n_player):
        # Begin QP
        m = grb.Model()
        m.setParam('OutputFlag', 0)

        payment = {}
        for batch_k in range(n_mini_batch):
            payment[batch_k] = {}
            for player_k in range(n_player):
                payment[batch_k][player_k] = m.addVar(
                    vtype=grb.GRB.CONTINUOUS, lb=0, ub=b[batch_k][player_k],
                    name='payment_%s_%s' % (batch_k, player_k))
        m.update()
        # pA >= beta
        for batch_k in range(n_mini_batch):
            for coalition_k, coalition_v in enumerate(beta[batch_k]):
                # Consider only coalitions with >0 blocking value
                if coalition_v <= 0:
                    continue
                # TODO, Paul: Can add another check whether coalition is already existing and only pick highest value for it
                sum_payments = 0
                for payment_k in range(len(payment[batch_k])):
                    sum_payments += payment[batch_k][payment_k] * A[batch_k, coalition_k, payment_k]
                m.addConstr(sum_payments >= coalition_v,
                            name='1_outpay_coalition_%s_%s' % (batch_k, coalition_k))
        return m

    def _add_objective_min_payments_and_solve(self, model, n_mini_batch, n_player, print_output=False):
        # min p1
        mu = 0
        mu_batch = {}
        for batch_k in range(n_mini_batch):
            mu_batch[batch_k] = 0
            for player_k in range(n_player):
                mu_batch[batch_k] += model.getVarByName("payment_%s_%s" % (batch_k, player_k))
            mu += mu_batch[batch_k]

        model.setObjective(mu, sense=grb.GRB.MINIMIZE)
        model.update()

        # Solve LP
        if print_output:
            model.write(os.path.join(os.path.expanduser('~'), 'bnelearn', 'GurobiSol', '%s_LP.lp' % self.rule))
        model.optimize()

        if print_output:
            try:
                model.write(os.path.join(os.path.expanduser('~'), 'bnelearn', 'GurobiSol', '%s_LP.sol' % self.rule))
            except:
                model.computeIIS()
                model.write(os.path.join(os.path.expanduser('~'), 'bnelearn', 'GurobiSol', '%s_LP.ilp' % self.rule))
                raise

        mu_out = {}
        for batch_k in range(n_mini_batch):
            mu_out[batch_k] = 0
            for player_k in range(n_player):
                mu_out[batch_k] += model.getVarByName("payment_%s_%s" % (batch_k, player_k)).X

        return model, mu_out

    def _add_objective_min_vcg_distance_and_solve(
            self, model, payments_vcg, n_mini_batch, n_player, print_output=False):
        loss = 0
        loss_batch = 0
        for batch_k in range(n_mini_batch):
            loss_batch = 0
            for player_k in range(n_player):
                loss_batch += (
                    model.getVarByName("payment_%s_%s" % (batch_k, player_k)) - payments_vcg[batch_k][player_k]
                ) * (model.getVarByName("payment_%s_%s" % (batch_k, player_k)) - payments_vcg[batch_k][player_k])
            loss += loss_batch

        model.setObjective(loss, sense=grb.GRB.MINIMIZE)
        model.update()

        # Solve QP
        if print_output:
            model.write(os.path.join(os.path.expanduser('~'), 'bnelearn', 'GurobiSol', '%s_QP.lp' % self.rule))
        model.optimize()
        if print_output:
            try:
                model.write(os.path.join(os.path.expanduser('~'), 'bnelearn', 'GurobiSol', '%s_QP.sol' % self.rule))
            except:
                model.computeIIS()
                model.write(os.path.join(os.path.expanduser('~'), 'bnelearn', 'GurobiSol', '%s_QP.ilp' % self.rule))
                raise

        payment_out = torch.zeros((n_mini_batch, n_player), device=payments_vcg.device)
        for batch_k in range(n_mini_batch):
            for player_k in range(n_player):
                payment_out[batch_k][player_k] = model.getVarByName("payment_%s_%s" % (batch_k, player_k)).X
        return payment_out

    @staticmethod
    def _add_constraint_min_payments(model, mu, n_mini_batch, n_player):
        # adjusted to LEQ with 1e-5 instead of equals (according to Bosshard code)
        # p1 = mu
        for batch_k in range(n_mini_batch):
            sum_payments = 0
            for player_k in range(n_player):
                sum_payments += model.getVarByName("payment_%s_%s" % (batch_k, player_k))
            model.addConstr(sum_payments <= mu[batch_k] + 1e-5,
                            name='2_min_payment_%s' % batch_k)

        model.update()
        return model

    def _run_batch_nearest_vcg_core_qpth_mpc(self, A, beta, payments_vcg, b, solver, min_core_payments=True):
        # Initialize the model.
        model = _OptNet_for_LLLLGG(self.device, A, beta, b, payments_vcg)
        if min_core_payments:
            model._add_objective_min_payments()
            mu = model(solver)
            model._add_objective_min_vcg_distance(mu)
        else:
            model._add_objective_min_vcg_distance()
        return model(solver)

    def _run_batch_nearest_vcg_core_cvxpy(self, A, beta, payments_vcg, b, min_core_payments=True):
        import cvxpy as cp
        from cvxpylayers.torch import CvxpyLayer
        n_batch, n_coalitions, n_player = A.shape

        mu_p = cp.Parameter(1)
        payment_vcg_p = cp.Parameter(n_player)
        payments_p = cp.Variable(n_player)
        G_p = cp.Parameter((n_player + n_coalitions, n_player))
        h_p = cp.Parameter(n_player + n_coalitions)
        G = torch.cat((-A, torch.eye(n_player, device=self.device).repeat(n_batch, 1, 1)), 1)
        h = torch.cat((-beta, b), 1)

        # Solve LP
        cons = [G_p @ payments_p <= h_p + 1e-7 * torch.ones(n_player + n_coalitions),
                payments_p @ -torch.eye(n_player) <= torch.zeros(n_player)]
        if min_core_payments:
            obj = cp.Minimize(cp.sum(payments_p))
            prob = cp.Problem(obj, cons)
            layer = CvxpyLayer(prob, parameters=[G_p, h_p], variables=[payments_p])
            mu, = layer(G, h)

        # Solve QP
        obj = cp.Minimize(cp.sum_squares(payments_p - payment_vcg_p))
        if min_core_payments:
            cons += [payments_p @ torch.ones(n_player).reshape(6, 1) <= mu_p + 1e-5 * torch.ones(1)]
        prob = cp.Problem(obj, cons)
        x = payments_vcg.clone().detach()

        if min_core_payments:
            layer = CvxpyLayer(prob, parameters=[payment_vcg_p, G_p, h_p, mu_p], variables=[payments_p])
            y, = layer(x, G, h, mu.sum(1).reshape(n_batch, 1))
        else:
            layer = CvxpyLayer(prob, parameters=[payment_vcg_p, G_p, h_p], variables=[payments_p])
            y, = layer(x, G, h)

        return y

    def run(self, bids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Performs a specific LLLLGG auction as in Seuken Paper (Bosshard et al. (2019))

        Args:
            bids: torch.Tensor
                of bids with dimensions (batch_size, n_players, 2) [0,Inf]
            bundles: torch.Tensor
                of bundles with dimensions (batch_size, 2, n_items), {0,1}

        Returns:
            allocation: torch.Tensor, dim (batch_size, n_bidders, 2)
            payments: torch.Tensor, dim (batch_size, n_bidders)
        """

        allocation, welfare = self._solve_allocation_problem(bids)
        if self.rule == 'vcg':
            payments = self._calculate_payments_vcg(bids, allocation, welfare)
        elif self.rule == 'first_price':
            payments = self._calculate_payments_first_price(bids, allocation)
        elif self.rule == 'nearest_vcg':
            payments = self._calculate_payments_nearest_vcg_core(bids, allocation, welfare)
        else:
            raise ValueError('Invalid Pricing rule!')
        # transform allocation
        allocation = allocation.view(bids.shape)

        return allocation.to(self.device), payments.to(self.device)


class CombinatorialAuction(Mechanism):
    """A combinatorial auction, implemented via (possibly parallel) calls to the gurobi solver.

       Args:
        rule: pricing rule

    """

    def __init__(self, rule='first_price', cuda: bool = True, bundles=None, parallel: int = 1):
        super().__init__(cuda)

        if rule not in ['vcg']:
            raise NotImplementedError(':(')

        # 'nearest_zero' and 'proxy' are aliases
        if rule == 'proxy':
            rule = 'nearest_zero'

        self.rule = rule
        self.parallel = parallel

        self.bundles = bundles
        self.n_items = len(self.bundles[0])

    def __mute(self):
        """suppresses stdout output from workers (avoid gurobi startup licence message clutter)"""
        sys.stdout = open(os.devnull, 'w')

    def run(self, bids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Performs a general Combinatorial auction

        Args:
            bids: torch.Tensor
                of bids with dimensions (batch_size, n_players, 2) [0,Inf]
            bundles: torch.Tensor
                of bundles with dimensions (batch_size, 2, n_items), {0,1}

        Returns:
            allocation: torch.Tensor, dim (batch_size, n_bidders, 2)
            payments: torch.Tensor, dim (batch_size, n_bidders)
        """

        # detect appropriate pool size
        pool_size = min(self.parallel, len(bids))

        # parallel version
        if pool_size > 1:

            iterator = bids.detach().cpu().split(1)
            n_chunks = len(iterator)

            torch.multiprocessing.set_sharing_strategy('file_system')
            with torch.multiprocessing.Pool(pool_size, initializer=self.__mute) as p:
                # as we handled chunks ourselves, each element of our list should be an individual chunk,
                # so the pool.map will get argument chunksize=1
                # The following code is wrapped to produce progess bar, without it simplifies to:
                # result = p.map(self.closure, split_tensor, chunksize=1)
                result = list(tqdm(
                    p.imap(self._run_single_batch, iterator, chunksize=1),
                    total=n_chunks, unit='chunks',
                    desc='Solving mechanism for batch_size {} with {} processes, chunk size of {}'.format(
                        n_chunks, pool_size, 1)
                ))
            allocation, payment = [torch.cat(x) for x in zip(*result)]

        else:
            iterator = bids.split(1)
            allocation, payment = [torch.cat(x) for x in zip(*map(self._run_single_batch, iterator))]

        return allocation.to(self.device), payment.to(self.device)

    def _run_single_batch(self, bids):
        """Runs the auction for a single batch of bids.

        Currently only supports bid languages where all players bid on the same number of bundles.

        Args:
            bids: torch.Tensor (1 x n_player x n_bundles)

        Returns:
            winners: torch.Tensor (1 x n_player x n_bundles), payments: torch.Tensor (1 x n_player)
        """

        # # for now we're assuming all bidders submit same number of bundle bids
        _, n_players, n_bundles = bids.shape  # this will not work if different n_bundles per player
        bids = bids.squeeze(0)
        model_all, assign_i_s = self._build_allocation_problem(bids)
        self._solve_allocation_problem(model_all)

        allocation = torch.tensor(model_all.getAttr('x', assign_i_s).values(),
                                  device=bids.device).view(n_players, n_bundles)
        if self.rule == 'vcg':
            payments = self._calculate_payments_vcg(bids, model_all, allocation, assign_i_s)
        else:
            raise ValueError('Invalid Pricing rule!')

        return allocation.unsqueeze(0), payments.unsqueeze(0)

    def _calculate_payments_vcg(self, bids, model_all, allocation, assign_i_s):
        """
        Caculating vcg payments
        """
        n_players, n_bundles = bids.shape
        utilities = (allocation * bids).sum(dim=1)  # shape: n_players
        # get relevant state of full model
        n_global_constr = len(model_all.getConstrs())
        global_objective = model_all.ObjVal  # pylint: disable = no-member
        # Solve allocation problem without each player to get vcg prices
        delta_tensor = torch.zeros(n_players, device=bids.device)
        for bidder in range(n_players):
            # additional constraints: no assignment to bidder i
            model_all.addConstrs((assign_i_s[(bidder, bundle)] <= 0 for bundle in range(n_bundles)))
            model_all.update()

            self._solve_allocation_problem(model_all)
            delta_tensor[bidder] = global_objective - model_all.ObjVal  # pylint: disable = no-member

            # get rid of additional constraints added above
            model_all.remove(model_all.getConstrs()[n_global_constr:])

        return utilities - delta_tensor

    def _solve_allocation_problem(self, model):
        """
        solving handed model

        Args: a gurobi model object

        Returns: nothing, model object is updated in place
        """
        model.setParam('OutputFlag', 0)
        # if print_gurobi_model:
        #     model.write(os.path.join(util.PATH, 'GurobiSol', '%s.lp' % model_name))
        model.optimize()

        # if print_gurobi_model:
        #     try:
        #         model.write(os.path.join(util.PATH, 'GurobiSol', '%s.sol' % model_name))
        #     except:
        #         model.computeIIS()
        #         model.write(util.PATH+'\\GurobiSol\\%s.ilp' % model_name)
        #         raise

    def _build_allocation_problem(self, bids):
        """
        Parameters
        ----------
        bids: torch.Tensor [n_bidders, n_bundles], valuation for bundles


        Returns
        ----------
        model: full gurobi model
        assign_i_s: dict of gurobi vars with keys (bidder, bundle), 1 if bundle is assigned to bidder, else 0
                    value of the gurobi var can be accessed with assign_i_s[key].X
        """
        # In the standard case every bidder has to bid on every bundle.
        n_players, n_bundles = bids.shape
        assert n_bundles == len(self.bundles), "Bidder 0 doesn't bid on all bundles"

        m = grb.Model()
        m.setParam('OutputFlag', 0)
        # m.params.timelimit = 600

        # assign vars
        assign_i_s = {}  # [None] * n_players
        for bidder in range(n_players):
            # number of bundles might be specific to the bidder
            for bundle in range(n_bundles):
                assign_i_s[(bidder, bundle)] = m.addVar(vtype=grb.GRB.BINARY,
                                                        name='assign_%s_%s' % (bidder, bundle))
        m.update()

        # Bidder can at most win one bundle
        # NOTE: this block can be sped up by ~20% (94.4 +- 4.9 microseconds vs 76.3+2.3 microseconds in timeit)
        # by replacing the inner for loop with
        # sum_winning_bundles = sum(list(assign_i_s.values())[bidder*N_BUNDLES:(bidder+1)*N_BUNDLES])
        # but that's probably not worth the loss in readability
        for bidder in range(n_players):
            sum_winning_bundles = grb.LinExpr()
            for bundle in range(n_bundles):
                sum_winning_bundles += assign_i_s[(bidder, bundle)]
            m.addConstr(sum_winning_bundles <= 1, name='1_max_bundle_bidder_%s' % bidder)

        # every item can be allocated at most once
        for item in range(self.n_items):
            sum_item = 0
            for (k1, k2) in assign_i_s:  # the keys are tuples thus pylint: disable=dict-iter-missing-items
                sum_item += assign_i_s[(k1, k2)] * self.bundles[k2][item]
            m.addConstr(sum_item <= 1, name='2_max_ass_item_%s' % item)

        objective = sum([var * coeff for var, coeff in zip(assign_i_s.values(), bids.flatten().tolist())])

        m.setObjective(objective, sense=grb.GRB.MAXIMIZE)
        m.update()

        return m, assign_i_s


class MultiBattleAllPayAuction(Mechanism):

    def __init__(self, cuda: bool = True):
        super().__init__(cuda)

    def run(self, bids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        
        Runs a (batch of) the standard version of the all pay auction.

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
        payments = bids.sum(dim=item_dim) # pay as bid
        allocations = torch.zeros(batch_size, n_players, n_items, device=self.device)

        bids_transposed = bids.transpose(item_dim, player_dim)
        _, winning_bidders = bids_transposed.max(dim = item_dim, keepdim=True)
        winning_bidders.transpose_(player_dim, item_dim)

        allocations.scatter_(player_dim, winning_bidders, 1)
        # Don't allocate items that have a winnign bid of zero.
        payments_per_item = payments.reshape((payments.shape[0], payments.shape[1], 1))
        allocations.masked_fill_(mask=payments_per_item == 0, value=0)

        return allocations, payments