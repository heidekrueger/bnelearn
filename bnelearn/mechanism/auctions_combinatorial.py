import os
import sys

from typing import Tuple
import warnings

import gurobipy as grb

# pylint: disable=E1102
import torch

from tqdm import tqdm

# For qpth #pylint:disable=ungrouped-imports
import torch.nn as nn
from qpth.qp import QPFunction

from .mechanism import Mechanism
from bnelearn.util import mpc
# from bnelearn.util import qpth_class
from time import perf_counter as timer
class _OptNet_for_LLLLGG(nn.Module):
    def __init__(self, device, A, beta, b, payment_vcg, precision=torch.double):
        """
        Build basic model
        s.t.
        pA >= beta
        p <= b
        ->
        G =     (-A   )
            diag(1,...,1)
        h = (-beta)
            (b      )

        See LLLLGGAuction._calculate_payments_nearest_vcg_core for details on variables.
        """
        self.n_batch, self.n_coalitions, self.n_player = A.shape  # pylint:disable=unused-variable

        super().__init__()
        self.device = device
        self.precision = precision
        self.payment_vcg = payment_vcg

        A = torch.as_tensor(A, dtype=precision, device=self.device)
        b = torch.as_tensor(b, dtype=precision, device=self.device)
        beta = torch.as_tensor(beta, dtype=precision, device=self.device)

        self.G = torch.cat(
            (
                -A,
                torch.eye(self.n_player, dtype=precision, device=self.device).repeat(self.n_batch, 1, 1),
                -torch.eye(self.n_player, dtype=precision, device=self.device).repeat(self.n_batch, 1, 1)
            ), 1)
        self.h= torch.cat(
            (
                -beta,
                b,
                torch.zeros([self.n_batch, self.n_player], dtype=precision, device=self.device)
            ), 1)
        
        self.e = torch.zeros(0, dtype=precision, device=self.device, requires_grad=True)
        self.mu = torch.zeros(0, dtype=precision, device=self.device, requires_grad=True)

        #for mpc
        self.e_=None
        self.mu_=None

        # will be set by methods
        self.Q = None
        self.q = None

    def _add_objective_min_payments(self):
        """
        Add objective to minimize total payments and solve LP:
        min p x 1

        Q = (0,...,0)
        q = (1,...,1)
        """
        self.Q = torch.diag(torch.tensor([1e-5, ] * self.n_player, dtype=self.precision, device=self.device)).repeat(self.n_batch,1,1)
        self.q = torch.ones([self.n_batch, self.n_player], dtype=self.precision, device=self.device)

    def _add_objective_min_vcg_distance(self, min_payments=None):
        """
        Add objective to minimize euclidean vcg distance QP:
        min (p-p_0)(p-p_0)

        Q = diag(2,...,2)
        q = -2p_0
        """
        if min_payments is not None:
            self.e = torch.ones([self.n_batch, 1, self.n_player], dtype=self.precision, device=self.device)
            self.mu = min_payments.sum(1).reshape(self.n_batch, 1)
            # #for mpc
            self.e_= torch.ones([self.n_batch, 1, self.n_player], dtype=self.precision, device=self.device)
            self.mu_ = min_payments.sum(1).reshape(self.n_batch, 1)#

        payment_vcg = torch.as_tensor(self.payment_vcg, dtype=self.precision, device=self.device)
        self.Q = torch.diag(torch.tensor([2, ] * self.n_player, dtype=self.precision, device=self.device)).repeat(self.n_batch,1,1)
        self.q = -2 * payment_vcg
    def forward(self, input=None):
        """input is not used, as problem is fully specified"""
        mpc_solver=mpc.mpc_class(max_iter=20)
        # detach all variables to set requires_grad=False
        self.Q_no_grad=self.Q.detach()
        self.q_no_grad=self.q.detach()
        self.G_no_grad=self.G.detach()
        self.h_no_grad=self.h.detach()
        if self.e_!=None:
            #append to e,mu to inequality constraints
            # self.G_no_grad=torch.cat((self.G_no_grad,self.e_),1).detach()
            # self.h_no_grad=torch.cat((self.h,self.mu_),1).detach()
            self.e_no_grad=self.e_.detach()
            self.mu_no_grad=self.mu_.detach()
        else:
            self.e_no_grad=None
            self.mu_no_grad=None
        # start=time.time()
        x_mpc,opt_mpc=mpc_solver.solve(self.Q_no_grad, self.q_no_grad, self.G_no_grad,
                                         self.h_no_grad, self.e_no_grad, self.mu_no_grad,
                                         print_warning=False)#,check_Q_psd=False)
        # print(time.time()-start)    
        # print(x_mpc.device, opt_mpc.device)
        # Q_LU, S_LU, R = qpth_class.pre_factor_kkt(self.Q, self.G, self.e)
        # x_qp,s,z,y=qpth_class.forward(self.Q, self.q, self.G, self.h, self.e, self.mu, 
        #                 Q_LU, S_LU, R, eps=1e-12, verbose=0, notImprovedLim=3, maxIter=25)
        
        # x_qp=QPFunction(
        #     verbose=-1, eps=1e-19, maxIter=100, notImprovedLim=10, check_Q_spd=False
        # )(self.Q, self.q, self.G, self.h, self.e, self.mu)
        # print("____________qpth results________________")
        # print(x_qp)
        # return QPFunction(
        #     verbose=-1, eps=1e-19, maxIter=100, notImprovedLim=10, check_Q_spd=False
        # )(self.Q, self.q, self.G, self.h, self.e, self.mu)
        # print solution returned by mpc
        # print("____________mpc results________________")
        # print(x_mpc.view(self.n_batch,self.n_player))
        # if torch.isnan(x_mpc).any():
        #     print(x_mpc)
        return x_mpc

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

        assert n_players == 3, "invalid n_players in LLG setting"
        assert n_items == 1, "invalid bid_dimensionality in LLG setting"  # dummy item is desired bundle for each player

        # move bids to gpu/cpu if necessary, get rid of unused item_dim
        bids = bids.squeeze(item_dim).to(self.device)  # batch_size x n_players
        # individual bids as batch_size x 1 tensors:
        b1, b2, bg = bids.split(1, dim=1)

        # allocate return variables
        payments = torch.zeros(batch_size, n_players, device=self.device)
        allocations = torch.zeros(batch_size, n_players, n_items, device=self.device)

        # 1. Determine efficient allocation
        locals_win = (b1 + b2 > bg).float()  # batch_size x 1
        allocations = locals_win * torch.tensor([[1., 1., 0.]], device=self.device) + \
                      (1 - locals_win) * torch.tensor([[0., 0., 1.]], device=self.device)  # batch x players

        if self.rule == 'first_price':
            payments = allocations * bids  # batch x players
        else:  # calculate local and global winner prices separately
            payments = torch.zeros(batch_size, n_players, device=self.device)
            global_winner_prices = b1 + b2  # batch_size x 1
            payments[:, [2]] = (1 - locals_win) * global_winner_prices

            local_winner_prices = torch.zeros(batch_size, 2, device=self.device)

            if self.rule in ['vcg', 'nearest_vcg']:
                # vcg prices are needed for vcg, nearest_vcg
                local_vcg_prices = (bg - bids[:, [1, 0]]).relu()

                if self.rule == 'vcg':
                    local_winner_prices = local_vcg_prices
                else:  # nearest_vcg
                    delta = 0.5 * (bg - local_vcg_prices[:, [0]] - local_vcg_prices[:, [1]])  # batch_size x 1
                    local_winner_prices = local_vcg_prices + delta  # batch_size x 2
            elif self.rule in ['proxy', 'nearest_zero']:
                # three cases when local bidders win:
                #  1. "both_strong": each local > half of global --> both play same
                #  2. / 3. one player 'weak': weak local player pays her bid, other pays enough to match global
                both_strong = ((bg <= 2 * b1) & (bg <= 2 * b2)).float()  # batch_size x 1
                first_weak = (2 * b1 < bg).float()
                # (second_weak implied otherwise)
                local_prices_case_both_strong = 0.5 * torch.cat(2 * [bg], dim=player_dim)
                local_prices_case_first_weak = torch.cat([b1, bg - b1], dim=player_dim)
                local_prices_case_second_weak = torch.cat([bg - b2, b2], dim=player_dim)

                local_winner_prices = both_strong * local_prices_case_both_strong + \
                                      first_weak * local_prices_case_first_weak + \
                                      (1 - both_strong - first_weak) * local_prices_case_second_weak
            elif self.rule == 'nearest_bid':
                case_yes = (bg < b1 - b2).float()  # batch_size x 1

                local_prices_case_yes = torch.cat([bg, torch.zeros_like(bg)], dim=player_dim)

                delta = 0.5 * (b1 + b2 - bg)
                local_prices_case_no = bids[:, [0, 1]] - delta

                local_winner_prices = case_yes * local_prices_case_yes + (1 - case_yes) * local_prices_case_no

            else:
                raise ValueError("invalid bid rule")

            payments[:, [0, 1]] = locals_win * local_winner_prices

        return (allocations.unsqueeze(-1), payments)  # payments: batches x players, allocation: batch x players x items


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
        from bnelearn.util import large_lists_LLLLGG  # pylint:disable=import-outside-toplevel
        super().__init__(cuda)

        if rule not in ['nearest_vcg', 'vcg', 'first_price']:
            raise NotImplementedError(':(')

        if rule == 'nearest_vcg':
            if core_solver not in ['gurobi', 'cvxpy', 'qpth']:
                raise NotImplementedError(':/')
        # 'nearest_zero' and 'proxy' are aliases
        if rule == 'proxy':
            rule = 'nearest_zero'
        self.rule = rule
        self.n_items = 8
        self.n_bidders = 6
        self.n_bundles = 2
        self.core_solver = core_solver
        self.parallel = parallel

        # solver might require 'cpu' even when `self.device=='cuda'`, we thus work with a copy
        _device = self.device
        if (parallel > 1 and core_solver == 'gurobi'):
            _device = 'cpu'
        self.solutions_sparse = torch.tensor(large_lists_LLLLGG.solutions_sparse, device=_device)

        self.solutions_non_sparse = torch.tensor(large_lists_LLLLGG.solutions_non_sparse,
                                                 dtype=torch.float, device=_device)

        self.subsolutions = torch.tensor(large_lists_LLLLGG.subsolutions, device=_device)

        self.player_bundles = torch.tensor([
            # Bundles
            # B1,B2,B3,B4, B5,B6,B7,B8, B9,B10,B11,B12
            [0, 1],
            [2, 3],
            [4, 5],
            [6, 7],
            [8, 9],
            [10, 11]
        ], dtype=torch.long, device=_device)

    def __mute(self):
        """suppresses stdout output from workers (avoid gurobi startup licence message clutter)"""
        sys.stdout = open(os.devnull, 'w')

    def _solve_allocation_problem(self, bids: torch.Tensor, dont_allocate_to_zero_bid=True):
        """
        Computes allocation and welfare

        Parameters
        ----------
        bids: torch.Tensor
            of bids with dimensions (batch_size, n_players=6, n_bids=2), values = [0,Inf]
        solutions: torch.Tensor
            of possible allocations.

        Returns
        -------
        allocation: torch.Tensor(batch_size, b_bundles = 18), values = {0,1}
        welfare: torch.Tensor(batch_size), values = [0, Inf]
        """
        solutions = self.solutions_non_sparse.to(self.device)

        n_batch, n_players, n_bundles = bids.shape
        bids_flat = bids.view(n_batch, n_players * n_bundles)
        solutions_welfare = torch.mm(bids_flat, torch.transpose(solutions, 0, 1))
        welfare, solution = torch.max(solutions_welfare, dim=1)  # maximizes over all possible allocations
        winning_bundles = solutions.index_select(0, solution)
        if dont_allocate_to_zero_bid:
            winning_bundles = winning_bundles * (bids_flat > 0)

        return winning_bundles, welfare

    def _calculate_payments_first_price(self, bids: torch.Tensor, allocation: torch.Tensor):
        """
        Computes first prices

        Parameters
        ----------
        bids: torch.Tensor
            of bids with dimensions (batch_size, n_players=6, n_bids=2), values = [0,Inf]
        allocation: torch.Tensor(batch_size, b_bundles = 18), values = {0,1}

        Returns
        -------
        payments: torch.Tensor(batch_size, n_bidders), values = [0, Inf]
        """
        n_batch, n_players, n_bundles = bids.shape
        return (allocation.view(n_batch, n_players, n_bundles) * bids).sum(dim=2)

    def _calculate_payments_vcg(self, bids: torch.Tensor, allocation: torch.Tensor, welfare: torch.Tensor):
        """
        Computes VCG prices

        Parameters
        ----------
        bids: torch.Tensor
            of bids with dimensions (batch_size, n_players=6, n_bids=2), values = [0,Inf]
        allocation: torch.Tensor(batch_size, b_bundles = 18), values = {0,1}
        welfare: torch.Tensor(batch_size), values = [0, Inf]

        Returns
        -------
        payments: torch.Tensor(batch_size, n_bidders), values = [0, Inf]
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
        '''
        Nearest VCG core payments by Day and Crampton (2012) [link to paper]
        Instead of computing all possible coalitions, or the most blocking respectively,
            we iterate through all possible subsolutions, containing all possible coalitions.
        We minimize the prices and solve the LP:
        mu = min p1
        pA >= beta
        p <= b

        and after, we minimize the deviation from VCG and solve the QP:
        min (p-p_0)(p-p_0)
        s.t.
        pA >= beta
        p <= b
        p1 == mu
        ------
        p_0: Parameter - VCG payments
        p: Variable - Core Payments
        ---
        beta = Parameter - coalitions willingness to pay
        beta = welfare(coalition) - sum_(j \\in coalition){b_j(S_j)} \\forall coalitions in subsolutions
        with b_j(S_j) being the bid of the actual allocation (their willingness to pay for what they already get.)
        ---
        A = Parameter - winning and not in coalition (1, else 0)
        b = Parameter - bid of winning bidders (0 if non winning)
        '''
        subsolutions = self.subsolutions.to(self.device)

        n_batch, n_player, n_bundle = bids.shape
        # Generate dense tensor of subsolutions
        subsolutions_dense = torch.sparse.FloatTensor(
            subsolutions.t(),
            torch.ones(len(subsolutions), device=self.device),
            torch.Size([subsolutions[-1][0] + 1, n_player * n_bundle], device=self.device)
        ).to_dense()
        # Compute beta
        coalition_willing_to_pay = torch.mm(bids.view(n_batch, n_player * n_bundle), subsolutions_dense.t())

        # For b_j(S_j) we need to consider the actual winning bid of j.
        # Therefore, we adjust the coalition and set 1 for each bundle of j
        winning_and_in_coalition = torch.einsum(
            'ij,kjl->kijl',
            subsolutions_dense.view(66, n_player, n_bundle).sum(dim=2),
            allocation.view(n_batch, n_player, n_bundle)).view(n_batch, 66, n_player * n_bundle)

        coalition_already_getting = torch.bmm(
            bids.view(n_batch, 1, n_player * n_bundle),
            winning_and_in_coalition.permute(0, 2, 1)).reshape(n_batch, 66)

        beta = coalition_willing_to_pay - coalition_already_getting
        # Fixing numerical imprecision (as occured before!)
        beta[beta < 1e-6] = 0

        assert beta.shape == (n_batch, 66), "beta has the wrong shape"

        A = allocation.view(n_batch, 1, n_player * n_bundle) - winning_and_in_coalition
        A = A.view(n_batch, 66, n_player, n_bundle).sum(dim=3)

        # Computing b
        b = torch.sum(allocation.view(n_batch, n_player, n_bundle) * bids.view(n_batch, n_player, n_bundle), dim=2)
        payments_vcg = self._calculate_payments_vcg(bids, allocation, welfare)

        A, beta = self._reduce_nearest_vcg(A, beta)
        # Choose core solver
        if self.core_solver == 'gurobi':
            payment = self._run_batch_nearest_vcg_core_gurobi(A, beta, payments_vcg, b)
        elif self.core_solver == 'cvxpy':
            payment = self._run_batch_nearest_vcg_core_cvxpy(A, beta, payments_vcg, b)
        elif self.core_solver == 'qpth':
            payment = self._run_batch_nearest_vcg_core_qpth(A, beta, payments_vcg, b)
        else:
            raise NotImplementedError(":/")
        return payment

    def _reduce_nearest_vcg(self, A, beta):
        start_time = timer()
        # TODO, Paul: speed up?
        n_batch, n_coalition, n_player = A.shape

        # remove coalitions that pay no extra (<=0)
        # min_true = min((beta <= 0).sum(1))
        # remove = torch.topk(beta.to(torch.float32), min_true, dim = 1, sorted=False, largest=False).indices #, largest=False
        # keep = torch.ones((n_batch, n_coalition), device = self.device, dtype=bool).scatter_(1,remove,False)
        # keep2 = torch.stack([keep]*n_player,2)
        # beta = beta.masked_select(keep).view(n_batch, n_coalition-min_true) #.shape
        # A = A.masked_select(keep2).view(n_batch, n_coalition-min_true, n_player) #.shape
        # print("Reduced by %s constraints" %min_true)
        # n_batch, n_coalition, n_player = A.shape

        ### For each coalition, consider only the highest bid
        # Get identical coalitions s.t. dimension are kept over all batches
        A_unique, A_unique_idx = A.unique(sorted=False,dim=1,return_inverse=True)
        
        ## Phase 1: For each coalition duplicate, find the max bid
        # Sort coalition bids decreasing per batch
        beta_sort, beta_sort_idx = beta.squeeze().sort(descending=True)
        # Sort the A unique indexing (matching unique to original) decreasing by beta -> coalitions with highest bid up
        A_unique_idx_sorted_by_beta = A_unique_idx[beta_sort_idx]
        # Now sort it increasing by its index -> starting with coalition 0, 1, 2,... always the one with highest bid on top
        A_unique_idx_sorted_complete, A_unique_idx_sorted_complete_idx = A_unique_idx_sorted_by_beta.sort(dim = 1, descending=False)

        ## Phase 2: Keep only the coalition duplicate with max bid
        # Create tensor to select only the first of a group
        tmp_select_first = torch.zeros((n_batch,n_coalition), dtype=int, device=self.device)
        tmp_select_first[:,0] = -1 
        tmp_select_first[:,1:] = A_unique_idx_sorted_complete[:,0:(n_coalition-1)]
        tmp_select_first = torch.tensor(A_unique_idx_sorted_complete - tmp_select_first, dtype=torch.bool, device=self.device)

        ## Phase 3: Compute back the beta indices matching to A_uniq with max bid
        # Select only the first unique coalition (indices!)
        A_unique_idx_sorted_complete_idx_Only_first = torch.masked_select(A_unique_idx_sorted_complete_idx, 
                                                                          tmp_select_first).view(n_batch,max(tmp_select_first.sum(1)))
        # Only keep the first indices of beta
        beta_sort_idx_Only_first = torch.gather(beta_sort_idx,1,A_unique_idx_sorted_complete_idx_Only_first)

        # Combine the betas to keep with the "unique" A set
        beta_only_first = torch.gather(beta,1,beta_sort_idx_Only_first)
        print("Reduced by {} constraints, in {:0.2f}. seconds".format((A.shape[1]-A_unique.shape[1]), (timer() - start_time)))

        return A_unique, beta_only_first

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
            m, mu = self._add_objective_min_payments_and_solve(m, n_mini_batch, n_player)
            # add minimal payments constraint to minimizing vcg distance problem
            m = self._add_constraint_min_payments(m, mu, n_mini_batch, n_player)

        # setup and solve minimizing vcg distance
        m.setParam('FeasibilityTol', 1e-9)
        m.setParam('MIPGap', 1e-9)
        payments = self._add_objective_min_vcg_distance_and_solve(m, payments_vcg, n_mini_batch, n_player)

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
                                      model.getVarByName("payment_%s_%s" % (batch_k, player_k)) - payments_vcg[batch_k][
                                  player_k]
                              ) * (model.getVarByName("payment_%s_%s" % (batch_k, player_k)) - payments_vcg[batch_k][
                    player_k])
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

    # Adjusted to LEQ with 1e-5 instead of equals (according to Bosshard code)
    # TODO: method could be a function (no self use)
    def _add_constraint_min_payments(self, model, mu, n_mini_batch, n_player):
        # p1 = mu
        for batch_k in range(n_mini_batch):
            sum_payments = 0
            for player_k in range(n_player):
                sum_payments += model.getVarByName("payment_%s_%s" % (batch_k, player_k))
            model.addConstr(sum_payments <= mu[batch_k] + 1e-5,
                            name='2_min_payment_%s' % batch_k)

        model.update()
        return model

    def _run_batch_nearest_vcg_core_qpth(self, A, beta, payments_vcg, b, min_core_payments=True):
        # Initialize the model.
        warnings.warn('Experimental! Do not use qpth at this state, since it is very imprecise for large batches.')
        model = _OptNet_for_LLLLGG(self.device, A, beta, b, payments_vcg)
        if min_core_payments:
            model._add_objective_min_payments()
            mu = model()
            model._add_objective_min_vcg_distance(mu)
        else:
            model._add_objective_min_vcg_distance()

        return model()

    def _run_batch_nearest_vcg_core_cvxpy(self, A, beta, payments_vcg, b, min_core_payments=True):
        import cvxpy as cp
        from cvxpylayers.torch import CvxpyLayer
        n_batch, n_coalitions, n_player = A.shape

        mu_p = cp.Parameter(1)
        payment_vcg_p = cp.Parameter(n_player)
        payments_p = cp.Variable(n_player)
        G_p = cp.Parameter((72, n_player))
        h_p = cp.Parameter(72)
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

        Parameters
        ----------
        bids: torch.Tensor
            of bids with dimensions (batch_size, n_players, 2) [0,Inf]
        bundles: torch.Tensor
            of bundles with dimensions (batch_size, 2, n_items), {0,1}

        Returns
        -------
        allocation: torch.Tensor(batch_size, n_bidders, 2)
        payments: torch.Tensor(batch_size, n_bidders)
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

        Parameters
        ----------
        bids: torch.Tensor
            of bids with dimensions (batch_size, n_players, 2) [0,Inf]
        bundles: torch.Tensor
            of bundles with dimensions (batch_size, 2, n_items), {0,1}

        Returns
        -------
        allocation: torch.Tensor(batch_size, n_bidders, 2)
        payments: torch.Tensor(batch_size, n_bidders)
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
