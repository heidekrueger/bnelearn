import pytest
import torch
from bnelearn.mechanism import LLLLGGAuction

"""Testing for identical results from solvers for the LLLLGG combinatorial auction implementations."""
torch.manual_seed(1)
torch.cuda.manual_seed(1)
bids_1 = torch.rand([2**5,6,2], dtype = torch.float)
# TODO: qpth becomes too inaccurate when batch > 2**5. Can we solve this?
# Console: pytest bnelearn/tests/test_llllgg_core_solver_comp.py -s
# (parallel, payment rule, bids, device)
ids, testdata = zip(*[
    ['nearest vcg - single - cpu', (1,'nearest_vcg', bids_1, 'cpu')],
    ['nearest vcg - multi - cpu', (8,'nearest_vcg', bids_1, 'cpu')],
    ['nearest vcg - single - gpu', (1,'nearest_vcg', bids_1, 'cuda')],
    ['nearest vcg - multi - gpu', (8,'nearest_vcg', bids_1, 'cuda')],
])

def run_LLLLGG_test(parallel, rule, bids, device):
    """Run comparison test for different solvers."""
    cuda = device == 'cuda' and torch.cuda.is_available()

    if device == 'cuda' and not cuda:
        pytest.skip("This test needs CUDA, but it's not available.")

    game_gurobi = LLLLGGAuction(rule=rule, cuda=cuda, core_solver='gurobi', parallel=parallel)
    game_cvxpy = LLLLGGAuction(rule=rule,cuda=cuda, core_solver='cvxpy')
    game_qpth = LLLLGGAuction(rule=rule,cuda=cuda, core_solver='qpth')

    allocation_gurobi, payments_gurobi = game_gurobi.run(bids.to(device))
    allocation_cvxpy, payments_cvxpy = game_cvxpy.run(bids.to(device))
    allocation_qpth, payments_qpth = game_qpth.run(bids.to(device))
    
    assert torch.equal(allocation_gurobi, allocation_cvxpy), "Allocation gap between gurobi and cvxpy"
    assert torch.equal(allocation_gurobi, allocation_qpth), "Allocation gap between gurobi and qpth"
    assert torch.equal(allocation_qpth, allocation_cvxpy), "Allocation gap between qpth and cvxpy"

    assert torch.allclose(payments_gurobi, payments_cvxpy, atol = 0.01), "Payments gap between gurobi and cvxpy"
    assert torch.allclose(payments_gurobi.double(), payments_qpth, atol = 0.01), "Payments gap between gurobi and qpth"
    assert torch.allclose(payments_qpth, payments_cvxpy.double(), atol = 0.01), "Payments gap between qpth and cvxpy"

@pytest.mark.parametrize("parallel, rule, bids, device", testdata, ids=ids)
def test_LLLLGG(parallel,rule,bids, device):
    """
    Testing batch_size > 1, VCG 0 prices, FP, global/local winning
    """
    run_LLLLGG_test(parallel, rule, bids, device)
