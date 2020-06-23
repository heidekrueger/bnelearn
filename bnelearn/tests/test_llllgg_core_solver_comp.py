import pytest
import torch
from bnelearn.mechanism import LLLLGGAuction

"""Testing for identical results from solvers for the LLLLGG combinatorial auction implementations."""
torch.manual_seed(1)
torch.cuda.manual_seed(1)
bids_1 = torch.rand([2**9,6,2], dtype = torch.float)
# With larger batch sizes both only manage 0.01
# Console: pytest bnelearn/tests/test_llllgg_core_solver_comp.py -s
# (parallel, payment rule, bids, device)
ids, testdata = zip(*[
    ['nearest vcg - single - cpu', (1,'nearest_vcg', bids_1, 'cpu')],
    ['nearest vcg - multi - cpu', (8,'nearest_vcg', bids_1, 'cpu')],
    ['nearest vcg - single - gpu', (1,'nearest_vcg', bids_1, 'cuda')],
    ['nearest vcg - multi - gpu', (8,'nearest_vcg', bids_1, 'cuda')],
])

def run_LLLLGG_test(parallel, rule, bids, device, solver_1, solver_2):
    """Run comparison test for different solvers."""
    cuda = device == 'cuda' and torch.cuda.is_available()

    if device == 'cuda' and not cuda:
        pytest.skip("This test needs CUDA, but it's not available.")

    game_solver_1 = LLLLGGAuction(rule=rule, cuda=cuda, core_solver=solver_1, parallel=parallel)
    game_solver_2 = LLLLGGAuction(rule=rule, cuda=cuda, core_solver=solver_2, parallel=parallel)

    allocation_solver_1, payments_solver_1 = game_solver_1.run(bids.to(device))
    allocation_solver_2, payments_solver_2 = game_solver_2.run(bids.to(device))

    assert torch.equal(allocation_solver_1, allocation_solver_2), \
        "Allocation gap between {} and {}".format(solver_1,solver_2)
    assert torch.allclose(payments_solver_1.double(), payments_solver_2, atol = 0.01), \
        "Payments gap between {} and {}".format(solver_1,solver_2)

@pytest.mark.parametrize("parallel, rule, bids, device", testdata, ids=ids)
def test_LLLLGG(parallel,rule,bids, device):
    """
    Testing batch_size > 1, VCG 0 prices, FP, global/local winning
    """
    #run_LLLLGG_test(parallel, rule, bids, device, 'gurobi','cvxpy')
    run_LLLLGG_test(parallel, rule, bids, device, 'gurobi','qpth')
    run_LLLLGG_test(parallel, rule, bids, device, 'gurobi','mpc')
