import pytest
import torch
from bnelearn.mechanism import LLLLGGAuction

"""Testing correctness of LLLLGG combinatorial auction implementations."""
bids_1 = torch.tensor([[
        #Bundle1, Bundle2
        [1,1], #L1
        [1,1], #L2
        [1,1], #L3*
        [1,1], #L4*
        [4,3], #G1*
        [3,3], #G2
    ],[
        [4,1], #L1*
        [6,3], #L2*
        [3,1], #L3*
        [5,1], #L4*
        [4,3], #G1
        [3,3], #G2
    ],[
        [4,1], #L1*
        [5,3], #L2*
        [3,1], #L3*
        [5,1], #L4*
        [6,3], #G1
        [3,3], #G2
    ],
    [
        [4,1], #L1*
        [5,3], #L2*
        [3,4], #L3*
        [5,1], #L4*
        [6,3], #G1
        [3,3], #G2
    ]], dtype=torch.float)

bids_2 = torch.tensor([[
    #Bundle1, Bundle2
        [0.1,0.9], #L1
        [0.1,0.1], #L2
        [0.9,0.1], #L3
        [0.1,0.9], #L4
        [1.1,1.1], #G1
        [1.2,1.2], #G2
    ]], dtype=torch.float)

#Only when allowing negative bids (nn.SELU())
bids_3 = torch.tensor([[
    #Bundle1, Bundle2
        [-0.5,1.0], #L1
        [0.9,1.0], #L2
        [0.9,-0.7], #L3
        [0.1,0.9], #L4
        [-0.9,1.0], #G1
        [1.2,1.2], #G2
    ]], dtype=torch.float)

bids_4 = torch.tensor([[
    #Bundle1, Bundle2
        [0.0,28.0], #L1*
        [0.0,20.0], #L2*
        [12.0,0.0], #L3
        [0.0,0.0], #L4
        [0.0,0.0], #G1
        [32.0,0.0], #G2
    ]], dtype=torch.float)

bids_5 = torch.tensor([[
    #Bundle1, Bundle2
        [0.0,0.0], #L1*
        [0.0,0.0], #L2*
        [0.0,0.0], #L3
        [0.0,0.0], #L4
        [0.0,0.0], #G1
        [0.0,0.0], #G2
    ]], dtype=torch.float)

expected_allocation_1 = torch.tensor([[
    #B1,B2,B3,B4,B5,B6,B7,B8,B9,B10,B11,B12
        [0,0], #L1
        [0,0], #L2
        [1,0], #L3
        [1,0], #L4
        [1,0], #G1
        [0,0], #G2
    ],[
        [1,0], #L1
        [1,0], #L2
        [1,0], #L3
        [1,0], #L4
        [0,0], #G1
        [0,0], #G2
    ],[
        [1,0], #L1
        [1,0], #L2
        [1,0], #L3
        [1,0], #L4
        [0,0], #G1
        [0,0], #G2
    ],[
        [1,0], #L1
        [1,0], #L2
        [1,0], #L3
        [1,0], #L4
        [0,0], #G1
        [0,0], #G2
    ]], dtype=torch.float)

expected_allocation_2 = torch.tensor([[
        #B1,B2,B3,B4,B5,B6,B7,B8,B9,B10,B11,B12
        [0,1], #L1
        [0,0], #L2
        [1,0], #L3
        [0,1], #L4
        [0,0], #G1
        [0,0], #G2
    ]], dtype=torch.float)

expected_allocation_3 = torch.tensor([[
        #B1,B2,B3,B4,B5,B6,B7,B8,B9,B10,B11,B12
        [0,0], #L1
        [1,0], #L2
        [1,0], #L3
        [0,0], #L4
        [0,0], #G1
        [0,1], #G2
    ]], dtype=torch.float)

expected_allocation_4 = torch.tensor([[
        #B1,B2,B3,B4,B5,B6,B7,B8,B9,B10,B11,B12
        [0,1], #L1
        [0,1], #L2
        [0,0], #L3
        [0,0], #L4
        [0,0], #G1
        [0,0], #G2
    ]], dtype=torch.float)

expected_allocation_5 = torch.tensor([[
        #B1,B2,B3,B4,B5,B6,B7,B8,B9,B10,B11,B12
        [0,0], #L1
        [0,0], #L2
        [0,0], #L3
        [0,0], #L4
        [0,0], #G1
        [0,0], #G2
    ]], dtype=torch.float)

# each test input takes form rule: string, bids:torch.tensor,
#                            expected_allocation: torch.tensor, expected_payments: torch.tensor
# Each tuple specified here will then be tested for all implemented solvers.
ids, testdata = zip(*[
    ['vcg - multi-batch', (1,'vcg', bids_1, expected_allocation_1, torch.Tensor(
                                          [[0.0, 0.0, 0.0, 0.0, 3.0, 0.0],
                                          [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                          [1.0, 2.0, 0.0, 0.0, 0.0, 0.0],
                                          [1.0, 2.0, 0.0, 1.0, 0.0, 0.0]]))],
    ['vcg - single-batch', (1,'vcg', bids_2, expected_allocation_2, torch.Tensor([[0.4, 0.0, 0.3, 0.4, 0.0, 0.0]]))],
    ['vcg - single-batch', (1,'vcg', bids_4, expected_allocation_4, torch.Tensor([[12.0, 12.0, 0.0, 0.0, 0.0, 0.0]]))],
    ['vcg - single-batch - zero_bids', (1,'vcg', bids_5, expected_allocation_5, torch.Tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]))],
    ['fp - multi-batch', (1,'first_price', bids_1, expected_allocation_1, torch.Tensor(
                                          [[0.0, 0.0, 1.0, 1.0, 4.0, 0.0],
                                          [4.0, 6.0, 3.0, 5.0, 0.0, 0.0],
                                          [4.0, 5.0, 3.0, 5.0, 0.0, 0.0],
                                          [4.0, 5.0, 3.0, 5.0, 0.0, 0.0]]))],
    ['fp - single-batch', (1,'first_price', bids_2, expected_allocation_2, torch.Tensor([[0.9, 0.0, 0.9, 0.9, 0.0, 0.0]]))],
    ['fp - single-batch - zero_bids', (1,'first_price', bids_5, expected_allocation_5, torch.Tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]))],
    #['fp - neg. bids and nn.SELU()', (1,'first_price', bids_3, expected_allocation_3, torch.Tensor([[0.5, 0.7, 1.5, 0.0, 0.9, 1.1]]))],

    ['nearest vcg - single-batch', (1,'nearest-vcg', bids_4, expected_allocation_4, torch.Tensor([[16.0, 16.0, 0.0, 0.0, 0.0, 0.0]]))],
    ['nearest vcg - single-batch - zero_bids', (1,'nearest-vcg', bids_5, expected_allocation_5, torch.Tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]))],
    ['nearest vcg - multi-batch', (1,'nearest-vcg', bids_1, expected_allocation_1, torch.tensor([
                                                                                    [0.0, 0.0, 0.5, 0.5, 3.5, 0.0],
                                                                                    [2.0, 2.0, 1.5, 1.5, 0.0, 0.0],
                                                                                    [2.5, 3.5, 1.5, 1.5, 0.0, 0.0],
                                                                                    [2.5, 3.5, 1.0, 2.0, 0.0, 0.0]], dtype = torch.float))],
    ['nearest vcg - single-batch - multi_core', (2,'nearest-vcg', bids_4, expected_allocation_4, torch.Tensor([[16.0, 16.0, 0.0, 0.0, 0.0, 0.0]]))],
    ['nearest vcg - multi-batch - multi_core', (2,'nearest-vcg', bids_1, expected_allocation_1, torch.tensor([
                                                                                    [0.0, 0.0, 0.5, 0.5, 3.5, 0.0],
                                                                                    [2.0, 2.0, 1.5, 1.5, 0.0, 0.0],
                                                                                    [2.5, 3.5, 1.5, 1.5, 0.0, 0.0],
                                                                                    [2.5, 3.5, 1.0, 2.0, 0.0, 0.0]], dtype = torch.float))]                                                                                
])

def run_LLLLGG_test(parallel, rule, device, bids, expected_allocation, expected_payments, solver):
    """Run correctness test for a given LLLLGG rule"""
    cuda = device == 'cuda' and torch.cuda.is_available()

    if device == 'cuda' and not cuda:
        pytest.skip("This test needs CUDA, but it's not available.")

    game = LLLLGGAuction(batch_size = len(bids), rule = rule, cuda=cuda, core_solver=solver, parallel=parallel)
    allocation, payments = game.run(bids.to(device))

    assert torch.equal(allocation, expected_allocation.to(device)), "Wrong allocation"
    if payments.dtype == torch.double:
        expected_payments = expected_payments.double()
    assert torch.allclose(payments, expected_payments.to(device), atol = 0.001), \
           "Unexpected payments returned by solver " + solver

@pytest.mark.parametrize("parallel, rule,bids,expected_allocation,expected_payments", testdata, ids=ids)
def test_LLLLGG(parallel, rule,bids,expected_allocation,expected_payments):
    """
    Testing batch_size > 1, VCG 0 prices, FP, global/local winning
    """
    run_LLLLGG_test(parallel, rule, 'cpu', bids, expected_allocation, expected_payments, 'gurobi')
    run_LLLLGG_test(parallel, rule, 'cpu', bids, expected_allocation, expected_payments, 'cvxpy')
    run_LLLLGG_test(parallel, rule, 'cpu', bids, expected_allocation, expected_payments, 'qpth')

    run_LLLLGG_test(parallel, rule, 'cuda', bids, expected_allocation, expected_payments, 'gurobi')
    run_LLLLGG_test(parallel, rule, 'cuda', bids, expected_allocation, expected_payments, 'cvxpy')
    run_LLLLGG_test(parallel, rule, 'cuda', bids, expected_allocation, expected_payments, 'qpth')
    

