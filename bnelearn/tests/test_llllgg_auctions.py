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

expected_allocation_3 = torch.tensor([[
        #B1,B2,B3,B4,B5,B6,B7,B8,B9,B10,B11,B12
        [0,0], #L1
        [1,0], #L2
        [1,0], #L3
        [0,0], #L4
        [0,0], #G1
        [0,1], #G2
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

# each test input takes form rule: string, bids:torch.tensor,
#                            expected_allocation: torch.tensor, expected_payments: torch.tensor
ids, testdata = zip(*[
    ['vcg - multi-batch', ('vcg', bids_1, expected_allocation_1, torch.Tensor(
                                        [[0.0, 0.0, 0.0, 0.0, 3.0, 0.0],
                                         [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                         [1.0, 2.0, 0.0, 0.0, 0.0, 0.0]]))],
    ['vcg - single-batch', ('vcg', bids_2, expected_allocation_2, torch.Tensor([[0.4, 0.0, 0.3, 0.4, 0.0, 0.0]]))],
    #['vcg - neg. bids and nn.SELU()', ('vcg', bids_3, expected_allocation_3, torch.Tensor([[0.5, 0.7, 1.5, 0.0, 0.9, 1.1]]))]
])

def run_LLLLGG_test(rule, device, bids, expected_allocation, expected_payments):
    """Run correctness test for a given LLLLGG rule"""
    cuda = device == 'cuda' and torch.cuda.is_available()

    if device == 'cuda' and not cuda:
        pytest.skip("This test needs CUDA, but it's not available.")

    game = LLLLGGAuction(batch_size = len(bids), rule = rule, cuda=cuda)
    allocation, payments = game.run(bids.to(device))

    assert torch.equal(allocation, expected_allocation.to(device)), "Wrong allocation"
    assert torch.allclose(payments, expected_payments.to(device)), "Wrong payments"

@pytest.mark.parametrize("rule,bids,expected_allocation,expected_payments", testdata, ids=ids)
def test_LLLLGG(rule,bids,expected_allocation,expected_payments):
    """
    Testing batch_size > 1, VCG 0 prices, global/local winning
    """
    run_LLLLGG_test(rule, 'cuda', bids, expected_allocation, expected_payments)
    run_LLLLGG_test(rule, 'cpu', bids, expected_allocation, expected_payments)
