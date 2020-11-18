import pytest
import torch
from bnelearn.mechanism import CombinatorialAuction

"""Testing correctness of LLLLGG combinatorial auction implementations."""
bids_1 = torch.tensor([[
    #Bundle1 (one item)
    [2], #L1
    [1], #L2
    [1.5], #L3
    ]], dtype=torch.float)

bids_2 = torch.tensor([[
    #Bundle1, Bundle2, Bundle3
    [0.9,0.0,0.0], #L1
    [0.0,0.9,0.0], #L2
    [0.0,0.0,1.9], #G1
    ]], dtype=torch.float)

expected_allocation_1 = torch.tensor([[
    #B1,B2,B3,B4,B5,B6,B7,B8,B9,B10,B11,B12
    [1], #L1
    [0], #L2
    [0], #L3
    ]], dtype=torch.float)

expected_allocation_2 = torch.tensor([[
    #B1,B2,B3,B4,B5,B6,B7,B8,B9,B10,B11,B12
    [0,0,0], #L1
    [0,0,0], #L2
    [0,0,1], #G2
    ]], dtype=torch.float)

bundles_1 = torch.tensor([
    #A
    [1] #B1
    ])

bundles_2 = torch.tensor([
    #A,B
    [1,0], #B1
    [0,1], #B2
    [1,1], #B3
    ])

# each test input takes form rule: string, bids:torch.tensor, bundles: torch.tensor,
#                            expected_allocation: torch.tensor, expected_payments: torch.tensor
ids, testdata = zip(*[
    ['vcg - single-item', ('vcg', bids_1, bundles_1, expected_allocation_1, torch.tensor([[1.5, 0.0, 0.0]]))],
    ['vcg - multi-item', ('vcg', bids_2, bundles_2, expected_allocation_2, torch.tensor([[0.0, 0.0, 1.8]]))]
])

def run_Combinatorial_test(rule, device, bids, bundle, expected_allocation, expected_VCG_payments):
    """Run correctness test for a given LLLLGG rule"""
    cuda = device == 'cuda' and torch.cuda.is_available()

    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip("CUDA not available. skipping...")

    game = CombinatorialAuction(rule = rule, cuda=cuda, bundles = bundle)
    allocation, payments = game.run(bids.to(device))

    assert torch.equal(allocation, expected_allocation.to(device)), "Wrong allocation"
    assert torch.allclose(payments, expected_VCG_payments.to(device)), "Wrong payments"

@pytest.mark.parametrize("rule,bids,bundles,expected_allocation,expected_payments", testdata, ids=ids)
def test_Combinatorial(rule,bids,bundles,expected_allocation,expected_payments):
    """ Tests allocation and payments in combinatorial auctions"""
    run_Combinatorial_test(rule, 'cpu', bids, bundles, expected_allocation, expected_payments)
    run_Combinatorial_test(rule, 'cuda', bids, bundles, expected_allocation, expected_payments)