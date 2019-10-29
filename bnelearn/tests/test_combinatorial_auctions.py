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

bundles_1 = torch.Tensor([
    #A
    [1] #B1
    ])

bundles_2 = torch.Tensor([
    #A,B
    [1,0], #B1
    [0,1], #B2
    [1,1], #B3
    ])

def run_Combinatorial_test(rule, device, bids, bundle, expected_allocation, expected_VCG_payments):
    """Run correctness test for a given LLLLGG rule"""
    cuda = device == 'cuda' and torch.cuda.is_available()

    if device == 'cuda' and not cuda:
        pytest.skip("This test needs CUDA, but it's not available.")

    game = CombinatorialAuction(rule = rule, cuda=cuda, bundles = bundle)
    allocation, payments = game.run(bids.to(device))

    assert torch.equal(allocation, expected_allocation.to('cuda')), "Wrong allocation"
    assert torch.equal(payments, expected_VCG_payments.to('cuda')), "Wrong payments"

def test_1_Combinatorial_vcg():
    """"""
    rule = 'vcg'
    expected_VCG_payments = torch.Tensor([[1.5, 0.0, 0.0]])
    run_Combinatorial_test(rule, 'cuda', bids_1, bundles_1, expected_allocation_1, expected_VCG_payments)

def test_2_Combinatorial_vcg():
    """"""
    rule = 'vcg'
    expected_VCG_payments = torch.Tensor([[0.0, 0.0, 1.8]])
    run_Combinatorial_test(rule, 'cuda', bids_2, bundles_2, expected_allocation_2, expected_VCG_payments)