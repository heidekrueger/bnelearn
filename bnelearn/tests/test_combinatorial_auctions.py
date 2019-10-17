import pytest
import torch
from bnelearn.mechanism import LLLLGGAuction

"""Testing correctness of LLLLGG combinatorial auction implementations."""
bids = torch.tensor([
    #Bundle1, Bundle2
    [1,1], #L1
    [1,1], #L2
    [1,1], #L3
    [1,1], #L4
    [4,3], #G1
    [3,3], #G2
    ], dtype=torch.float)

expected_allocation = torch.tensor([
    #B1,B2,B3,B4,B5,B6,B7,B8,B9,B10,B11,B12
    [0,0,0,0,0,0,0,0,0,0,0,0], #L1
    [0,0,0,0,0,0,0,0,0,0,0,0], #L2
    [0,0,0,0,1,0,0,0,0,0,0,0], #L3
    [0,0,0,0,0,0,1,0,0,0,0,0], #L4
    [0,0,0,0,0,0,0,0,1,0,0,0], #G1
    [0,0,0,0,0,0,0,0,0,0,0,0], #G2
    ], dtype=torch.float)

def run_combinatorial_test(rule, device, expected_VCG_payments):
    """Run correctness test for a given LLLLGG rule"""
    cuda = device == 'cuda' and torch.cuda.is_available()

    if device == 'cuda' and not cuda:
        pytest.skip("This test needs CUDA, but it's not available.")

    game = LLLLGGAuction(rule = rule, cuda=cuda)
    allocation, payments = game.run(bids.to(device))

    assert allocation == expected_allocation.tolist(), "Wrong allocation"
    assert payments == expected_VCG_payments, "Wrong payments"

def test_combinatorial_nearest_vcg():
    """"""

    rule = 'nearest_vcg'
    expected_VCG_payments = [0,0,0,0,3,0]
    run_combinatorial_test(rule, 'cpu', expected_VCG_payments)
    #run_llg_test(rule, 'cuda', expected_payments)