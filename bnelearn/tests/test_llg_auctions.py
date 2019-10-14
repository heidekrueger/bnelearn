"""Testing correctness of LLG combinatorial auction implementations."""
import pytest
import torch
from bnelearn.mechanism import LLGAuction

bids = torch.tensor([
    [1., 1., 2.1], # global bidder wins
    [8., 6., 10.], # see Ausubel and Baranov 2019 Fig 2
    [12., 6., 10]  # same
    ]).unsqueeze(-1)

expected_allocation = torch.tensor([
        [ [0], [0], [1] ],
        [ [1], [1], [0] ],
        [ [1], [1], [0]]
    ], dtype=torch.float)


def run_llg_test(rule, device, expected_payments):
    """Run correctness test for a given llg rule on given device"""
    cuda = device == 'cuda' and torch.cuda.is_available()

    if device == 'cuda' and not cuda:
        pytest.skip("This test needs CUDA, but it's not available.")

    game = LLGAuction(rule = rule, cuda=cuda)
    allocation, payments = game.run(bids.to(device))

    assert torch.equal(allocation, expected_allocation.to(device))
    assert torch.equal(payments, expected_payments.to(device))


def test_LLG_first_price():
    """FP should run on CPU and GPU and return expected results."""

    rule = 'first_price'
    expected_payments = torch.tensor([
        [0. , 0., 2.1],
        [8., 6., 0.],
        [12., 6., 0.]])

    run_llg_test(rule, 'cpu', expected_payments)
    run_llg_test(rule, 'cuda', expected_payments)


def test_LLG_vcg():
    """LLG with VCG rule should run on CPU and GPU and return expected results."""

    rule = 'vcg'
    expected_payments = torch.tensor([
        [0. , 0., 2.0],
        [4., 2., 0.],
        [4., 0., 0.]])

    run_llg_test(rule, 'cpu', expected_payments)
    run_llg_test(rule, 'cuda', expected_payments)

def test_LLG_proxy():
    """LLG with proxy rule should run on CPU and GPU and return expected results."""

    rule = 'proxy'
    expected_payments = torch.tensor([
        [0. , 0., 2.0],
        [5., 5., 0.],
        [5., 5., 0.]])

    run_llg_test(rule, 'cpu', expected_payments)
    run_llg_test(rule, 'cuda', expected_payments)

def test_LLG_nearest_vcg():
    """LLG with nearest-VCG rule should run on CPU and GPU and return expected results."""

    rule = 'nearest_vcg'
    expected_payments = torch.tensor([
        [0. , 0., 2.],
        [6., 4., 0.],
        [7., 3., 0.]])

    run_llg_test(rule, 'cpu', expected_payments)
    run_llg_test(rule, 'cuda', expected_payments)

def test_LLG_nearest_bid():
    """LLG with nearest-bid rule should run on CPU and GPU and return expected results."""

    rule = 'nearest_bid'
    expected_payments = torch.tensor([
        [0. , 0., 2.],
        [6., 4., 0.],
        [8., 2., 0.]])

    run_llg_test(rule, 'cpu', expected_payments)
    run_llg_test(rule, 'cuda', expected_payments)
