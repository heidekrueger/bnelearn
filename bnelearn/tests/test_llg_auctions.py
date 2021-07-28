"""Testing correctness of LLG combinatorial auction implementations."""
import pytest
import torch
from bnelearn.mechanism import LLGAuction, LLGFullAuction
import warnings

@pytest.fixture(autouse=True)
def check_gurobipy():
    pytest.importorskip('gurobipy')      
    if not pytest.gurobi_licence_valid:
        warnings.warn("The Gurobipy is installed but no valid licence available, the test will fail")

bids = torch.tensor([
    [1., 1., 2.1], # global bidder wins
    [8., 6., 10.], # see Ausubel and Baranov 2019 Fig 2
    [12., 6., 10], # same
    [4. , 6., 9],  # weak first player in proxy rule
    [6. , 4. ,9.]  # weak second player in proxy rule
    ]).unsqueeze(-1)

expected_allocation = torch.tensor([
        [ [0], [0], [1] ],
        [ [1], [1], [0] ],
        [ [1], [1], [0] ],
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

    # Test whether the auction also accepts multiple batch dimensions
    def add_dim(tensor):
        repititions = 2
        return tensor.clone().unsqueeze(0).repeat_interleave(repititions,0)

    allocation, payments = game.run(add_dim(bids.to(device)))

    assert torch.equal(allocation, add_dim(expected_allocation.to(device))), \
        """auction did not handle multiple batch dims correctly!"""
    assert torch.equal(payments, add_dim(expected_payments.to(device))), \
        """auction did not handle multiple batch dims correctly!"""


def test_LLG_first_price():
    """FP should run on CPU and GPU and return expected results."""

    rule = 'first_price'
    expected_payments = torch.tensor([
        [0. , 0., 2.1],
        [8., 6., 0.],
        [12., 6., 0.],
        [4., 6., 0.],
        [6., 4., 0.]])

    run_llg_test(rule, 'cpu', expected_payments)
    run_llg_test(rule, 'cuda', expected_payments)


def test_LLG_vcg():
    """LLG with VCG rule should run on CPU and GPU and return expected results."""

    rule = 'vcg'
    expected_payments = torch.tensor([
        [0. , 0., 2.0],
        [4., 2., 0.],
        [4., 0., 0.],
        [3., 5., 0.],
        [5., 3., 0.]])

    run_llg_test(rule, 'cpu', expected_payments)
    run_llg_test(rule, 'cuda', expected_payments)

def test_LLG_proxy():
    """LLG with proxy rule should run on CPU and GPU and return expected results."""

    rule = 'proxy'
    expected_payments = torch.tensor([
        [0. , 0., 2.0],
        [5., 5., 0.],
        [5., 5., 0.],
        [4., 5., 0.],
        [5., 4., 0.]])

    run_llg_test(rule, 'cpu', expected_payments)
    run_llg_test(rule, 'cuda', expected_payments)

def test_LLG_nearest_vcg():
    """LLG with nearest-VCG rule should run on CPU and GPU and return expected results."""

    rule = 'nearest_vcg'
    expected_payments = torch.tensor([
        [0. , 0., 2.],
        [6., 4., 0.],
        [7., 3., 0.],
        [3.5, 5.5, 0.],
        [5.5, 3.5, 0.]])

    run_llg_test(rule, 'cpu', expected_payments)
    run_llg_test(rule, 'cuda', expected_payments)

def test_LLG_nearest_bid():
    """LLG with nearest-bid rule should run on CPU and GPU and return expected results."""

    rule = 'nearest_bid'
    expected_payments = torch.tensor([
        [0. , 0., 2.],
        [6., 4., 0.],
        [8., 2., 0.],
        [3.5, 5.5, 0.],
        [5.5, 3.5, 0.]])

    run_llg_test(rule, 'cpu', expected_payments)
    run_llg_test(rule, 'cuda', expected_payments)


llgfull_bids = torch.tensor(
    [[[1, 1, 0], [0, 2, 2], [0, 0, 2]],
     [[0, 1, 0], [1, 3, 0], [0, 0, 1]],
     [[2, 2, 0], [0, 2, 2], [0, 0, 6]],
     [[2, 2, 0], [0, 0, 0], [4, 4, 0]],
     [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
     [[1, 1, 4], [0, 0, 0], [0, 0, 0]],
     [[.7, .2, .4], [.2, .5, .4], [2, 2, 1]],
     [[1, 0, 2], [0, 0, 1], [2, 2, 0]],
     [[3, 0, 2], [1, 1, 3], [0, 2, 1]]],
    dtype=torch.float)

llgfull_allocations = torch.tensor(
    [[[1, 0, 0], [0, 1, 0], [0, 0, 0]],
     [[0, 0, 0], [1, 1, 0], [0, 0, 0]],
     [[0, 0, 0], [0, 0, 0], [0, 0, 1]],
     [[0, 0, 0], [0, 0, 0], [1, 1, 0]],
     [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
     [[0, 0, 1], [0, 0, 0], [0, 0, 0]],
     [[0, 0, 0], [0, 0, 0], [1, 1, 0]],
     [[0, 0, 0], [0, 0, 0], [1, 1, 0]],
     [[1, 0, 0], [0, 0, 0], [0, 1, 0]]],
    dtype=torch.int8)

llgfull_payments_vcg = torch.tensor(
    [[0.0, 1.0, 0.0],
     [0.0, 1.0, 0.0],
     [0.0, 0.0, 4.0],
     [0.0, 0.0, 4.0],
     [0.0, 0.0, 0.0],
     [0.0, 0.0, 0.0],
     [0.0, 0.0, 1.2],
     [0.0, 0.0, 2.0],
     [1.0, 0.0, 1.0]]
)

llgfull_payments_nearest_vcg = torch.tensor(
    [[0.5, 1.5, 0.0],
     [0.0, 1.0, 0.0],
     [0.0, 0.0, 4.0],
     [0.0, 0.0, 4.0],
     [0.0, 0.0, 0.0],
     [0.0, 0.0, 0.0],
     [0.0, 0.0, 1.2],
     [0.0, 0.0, 2.0],
     [1.5, 0.0, 1.5]],
)

llgfull_payments_mrcs_favored = torch.tensor(
    [[1.0, 1.0, 0.0],
     [0.0, 1.0, 0.0],
     [0.0, 0.0, 4.0],
     [0.0, 0.0, 4.0],
     [0.0, 0.0, 0.0],
     [0.0, 0.0, 0.0],
     [0.0, 0.0, 1.2],
     [0.0, 0.0, 2.0],
     [1.0, 0.0, 1.0]],
)

def test_LLG_full():
    """LLG setting with complete combinatrial (3d) bids."""
    # TODO Nils: Warning - pricing rule seems to be not deterministic!
    #            Watch e.g. last instance of `llgfull_payments_mrcs_favored`
    device = 'cuda'

    # VCG
    vcg_mechanism = LLGFullAuction(rule='vcg', cuda=device)
    allocations, payments_vcg_computed = vcg_mechanism.run(llgfull_bids.to(device))
    assert torch.equal(allocations, llgfull_allocations.to(device))
    assert torch.equal(payments_vcg_computed, llgfull_payments_vcg.to(device))

    # Nearest VCG
    nearest_vcg_mechanism = LLGFullAuction(rule='nearest_vcg', cuda=device)
    _, payments_llgfull_computed = nearest_vcg_mechanism.run(llgfull_bids.to(device))
    assert torch.allclose(payments_llgfull_computed, llgfull_payments_nearest_vcg.to(device),
                          atol=0.0001)

    # Favors bidder 1: she pays VCG prices
    mrcs_favored_mechanism = LLGFullAuction(rule='mrcs_favored', cuda=device)
    _, payments_favored_computed = mrcs_favored_mechanism.run(llgfull_bids.to(device))
    assert torch.allclose(payments_favored_computed, llgfull_payments_mrcs_favored.to(device),
                          atol=0.001)
    assert torch.allclose(payments_vcg_computed[:, 1], payments_favored_computed[:, 1],
                          atol=0.0001), 'agent 1 should pay VCG prices'
