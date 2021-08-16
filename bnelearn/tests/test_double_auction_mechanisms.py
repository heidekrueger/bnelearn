import pytest
import torch
from torch._C import Value
from bnelearn.mechanism.double_auctions_single_item import kDoubleAuction, VickreyDoubleAuction


bids = torch.tensor([
    [1., 0.5], # buyer bids more than ask price
    [7., 3.], # same
    [3., 3.], # equal bid and ask price
    [4. , 6.],  # buyer bids less than ask price
    ]).unsqueeze(-1)

expected_allocation = torch.tensor([
        [ [1], [1] ],
        [ [1], [1] ],
        [ [1], [1] ],
        [ [0], [0] ]
    ], dtype=torch.float)

def get_da_mechanism(rule: str, n_buyers, n_sellers, k, cuda):
    if rule == 'k_price':
        return kDoubleAuction(n_buyers=n_buyers, n_sellers=n_sellers, k=k, cuda=cuda)
    elif rule == 'vickrey_price':
        return VickreyDoubleAuction(n_buyers=n_buyers, n_sellers=n_sellers, cuda=cuda)
    else:
        raise ValueError('No valid double auction mechanism type chosen!')


def run_BilateralBargaining_mechanism_test(rule: str, k: float, device, expected_payments):
    """Run correctness test for the kDoubleAuction in the bilarteral bargaining case"""
    cuda = device == 'cuda' and torch.cuda.is_available()
    n_buyers = 1
    n_sellers = 1

    if device == 'cuda' and not cuda:
        pytest.skip("This test needs CUDA, but it's not available.")

    game = get_da_mechanism(rule=rule, n_buyers=n_buyers, n_sellers=n_sellers, k=k, cuda=cuda)
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

def test_BilateralBargaining_kDoubleAuction():
    """FP should run on CPU and GPU and return expected results."""

    k_0_expected_payments = torch.tensor([
        [0.5 , 0.5],
        [3. , 3.],
        [3. , 3.],
        [0. , 0.]
        ])
    
    k_05_expected_payments = torch.tensor([
        [0.75 , 0.75],
        [5. , 5.],
        [3. , 3.],
        [0. , 0.]
        ])

    k_1_expected_payments = torch.tensor([
        [1. , 1.],
        [7. , 7.],
        [3. , 3.],
        [0. , 0.]
        ])

    run_BilateralBargaining_mechanism_test(rule='k_price', k=0., device='cpu', expected_payments=k_0_expected_payments)
    run_BilateralBargaining_mechanism_test(rule='k_price', k=0., device='cuda', expected_payments=k_0_expected_payments)

    run_BilateralBargaining_mechanism_test(rule='k_price', k=0.5, device='cpu', expected_payments=k_05_expected_payments)
    run_BilateralBargaining_mechanism_test(rule='k_price', k=0.5, device='cuda', expected_payments=k_05_expected_payments)

    run_BilateralBargaining_mechanism_test(rule='k_price', k=1., device='cpu', expected_payments=k_1_expected_payments)
    run_BilateralBargaining_mechanism_test(rule='k_price', k=1., device='cuda', expected_payments=k_1_expected_payments)