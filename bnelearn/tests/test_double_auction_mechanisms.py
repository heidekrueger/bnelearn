import pytest
import torch
from bnelearn.mechanism.double_auctions_single_item import kDoubleAuction, VickreyDoubleAuction


# ###### Bilateral Bargaining Test Scenario ######## #
bids_1b1s = torch.tensor([
    [1., 0.5], # buyer bids more than ask price
    [7., 3.], # same
    [3., 3.], # equal bid and ask price
    [4. , 6.],  # buyer bids less than ask price
    ]).unsqueeze(-1)

expected_allocation_1b1s = torch.tensor([
        [ [1], [1] ],
        [ [1], [1] ],
        [ [1], [1] ],
        [ [0], [0] ]
    ], dtype=torch.float)


# ###### 1 Buyer, 2 Sellers Tests Scenario ######## #
bids_1b2s = torch.tensor([
    [1., .5, .3], # buyer bids more than both sellers
    [7., 3., 5.], # same
    [4., 6., 7.],  # buyer bids less than ask price
    [5., 7., 3.] # Buyer bids more than one seller
    ]).unsqueeze(-1)

expected_allocation_1b2s = torch.tensor([
        [ [1], [0], [1] ],
        [ [1], [1], [0] ],
        [ [0], [0], [0] ],
        [ [1], [0], [1] ]
    ], dtype=torch.float)


def get_da_mechanism(rule: str, n_buyers, n_sellers, k, cuda):
    if rule == 'k_price':
        return kDoubleAuction(n_buyers=n_buyers, n_sellers=n_sellers, k_value=k, cuda=cuda)
    elif rule == 'vickrey_price':
        return VickreyDoubleAuction(n_buyers=n_buyers, n_sellers=n_sellers, cuda=cuda)
    else:
        raise ValueError('No valid double auction mechanism type chosen!')


def run_DoubleAuction_mechanism_test(rule: str, device, expected_payments, bids, expected_allocation, k: float=0.5, n_buyers: int=1, n_sellers: int=1):
    """Run correctness test for the kDoubleAuction in the bilarteral bargaining case"""
    cuda = device == 'cuda' and torch.cuda.is_available()

    if device == 'cuda' and not cuda:
        pytest.skip("This test needs CUDA, but it's not available.")

    game = get_da_mechanism(rule=rule, n_buyers=n_buyers, n_sellers=n_sellers, k=k, cuda=cuda)
    allocation, payments = game.run(bids.to(device))

    assert torch.equal(allocation, expected_allocation.to(device))
    assert torch.equal(payments, expected_payments.to(device))

    # Test whether the auction also accepts multiple batch dimensions
    run_test_for_multiple_batch_dimension(device, expected_payments, game, bids, expected_allocation)

def run_test_for_multiple_batch_dimension(device, expected_payments, game, bids, expected_allocation):
    def add_dim(tensor):
        repititions = 2
        return tensor.clone().unsqueeze(0).repeat_interleave(repititions,0)

    allocation, payments = game.run(add_dim(bids.to(device)))

    assert torch.equal(allocation, add_dim(expected_allocation.to(device))), \
        """auction did not handle multiple batch dims correctly!"""
    assert torch.equal(payments, add_dim(expected_payments.to(device))), \
        """auction did not handle multiple batch dims correctly!"""

def test_BilateralBargaining_kDoubleAuction():
    """k Double Auction should run on CPU and GPU and return expected results."""

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

    run_DoubleAuction_mechanism_test(rule='k_price', k=0., device='cpu', expected_payments=k_0_expected_payments, bids=bids_1b1s, expected_allocation=expected_allocation_1b1s)
    run_DoubleAuction_mechanism_test(rule='k_price', k=0., device='cuda', expected_payments=k_0_expected_payments, bids=bids_1b1s, expected_allocation=expected_allocation_1b1s)

    run_DoubleAuction_mechanism_test(rule='k_price', k=0.5, device='cpu', expected_payments=k_05_expected_payments, bids=bids_1b1s, expected_allocation=expected_allocation_1b1s)
    run_DoubleAuction_mechanism_test(rule='k_price', k=0.5, device='cuda', expected_payments=k_05_expected_payments, bids=bids_1b1s, expected_allocation=expected_allocation_1b1s)

    run_DoubleAuction_mechanism_test(rule='k_price', k=1., device='cpu', expected_payments=k_1_expected_payments, bids=bids_1b1s, expected_allocation=expected_allocation_1b1s)
    run_DoubleAuction_mechanism_test(rule='k_price', k=1., device='cuda', expected_payments=k_1_expected_payments, bids=bids_1b1s, expected_allocation=expected_allocation_1b1s)

def test_BilateralBargaining_vickreyAuction():
    """Vickrey Double Auction should run on CPU and GPU and return expected results."""

    expected_payments = torch.tensor([
        [0.5 , 1.0],
        [3. , 7.],
        [3. , 3.],
        [0. , 0.]
        ])

    run_DoubleAuction_mechanism_test(rule='vickrey_price', device='cpu', expected_payments=expected_payments, bids=bids_1b1s, expected_allocation=expected_allocation_1b1s)
    run_DoubleAuction_mechanism_test(rule='vickrey_price', device='cuda', expected_payments=expected_payments, bids=bids_1b1s, expected_allocation=expected_allocation_1b1s)


"""def test_one_buyer_two_sellers_kDoubleAuction():

    k_0_expected_payments = torch.tensor([
        [.3 , .0, .3],
        [3. , 3., .0],
        [0. , 0., 0.],
        [3. , 0., 3.]
        ])
    
    k_05_expected_payments = torch.tensor([
        [0.65 , 0., 0.65],
        [5. , 5., 0.],
        [0. , 0., 0.],
        [4. , 0., 4.]
        ])

    k_1_expected_payments = torch.tensor([
        [1. , 0., 1.],
        [7. , 7., 0.],
        [0. , 0., 0.],
        [5. , 0., 5.]
        ])

    run_DoubleAuction_mechanism_test(rule='k_price', k=0., device='cpu', expected_payments=k_0_expected_payments, n_buyers=1, n_sellers=2)
    run_DoubleAuction_mechanism_test(rule='k_price', k=0., device='cuda', expected_payments=k_0_expected_payments, n_buyers=1, n_sellers=2)

    run_DoubleAuction_mechanism_test(rule='k_price', k=0.5, device='cpu', expected_payments=k_05_expected_payments, n_buyers=1, n_sellers=2)
    run_DoubleAuction_mechanism_test(rule='k_price', k=0.5, device='cuda', expected_payments=k_05_expected_payments, n_buyers=1, n_sellers=2)

    run_DoubleAuction_mechanism_test(rule='k_price', k=1., device='cpu', expected_payments=k_1_expected_payments, n_buyers=1, n_sellers=2)
    run_DoubleAuction_mechanism_test(rule='k_price', k=1., device='cuda', expected_payments=k_1_expected_payments, n_buyers=1, n_sellers=2)"""
