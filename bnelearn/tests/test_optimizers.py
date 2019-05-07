import warnings
import pytest
import torch
from bnelearn.strategy import NeuralNetStrategy
from bnelearn.mechanism import StaticMechanism
from bnelearn.bidder import Bidder
from bnelearn.optimizer import ES, SimpleReinforce
from bnelearn.environment import AuctionEnvironment

"""Setup shared objects"""

cuda = torch.cuda.is_available()
device = 'cuda' if cuda else 'cpu'
BATCH_SIZE = 2**20
SIZE_HIDDEN_LAYER = 20
input_length = 1

epoch = 1000
LEARNING_RATE = 2e-2
lr_decay = True
lr_decay_every = 2000
lr_decay_factor = 0.8

sigma = .1 #ES noise parameter
n_perturbations = 32

u_lo = 0
u_hi = 10

def strat_to_bidder(strategy, batch_size):
    return Bidder.uniform(u_lo,u_hi, strategy, batch_size = batch_size, n_players=1)

mechanism = StaticMechanism(cuda=cuda)

# TODO: write tests
def test_static_mechanism():
    """Test whether the mechanism for testing the optimizers returns expected results"""

    if BATCH_SIZE < 2**15:
        pytest.skip("Batch size too low to perform this test!")

    model = NeuralNetStrategy(
        input_length,
        size_hidden_layer = SIZE_HIDDEN_LAYER,
        requires_grad=False
        ).to(device)
    bidder = strat_to_bidder(model, BATCH_SIZE)

    bidder.valuations.zero_().add_(10)
    bids = bidder.valuations.clone().detach().view(-1,1,1)
    bids.zero_().add_(10)

    allocations, payments = mechanism.run(bids)
    # subset for single player
    allocations = allocations[:,0,:].view(-1, 1)
    payments = payments[:,0].view(-1)
    utilities = bidder.get_utility(allocations=allocations, payments=payments)

    assert torch.isclose(utilities.mean(), torch.tensor(5., device=device), atol=1e-2), \
        "StaticMechanism returned unexpected rewards."

    bids.add_(-5)
    allocations, payments = mechanism.run(bids)
    # subset for single player
    allocations = allocations[:,0,:].view(-1, 1)
    payments = payments[:,0].view(-1)
    utilities = bidder.get_utility(allocations=allocations, payments=payments)
    assert torch.isclose(utilities.mean(), torch.tensor(3.75, device=device), atol=1e-2), \
        "StaticMechanism returned unexpected rewards."

def test_ES_optimizer():
    model = NeuralNetStrategy(input_length, size_hidden_layer = SIZE_HIDDEN_LAYER, requires_grad=False).to(device)
    bidder = strat_to_bidder(model, BATCH_SIZE)
    env = AuctionEnvironment(
        mechanism,
        agents = [bidder],
        strategy_to_bidder_closure=strat_to_bidder,
        max_env_size=1,
        batch_size = BATCH_SIZE,
        n_players=1
        )

    optimizer = ES(
        model=model,
        environment = env,
        lr = LEARNING_RATE,
        sigma=sigma,
        n_perturbations=n_perturbations
        )

    torch.cuda.empty_cache()

    for e in range(epoch+1):

        # lr decay?
        if lr_decay and e % lr_decay_every == 0 and e > 0:
            learning_rate = learning_rate * lr_decay_factor
            for param_group in optimizer.param_groups:
                param_group['lr'] = learning_rate
            #writer.add_scalar('hyperparams/learning_rate', learning_rate, e)

        #print(list(env._generate_opponent_bids()))
        # always: do optimizer step
        utility = -optimizer.step()
        #writer.add_scalar('eval/utility', utility, e)

        # plot + eval
        if e % 10 == 0:
            # plot current function output
            bidder = strat_to_bidder(model, BATCH_SIZE)
            bidder.draw_valuations_()
            v = bidder.valuations
            b = bidder.get_action()
            share = (b/v).mean()
            diff = (b-v).mean()
            #writer.add_scalar('eval/utility', utility, e)
            #writer.add_scalar('eval/share', share, e)

            print("Epoch {}: \tavg bid: share {:2f}, diff {:2f},\tutility: {:2f}".format(e, share, diff, utility))



    torch.cuda.empty_cache()
