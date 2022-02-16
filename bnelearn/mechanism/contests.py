import torch

from typing import Callable, Tuple

from .mechanism import Mechanism

# helper functions
def csf_all_pay(bids: torch.Tensor, player_dim: int, allocations: torch.Tensor, *args):

    _, winning_bidders = bids.max(dim=player_dim, keepdim=True)  # shape of each: [batch_size, 1, n_items]

    # highest bidder wins with probability 1
    allocations.scatter_(player_dim, winning_bidders, 1)

    # tie breaking rule -> first by indix gets bid (should occur with prop = 0, and thus, not be a problem)
    return allocations

## logit transformations
### tullock 
def _tullock_effects(investments: torch.Tensor, alpha):
    return investments ** alpha

### additive noise h() = e + phi, where phi is random noise
def _additive_noise(investments: torch.Tensor, alpha):
    return investments + alpha

### gradstein's multiplicative form
# TODO: alpha is individual for all bidders
def _multiplicative_noise(investments: torch.Tensor, q):
    return investments * q 

### gradstein + tullock (corchon2010foundations)
def _generalized_tullock(investments: torch.tensor, q, alpha):
    return (investments * q) + alpha 

# Generalized implementation of logit (additive form)
def csf_logit(bids: torch.Tensor, player_dim: int, allocations: torch.Tensor, transformation: Callable = None):
    
    # if no transformation is given, use identity function
    if transformation is None:
        transformation = lambda x: x

    # applay transformation of efforts
    transformed_bids = transformation(bids)

    # determine sum of all transformed efforts for each prize as denimator
    denominator = transformed_bids.transpose(-1, player_dim).sum(-1).unsqueeze(-1)

    winning_probs = torch.div(transformed_bids, denominator)
    
    return winning_probs

# based on hirshleifer 
def csf_relative_difference(bids: torch.Tensor, player_dim: int, allocations: torch.Tensor, transformation: Callable = None):

    # TODO dict für die ganzen csf params machen, dann in den entsprechendne funktionen die parameter extrahieren.
    # TODO extend to multiple prices..
    alpha = 0.1
    beta = 1
    s = 0.2

    bids_t = bids.transpose(-1, player_dim)


    # two player setting
    if bids.shape[player_dim] == 2:

        try:

            bids[bids.sum(dim=player_dim).squeeze()>0] = torch.stack((alpha + beta * ((bids_t[bids_t.sum(dim=-1)>0][..., 0] - s*bids_t[bids_t.sum(dim=-1)>0][..., 1])/(bids_t.sum(dim=-1).squeeze())), 
                                                                    alpha + beta * ((bids_t[bids_t.sum(dim=-1)>0][..., 1] - s*bids_t[bids_t.sum(dim=-1)>0][..., 0])/(bids_t.sum(dim=-1).squeeze())))) \
                                                                    .transpose(-1, 0).unsqueeze(-1)

            bids[bids.sum(dim=player_dim).squeeze()==0] = 0.5
        except:
            print(2)
 
    else:
        pass

    return bids

# probit model 

# che and gale (piece-wise linear difference form)

#### PAYMENT RULES
def payment_first_price(bids: torch.Tensor, *args):

    payments = bids.sum(-1)
    return payments    

def payment_second_price(bids: torch.Tensor, player_dim: int):

    # Update winners to pay only second highest price
    ## Get best k=2 values
    k_highest, _ = bids.topk(k=2, dim=player_dim)
    second_highest, _ = k_highest.min(dim=player_dim, keepdim=True)

    _, highest_bidder = bids.max(dim=player_dim, keepdim=True)

    payments = bids.scatter(player_dim, highest_bidder, second_highest)

    # Summarize payments
    payments = payments.sum(-1)

    return payments 

class Contest(Mechanism):

    def __init__(self, cuda: bool = True, payment_rule: str = None, csf: str = None):

        assert payment_rule in ["first_price", "second_price"]
        assert csf in ["all_pay", "logit", "relative_difference"]
        
        if payment_rule == "first_price":
            self.payment_rule = payment_first_price
        else:
            self.payment_rule = payment_second_price

        if csf == "all_pay":
            self.csf = csf_all_pay
        elif csf == "logit":
            self.csf = csf_logit
        elif csf == "relative_difference":
            self.csf = csf_relative_difference

        super().__init__(cuda=cuda)

    def run(self, bids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:

        # Part 1: Determine allocation based on pre-defined CSF
        assert bids.dim() >= 3, "Bid tensor must be at least 3d (*batch_dims x players x items)"
        assert (bids >= 0).all().item(), "All bids must be nonnegative."

        # move bids to gpu/cpu if necessary
        bids = bids.to(self.device)

        # name dimensions
        *batch_dims, player_dim, item_dim = range(bids.dim())  # pylint: disable=unused-variable
        *batch_sizes, n_players, n_items = bids.shape

        allocations = torch.zeros(*batch_sizes, n_players, n_items, device=bids.device)
        payments = torch.zeros(*batch_sizes, n_players, n_items, device=bids.device)

        ## Determine probabilities of winning using CSF
        winning_probs = self.csf(bids, player_dim, allocations)

        ## determine random allocation based on winning probs
        dist = torch.distributions.Multinomial(probs=winning_probs.transpose(item_dim, player_dim))
        sample = dist.sample()
        try:
            winners = torch.nonzero((sample[..., :] > 0)).reshape(*batch_sizes, n_items, len(bids.shape))[..., -1]
        except:
            print(2)

        # TODO multi dimensional batch sizes does not work here....

        try:
            allocations.scatter_(player_dim, winners.unsqueeze(-1), 1)
        except:
            print("a")

        # sanity check
        if (allocations.sum(player_dim) > 1).any():
            print("there is still an error...")


        # Part 2: Determine payments based on pre-defindes pricing_rule 
        payments = self.payment_rule(bids, player_dim)

        return allocations, payments