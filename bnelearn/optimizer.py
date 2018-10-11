import torch
from torch.optim.optimizer import Optimizer, required
from copy import deepcopy

class ES(Optimizer):
    r"""Implements Evolutionary Strategy as in `Salimans et al (2017) https://arxiv.org/pdf/1703.03864.pdf`

    Args:
        model (nn.Module): The base Module that will be optimized.
        params (iterable, optional): iterable of parameters to optimize or dicts defining parameter groups.
            If None, all params of `model` are updated.
            If given, should be a subset of `model.parameters()`. TODO: Not yet implemented!
        lr (float): learning rate for SGD-like update.
        sigma (float): the standard deviation of perturbation noise.
        n_perturbations (integer): number of perturbations created in each step
        noise_size (long): length of the shared noise vector, 
            default is 100000000 (~512mb at half precision or ~1gb at full precision)
        noise_type (torch.dtype): precision of noise, default is torch.half (16bit)
    """

    def __init__(self, model: torch.nn.Module, params=None,
                 lr = required, sigma = required, n_perturbations=64,
                 noise_size=100000000, noise_type = torch.half):
        
        # validation checks
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if sigma is not required and sigma <= 0.0:
            raise ValueError("Invalid perturbation covariance: {}".format(sigma))
        if n_perturbations < 1:
            raise ValueError("Invalid number of perturbations: {}".format(n_perturbations))

        if not params:
            params = model.parameters()
        else:
            raise NotImplementedError("Partial optimization of the network is not supported yet.")

        # initialize super
        defaults = dict(lr=lr, sigma=sigma, n_perturbations=n_perturbations,
                        noise_size=noise_size, noise_type=noise_type)

        super(ES, self).__init__(params, defaults)
        if len(self.param_groups) > 1:
            raise NotImplementedError("Multiple Parameter groups found. ES only currently only supports a single group.")
        
        # additional members deliberately not handled by super
        self.model = model
        self._initialize_noise()

    def step(self, closure = None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model 
            and returns the loss.
        """        

        base_params = self.param_groups[0]['params']
        lr = self.defaults['lr']
        simga = self.defaults['sigma']
        n_perturbations = self.defaults['n_perturbations']

        # 1. Create a population of perturbations of the original model
        population = [deepcopy(self.model) for _ in range(n_perturbations)]

        

        print(base_params)
        print(population)
        # 2. let each of these play against the environment and get their utils
        # 3. calculate the gradient update
        # 4. apply the gradient update to the base model
        # 5. return the loss (how? why?)
        return None

    def _initialize_noise():
        device = next(self.model.parameters()).device
        size = self.defaults['noise_size']
        dtype = self.defaults['noise_type']
        sigma = self.defaults['sigma']
        return torch.zeros(size, dtype = dtype, device=device).normal_(mean=0.0, std=sigma)

    def _delete_noise():
        del self.noise




