## false positives for Iterable and Optimizer
#pylint: disable=E0611
from collections import Iterable, deque
from torch.optim.optimizer import Optimizer, required
#pylint: disable=E0611
import torch
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from bnelearn.environment import Environment
from copy import deepcopy

dynamic = object()

class ES(Optimizer):
    """Implements Evolutionary Strategy as in `Salimans et al (2017) https://arxiv.org/pdf/1703.03864.pdf`

    Args:
        model (nn.Module): The base model that will be optimized.
            Initially needed as ipnut for knowing model architecture. After optim steps have been performed,
            this will serve as the current 'state of the art' base model and will be consequently updated.
        environment (iterable[nn.Module (or Bidder?)]): environment of strategies, number of players and 
            a mechanism that permutations will be evaluated against in each optimization step.
            If given, fixed env will be used in each step (e.g. for assymetric case), (with possible external updatex via `update_env`)
            If none, will use a dynamic Deque of up to max_env_size most recent base models. (for symmetric case)
        params (iterable, optional): iterable of parameters to optimize or dicts defining parameter groups.
            If None, all params of `model` are updated.
            If given, should be a subset of `model.parameters()`. TODO: Not yet implemented!
        lr (float): learning rate for SGD-like update.
        sigma (float): the standard deviation of perturbation noise.
        n_perturbations (integer): number of perturbations created in each step
        # not used: noise_size (long): length of the shared noise vector, 
            default is 100000000 (~512mb at half precision or ~1gb at full precision)
        # not used: noise_type (torch.dtype): precision of noise, default is torch.half (16bit),
        max_env_size (int, optional): maximum number of simulated opponents in the environment if no 
            fixed environment is specified. 
    """

    def __init__(self, model: torch.nn.Module, environment: Environment, params=None,
                 lr=required, sigma=required, n_perturbations=64, env_type=dynamic,
                 noise_size=100000000, noise_type=torch.half):

        # validation checks
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if sigma is not required and sigma <= 0.0:
            raise ValueError("Invalid perturbation covariance: {}".format(sigma))
        if n_perturbations < 1:
            raise ValueError("Invalid number of perturbations: {}".format(n_perturbations))
        assert isinstance(environment, Environment), "Invalid Environment"

        if not params:
            params = model.parameters()
        else:
            raise NotImplementedError("Partial optimization of the network is not supported yet.") 

        # initialize super
        defaults = dict(lr=lr, sigma=sigma, n_perturbations=n_perturbations,
                        noise_size=noise_size, noise_type=noise_type)

        super(ES, self).__init__(params, defaults)

        if len(self.param_groups) > 1:
            raise NotImplementedError("Multiple Parameter groups found. Currently only supports a single group.")

        # additional members deliberately not handled by super
        self.model = model
        if environment.is_empty() and env_type is dynamic:
            # for self play, add model into environemtn
            environment.push_agent(deepcopy(model))
        self.environment = environment
        self.env_type = env_type
        # do not use shared noise for now
        # self._initialize_noise()

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model 
            and returns the loss.
        """

        base_params = self.param_groups[0]['params']
        lr = self.defaults['lr']
        sigma = self.defaults['sigma']
        n_perturbations = self.defaults['n_perturbations']

        # 1. Create a population of perturbations of the original model
        population = (self._perturb_model(self.model) for _ in range(n_perturbations))
        # 2. let each of these play against the environment and get their utils
        # both of these as a row-matrix. i.e.
        # rewards: n_perturbations x 1
        # epsilons: n_perturbations x parameter_length
        self.environment.draw_valuations_()
        rewards, epsilons = (torch.cat(tensors).view(n_perturbations, -1)
                             for tensors in zip(*(
                                (self.environment.get_reward(model).view(1),
                                 epsilon)
                                for (model, epsilon) in population
                                ))
                            )

        # 3. calculate the gradient update
        weighted_noise = (rewards * epsilons).sum(dim=0)
        #print(weighted_noise / n_perturbations / sigma)
        new_base_params = \
            parameters_to_vector(base_params) + \
            lr / n_perturbations / sigma * weighted_noise


        # 4. apply the gradient update to the base model
        vector_to_parameters(new_base_params, base_params)
        vector_to_parameters(new_base_params, self.model.parameters())

        # add new model to environment
        if self.env_type is dynamic:
            self.environment.push_agent(deepcopy(self.model))

        utility = self.environment.get_reward(self.model)
        # 5. return the loss
        return -utility

    ## not using shared noise matrix for now
    #def _initialize_noise(self):
    #    device = next(self.model.parameters()).device
    #    size = self.defaults['noise_size']
    #    dtype = self.defaults['noise_type']
    #    sigma = self.defaults['sigma']
    #    return torch.zeros(size, dtype = dtype, device=device).normal_(mean=0.0, std=sigma)

    #def _delete_noise(self):
    #    del self.noise


    def _perturb_model(self, model: torch.nn.Module):
        sigma = self.defaults['sigma']
        perturbed = deepcopy(model)

        params_flat = parameters_to_vector(model.parameters())
        noise = torch.zeros_like(params_flat).normal_(mean=0.0, std=sigma)
        # copy perturbed params into copy
        vector_to_parameters(params_flat + noise, perturbed.parameters())

        return perturbed, noise

    def _update_env(self, new_env: Iterable):
        for agent in new_env:
            self.environment.push_agent(agent)
