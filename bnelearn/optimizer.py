# -*- coding: utf-8 -*-
from collections.abc import Iterable
#from collections import deque

from copy import deepcopy

import torch
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from torch.optim.optimizer import Optimizer, required #pylint: disable=no-name-in-module # false positive

from bnelearn.environment import Environment
from bnelearn.strategy import Strategy

dynamic = object()

class ES(Optimizer):
    """Implements Evolutionary Strategy similar to `Salimans et al (2017) https://arxiv.org/pdf/1703.03864.pdf`

        Compared to SGD-like optimizers, ES essentially gradients with ES-pseudo-gradients.
        This implementation extends Salimans et al by the following points:
        - The candidate weights are (optinally) calculated using a baseline, i.e.
          the weight is based on (reward - baseline) rather than just the reward.
        - Optionally, not just vanilla SGD but momentum updates are enabled,
          we use the following definition of momentum:

          delta = momentum * prev_delta + lr * pseudogradient


    Args:
        model (nn.Module): The base model that will be optimized.
            Initially needed as ipnut for knowing model architecture. After optim steps have been performed,
            this will serve as the current 'state of the art' base model and will be consequently updated.
        environment (iterable[nn.Module (or Bidder?)]): environment of strategies, number of players and
            a mechanism that permutations will be evaluated against in each optimization step.
            If given, fixed env will be used in each step (e.g. for assymetric case),
            (with possible external updatex via `update_env`)
            If none, will use a dynamic Deque of up to max_env_size most recent base models. (for symmetric case)
        params (iterable, optional): iterable of parameters to optimize or dicts defining parameter groups.
            If None, all params of `model` are updated.
            If given, should be a subset of `model.parameters()`. TODO: (implementation for subset not tested!)
        lr (float): learning rate for SGD-like update.
        momentum (float): momentum parameter.
        sigma (float): the standard deviation of perturbation noise.
        n_perturbations (int): number of perturbations created in each step
        baseline (bool or float): baseline reward used for weighting parameter noise.
            Default (True) is current utility on current batch.
        # not used: noise_size (long): length of the shared noise vector,
            default is 100000000 (~512mb at half precision or ~1gb at full precision)
        # not used: noise_type (torch.dtype): precision of noise, default is torch.half (16bit),
        max_env_size (int, optional): maximum number of simulated opponents in the environment if no
            fixed environment is specified.

        TODO: parallelize candidate creation and evaluation, given enough memory and compute,
        this should scale linearly in the number of candidates.
        Salimans et al also describe a method to efficiently scale this parallelization over machines, essentially using
        shared sampled noise as a common pseudo-number-generator with minimal need for computation between the nodes.
        (implementing this would likely be overkill and not useful in our case.)
    """

    def __init__(self, model: torch.nn.Module, environment: Environment, params=None,
                 lr=required, momentum=0, sigma=required, n_perturbations=64,
                 baseline=True, env_type=dynamic, player_position=None
                ):

        # validation checks
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Inavalid momentum parameter: {}".format(lr))
        if sigma is not required and sigma <= 0.0:
            raise ValueError("Invalid perturbation covariance: {}".format(sigma))
        if n_perturbations < 1:
            raise ValueError("Invalid number of perturbations: {}".format(n_perturbations))
        assert isinstance(baseline, (bool, int, float)), "Invalid baseline parameter."
        assert isinstance(environment, Environment), "Invalid Environment"

        if not params:
            params = model.parameters()
        else:
            raise NotImplementedError("Partial optimization of the network is not supported yet.")

        # initialize super
        defaults = dict(
            lr=lr, momentum=momentum, baseline=baseline,
            sigma=sigma, n_perturbations=n_perturbations
            )

        super(ES, self).__init__(params, defaults)

        # additional members deliberately not handled by super
        self.model = model
        self.player_position = player_position
        if environment.is_empty() and env_type is dynamic:
            # for self play, add initial model into environment
            environment.push_agent(deepcopy(model))
        self.environment = environment
        self.env_type = env_type

    def __setstate__(self, state):
        super(ES, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('momentum', 0)

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
            and returns the loss.
        """


        for group in self.param_groups:
            base_params = group['params']
            lr = group['lr']
            momentum = group['momentum']
            sigma = group['sigma']
            n_perturbations = group['n_perturbations']
            baseline = group['baseline']

            # set baseline. current reward if True
            if baseline is True: # run only for True, not for nonzero number!
                baseline = self.environment.get_reward(self.model, self.player_position).view(1)
            else: # False, Int or Float
                baseline = torch.tensor(float(baseline), device=base_params[0].device)

            # 1. Create a population of perturbations of the original model
            population = (self._perturb_model(self.model) for _ in range(n_perturbations))
            # 2. let each of these play against the environment and get their utils
            # both of these as a row-matrix. i.e.
            # rewards: n_perturbations x 1
            # epsilons: n_perturbations x parameter_length
            self.environment.prepare_iteration()
            rewards, epsilons = (
                torch.cat(tensors).view(n_perturbations, -1)
                for tensors in zip(*(
                    (
                        self.environment.get_reward(model, self.player_position).view(1),
                        epsilon
                    )
                    for (model, epsilon) in population
                    ))
                )

            # 3. calculate the gradient update
            ## TODO: fails if model not expl. on gpu because rewards is on cuda,
            #        but eps is on cpu. why?
            weighted_noise_vector = ((rewards -baseline) * epsilons).sum(dim=0)
            # create a copy of the parameters to store the updates in
            # (we need the same structure as the group params for the loop below)
            param_noise = deepcopy(base_params)
            vector_to_parameters(weighted_noise_vector, param_noise)

            # 4. iterate through the model parameters and apply the updates
            for p, p_noise in zip(group['params'], param_noise):
                #d_p = lr / n_perturbations / sigma * weighted_noise
                d_p = lr / n_perturbations / sigma * p_noise

                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        # first iteration: create momentum buffer
                        buf = param_state['momentum_buffer'] = torch.clone(d_p).detach()
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(d_p)

                    d_p = buf

                # apply the update
                p.data.add_(d_p)


        # add new model to environment in dynamic environments
        if self.env_type is dynamic:
            self.environment.push_agent(deepcopy(self.model))

        utility = self.environment.get_reward(self.model, self.player_position)
        # 5. return the loss
        return -utility

    def _perturb_model(self, model: torch.nn.Module):
        """
            Returns a randomly perturbed copy of a model [torch.nn.Module],
            as well as the noise vector used to generate the perturbation.
        """
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


class SimpleReinforce(Optimizer):
    r"""Implements simple version of REINFORCE-algorithm, i.e. SGD optimization with gradients acquired via the
        (continuous action-space and deterministic-policy case) Policy Gradient Theorem.
        (Silver et al., 2014 http://proceedings.mlr.press/v32/silver14.html)

        Currently, this suffers from the known stale-gradient problem in FPSB/vickrey auctions and does not learn!
        (Silver et al, 2014 gives convergence guarantees, however the q(s,a)-function in FPSB/Vickrey
         auctions (i.e. utility(bid, valuations of other players))
         violates condition A1 in the proof (supplementary material) due to the discontinuity
         of the allocation/payment/utility functions as b_max_-i.)

        As such, this code should be considered experimental as it has not been verified to work.

        TODO: Test in other setting
        TODO: parallelize perturbations
        TODO: possible sign-error in update step?

    Example:
        >>> optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
        >>> optimizer.zero_grad()
        >>> loss_fn(model(input), target).backward()
        >>> optimizer.step()

    """

    def __init__(self, model: torch.nn.Module, environment,
                 params=None, lr=required, sigma=required, n_perturbations=64,
                 env_type=dynamic, player_position=None):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not params:
            params = model.parameters()

        defaults = dict(lr=lr, sigma=sigma, n_perturbations=n_perturbations)
        super(SimpleReinforce, self).__init__(params, defaults)

        # additional members deliberately not handled by super
        self.model = model
        self.player_position = player_position
        if environment.is_empty() and env_type is dynamic:
            # for self play, add initial model into environment
            environment.push_agent(deepcopy(model))
        self.environment = environment
        self.env_type = env_type

    def __setstate__(self, state):
        super(SimpleReinforce, self).__setstate__(state)

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """

        if closure is not None:
            loss = closure()

        base_model = self.model
        device = next(base_model.parameters()).device
        params = self.param_groups[0]['params']
        lr = self.defaults['lr']
        n_perturbations = self.defaults['n_perturbations']


        population = (self._perturb_model(self.model) for _ in range(n_perturbations))

        self.environment.prepare_iteration()
        #reward = torch.zeros(1, device=device)

        dp = {}
        for p in params:
            dp[p] = torch.zeros_like(p.data)

        for (model, epsilon) in population:
            base_model.zero_grad()

            reward = self.environment.get_reward(model, self.player_position).view(1)
            #print("reward: {}".format(reward))
            # calculate grad
            action = self.environment._bidder_from_strategy(model).get_action()
            # perform gradient-on log action
            (action[0]).backward()

            for p in params:
                if p.grad is None:
                    continue
                dp[p].add_(p.grad.data * reward)

        for p in params:
            dp[p].div_(n_perturbations)
            # TODO: possibly missing minus?
            p.data.add_(self.param_groups[0]['lr'], dp[p])

         # add new model to environment in dynamic environments
        if self.env_type is dynamic:
            self.environment.push_agent(deepcopy(self.model))

        utility = self.environment.get_reward(self.model, self.player_position)

        return -utility

    class PerturbedActionModel(torch.nn.Module, Strategy):
        """Represents a perturbed-action version of a Strategy, used
           as a candidate in REINFORCE-policy gradient optimizer above
        """
        def __init__(self, model, epsilon):
            torch.nn.Module.__init__(self)
            self.epsilon = epsilon
            self.base_model = model

        def forward(self, x):
            x = self.base_model.forward(x)
            x.add_(self.epsilon).relu_()
            return x

        def play(self,inputs):
            return self.forward(inputs)

    def _perturb_model(self, model: torch.nn.Module) -> torch.nn.Module:

        sigma = self.defaults['sigma']

        epsilon = torch.zeros(1, device=next(self.model.parameters()).device).normal_(mean=0.0, std=sigma)
        perturbed = self.PerturbedActionModel(model, epsilon)

        return perturbed, epsilon
