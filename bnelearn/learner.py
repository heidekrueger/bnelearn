"""Implements multi-agent learning rules"""

import warnings
from copy import deepcopy

from abc import ABC, abstractmethod
from typing import Tuple, Type, Callable

import torch
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from bnelearn.environment import Environment


class Learner(ABC):
    """A learning rule used to update a player's policy in self-play"""

    @abstractmethod
    def update_strategy(self) -> None:
        """Updates the player's strategy."""
        raise NotImplementedError()

class GradientBasedLearner(Learner):
    """A learning rule that is based on computing some version of (pseudo-)
       gradient, then applying an SGD-like update via a `torch.optim.Optimizer`
    """
    def __init__(self,
                 model: torch.nn.Module, environment: Environment,
                 optimizer_type: Type[torch.optim.Optimizer], optimizer_hyperparams: dict,
                 strat_to_player_kwargs: dict = None):
        self.model = model
        self.params = model.parameters
        self.n_parameters = sum([p.numel() for p in self.params()])

        self.environment = environment

        self.strat_to_player_kwargs = strat_to_player_kwargs if strat_to_player_kwargs else {}
            # warn if weird initialization
        if 'player_position' not in self.strat_to_player_kwargs.keys():
            warnings.warn('You haven\'t specified a player_position to evaluate the model. Defaulting to position 0.')
            self.strat_to_player_kwargs['player_position'] = 0

        if not isinstance(optimizer_hyperparams, dict):
            raise ValueError('Optimizer hyperparams must be a dict (even if empty).')
        self.optimizer_hyperparams = optimizer_hyperparams
        self.optimizer: torch.optim.Optimizer = optimizer_type(self.params(), **self.optimizer_hyperparams)

    @abstractmethod
    def _set_gradients(self):
        """Calculate current (pseudo)gradient for all params."""

    def update_strategy(self, closure: Callable=None) -> None or torch.Tensor: # pylint: disable=arguments-differ
        """Performs one model-update to the player's strategy.

        Params:
            closure: (optional) Callable that recomputes model loss.
                Required by some optimizers such as LBFGS. When given,
                optimizer.step() (and thus this function) return the last
                evaluated loss. (Usually evaluated BEFORE the model update).
                For correct usage see:
                https://pytorch.org/docs/stable/optim.html#optimizer-step-closure

        Returns: None or loss evaluated by closure. (See above.)
        """

        self.optimizer.zero_grad()
        self._set_gradients()
        return self.optimizer.step(closure=closure)

    def update_strategy_and_evaluate_utility(self, closure=None):
        """updates model and returns utility after the update."""
        self.update_strategy(closure)
        return self.environment.get_strategy_reward(
            self.model,
            **self.strat_to_player_kwargs
            ).detach()

class ESPGLearner(GradientBasedLearner):
    """ Neural Self-Play with Evolutionary Strategy Pseudo-PG

    Uses pseudo-policy gradients calculated as

            ```(rewards - baseline).mean() * epsilons / sigma²```

    over a population of models perturbed by parameter noise epsilon yielding
    perturbed rewards.

    Arguments:
        model: bnelearn.bidder
        environment: bnelearn.Environment
        hyperparams: dict
            (required:)
                population_size: int
                sigma: float
                scale_sigma_by_model_size: bool
            (optional:)
                normalize_gradients: bool (default: False)
                    If true will scale rewards to N(0,1) in weighted-noise update:
                    (F - baseline).mean()/sigma/F.std() resulting in an (approximately)
                    normalized vector pointing in the same direction as the true
                    gradient. (normalization requires small enough sigma!)
                    If false or not provided, will approximate true gradient
                    using current utility as a baseline for variance reduction.
                baseline: ('current_reward', 'mean_reward' or a float.)
                    If 'current_reward', will use current utility before update as a baseline.
                    If 'mean_reward', will use mean of candiate rewards.
                        For small perturbations, 'mean_reward' is cheaper to compute
                        (one fewer game played) and yields slightly lower gradient
                        sample variance but yields a biased estimate of the true gradient:

                            Expect(ES_grad with mean) = (pop_size - 1) / pop_size * true_grad

                    If a float is given, will use that float as reward.
                    Defaults to 'current_reward' if normalize_gradients is False, or
                    to 'mean_reward' if normalize_gradients is True.

        optimizer_type: Type[torch.optim.Optimizer]
            A class implementing torch's optimizer interface used for parameter update step.
        strat_to_player_kwargs: dict
                dict of arguments provided to environment used for evaluating
                utility of current and candidate strategies.
    """
    def __init__(self,
                 model: torch.nn.Module, environment: Environment, hyperparams: dict,
                 optimizer_type: Type[torch.optim.Optimizer], optimizer_hyperparams: dict,
                 strat_to_player_kwargs: dict = None):
        # Create and validate optimizer
        super().__init__(model, environment,
                         optimizer_type, optimizer_hyperparams,
                         strat_to_player_kwargs)

        # Validate ES hyperparams
        if not set(['population_size', 'sigma', 'scale_sigma_by_model_size']) <= set(hyperparams):
            raise ValueError(
                'Missing hyperparams for ES. Provide at least, population size, sigma and scale_sigma_by_model_size.')
        if not isinstance(hyperparams['population_size'], int) or hyperparams['population_size'] < 2:
            # one is invalid because there will be zero variance, leading to div by 0 errors
            raise ValueError('Please provide a valid `population_size` parameter >=2')

        # set hyperparams
        self.population_size = hyperparams['population_size']
        self.sigma = float(hyperparams['sigma'])
        self.sigma_base = self.sigma
        if hyperparams['scale_sigma_by_model_size']:
            self.sigma = self.sigma / self.n_parameters

        if 'normalize_gradients' in hyperparams and hyperparams['normalize_gradients']:
            self.normalize_gradients = True
            self.baseline = 'mean_reward'
        else:
            self.normalize_gradients = False
            self.baseline = 'current_reward'

        # overwrite baseline method if provided
        if 'baseline' in hyperparams:
            self.baseline_method = hyperparams['baseline']
            if not isinstance(self.baseline_method, float) \
                    and not self.baseline_method in ['current_reward', 'mean_reward']:
                raise ValueError('Invalid baseline provided. Should be float or '\
                    + 'one of "mean_reward", "current_reward"')


    def _set_gradients(self):
        """Calculates ES-pseudogradients and applies them to the model parameter
           gradient data.

            ES gradient is calculated as:
            mean( rewards - baseline) * epsilons / sigma²
            and approximates the true gradient.

            In case of gradient normalization, we do not calculate a baseline
            and instead use the following pseudogradient:
            mean(rewards - rewards.mean()) / sigma / rewards.std()
            For small sigma, this will yield a vector that points in the same
            direction as the gradient and has length (slightly smaller than) 1.
            Furthermore, the gradient samples will have low variance
            Note that for large sigma, this grad becomes smaller tha
        """

        ### 1. if required redraw valuations / perform random moves (determined by env)
        self.environment.prepare_iteration()
        ### 2. Create a population of perturbations of the original model
        population = (self._perturb_model(self.model) for _ in range(self.population_size))
        ### 3. let each candidate against the environment and get their utils ###
        # both of these as a row-matrix. i.e.
        # rewards: population_size x 1
        # epsilons: population_size x parameter_length

        rewards, epsilons = (
            torch.cat(tensors).view(self.population_size, -1)
            for tensors in zip(*(
                (
                    self.environment.get_strategy_reward(
                        model, **self.strat_to_player_kwargs).detach().view(1),
                    epsilon
                )
                for (model, epsilon) in population
                ))
            )
        ### 4. calculate the ES-pseuogradients   ####
        # See ES_Analysis notebook in repository for more information about where
        # these choices come from.
        baseline = \
            self.environment.get_strategy_reward(self.model,**self.strat_to_player_kwargs).detach().view(1) \
                if self.baseline == 'current_reward' \
            else rewards.mean(dim=0) if self.baseline == 'mean_reward' \
            else self.baseline # a float

        denominator = self.sigma * rewards.std() if self.normalize_gradients else self.sigma**2

        if denominator == 0:
            # all candidates returned same reward and normalize is true --> stationary
            gradient_vector = torch.zeros_like(parameters_to_vector(self.params()))
        else:
            gradient_vector = ((rewards - baseline)*epsilons).mean(dim=0) / denominator

        # put gradient vector into same format as model parameters
        gradient_params = deepcopy(list(self.params()))
        vector_to_parameters(gradient_vector, gradient_params)

        ### 5. assign gradients to model gradient ####
        # We actually _add_ to existing gradient (as common in pytorch), to make it
        # possible to accumulate gradients over multiple batches.
        # When this is not desired (most of the time!), you need to flush the gradients
        # before calling this method.

        # NOTE: torch.otpimizers minimize but we use a maximization formulation
        # in the rewards, thus we need to use the negative gradient here.

        for p, d_p in zip(self.params(), gradient_params):
            if p.grad is not None:
                p.grad.add_(-d_p)
            else:
                p.grad = -d_p

    def _perturb_model(self, model: torch.nn.Module) -> Tuple[torch.nn.Module, torch.Tensor]:
        """
        Returns a randomly perturbed copy of a model [torch.nn.Module],
        as well as the noise vector used to generate the perturbation.
        """
        perturbed = deepcopy(model)

        params_flat = parameters_to_vector(model.parameters())
        noise = torch.zeros_like(params_flat).normal_(mean=0.0, std=self.sigma)
        # copy perturbed params into copy
        vector_to_parameters(params_flat + noise, perturbed.parameters())

        return perturbed, noise

class DPGLearner(GradientBasedLearner):
    """Neural Self-Play with Deterministic Policy Gradients."""
    def __init__(self):
        super().__init__()
        raise NotImplementedError()

    def _set_gradients(self):
        raise NotImplementedError()