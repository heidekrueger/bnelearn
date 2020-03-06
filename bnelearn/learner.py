"""Implements multi-agent learning rules"""

import warnings
from copy import deepcopy

from abc import ABC, abstractmethod
from typing import Tuple, Type, Callable

import torch
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from bnelearn.environment import Environment
from scripts.utils_nils import stopping_criterion
import numpy as np

class ExperienceReplay():
    """
    Helper class for ´Learner´ which memorizes past opponent´s policies.
    Inspired partly by ´Prioritized Experience Replay´ by Tom Schaul et al. [1].

        TODO:
            - might be closer ro other method not experience replay!
            - faster access to memory with binary heap data structure

        Args:
            n_memory_size: int, max capacity of meorized policies
            replay_prob: float, probabilty of playing against remembered environment
                compared to playing against current environment. 
            prioritization: float, >= 0, alpha in [1], 0 corresponding to the uniform case.
            forgetting: str, rule to override memories.
            eps: float, as in [1] value for remembering of any memory with prpbabilty
                greater zero.
    """
    def __init__(
            self,
            n_memory_size: int = 16,
            replay_prob: float = 0.5,
            prioritization: float = 1.0,
            forgetting: str = 'prioritized',
            eps: float = 1e-13,
        ):
        assert n_memory_size > 0, '´n_memory_size´ must be positive ineteger.'
        assert replay_prob >= 0 and replay_prob <= 1, '´replay_prob´ must be within [0, 1].'
        assert prioritization >= 0, 'prioritization must be within [0, 1].'

        self.n_memory_size = n_memory_size
        self.replay_prob = replay_prob
        self.forgetting = forgetting
        self.prioritization = prioritization
        self.eps = eps
        self.n_memories = 0

        self.current_memory_index = -1
        self.experienced_strategies = []

        if self.forgetting == 'fifo':
            self.memory_position = 0

    def _create_memory_entry(self, priority, agent):
        return [
            agent.player_position,
            priority + self.eps,
            deepcopy(agent.strategy),
            self.n_memories
        ]

    def append_memory(self, environment, priority, player_position):
        """
        Add new memory

        Parameters:
            environment: environment, in which we are in.
            priority: float, how much we learned based on this experience. 
            player_position: int, position of this player. All other players will be
                momorized.
        """
        for agent in (a for a in environment.agents if a.player_position != player_position):
            if self.n_memories < self.n_memory_size:
                self.experienced_strategies.append(self._create_memory_entry(priority, agent))
                self.n_memories += 1
            else: # have to forget a memory
                if self.forgetting == 'prioritized':
                    memory_position = np.argmin([i[1] for i in self.experienced_strategies])
                    if priority < self.experienced_strategies[memory_position][1]:
                        # new memory is discarded due to low priority
                        return
                elif self.forgetting == 'random':
                    memory_position = np.random.randint(self.n_memory_size)
                elif self.forgetting == 'fifo':
                    memory_position = self.memory_position
                    self.memory_position = (self.memory_position + 1) % self.n_memory_size
                elif self.forgetting == 'ignore':
                    return
                else:
                    raise NotImplementedError
                self.experienced_strategies[memory_position] = self._create_memory_entry(priority, agent)

    def get_memory(
            self,
            player_position
        ) -> Tuple[int, torch.nn.Module]:
        """Access memory"""
        pos = [i for i, p in enumerate(self.experienced_strategies)
               if p[0] == player_position]

        if len(pos) == 0:
            return -1, None

        # choose memory based on priorities
        denominator = sum(mem[1] ** self.prioritization for mem in (self.experienced_strategies[i] for i in pos))
        drawn_choice = np.random.rand()
        choice_level = 0
        i = 0
        while i < len(pos) and choice_level < drawn_choice:
            choice_level += (self.experienced_strategies[pos[i]][1] ** self.prioritization) / denominator
            i += 1
        i -= 1

        strategy = self.experienced_strategies[pos[i]][2]

        self.current_memory_index = pos[i]
        return player_position, strategy

    def set_priority(self, memory_position, priority):
        """Update priority of latly accessed memory"""
        self.experienced_strategies[memory_position][1] = priority + self.eps

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
                 lr_scheduler = None, strat_to_player_kwargs: dict = None,
                 experience: ExperienceReplay = None):
        self.model = model
        self.params = model.parameters
        self.n_parameters = sum([p.numel() for p in self.params()])

        self.environment = environment
        self.experience = experience
        self.gradient_vector = torch.zeros_like(parameters_to_vector(self.params()))

        self.strat_to_player_kwargs = strat_to_player_kwargs if strat_to_player_kwargs else {}
            # warn if weird initialization
        if 'player_position' not in self.strat_to_player_kwargs.keys():
            warnings.warn('You haven\'t specified a player_position to evaluate the model. Defaulting to position 0.')
            self.strat_to_player_kwargs['player_position'] = 0

        if not isinstance(optimizer_hyperparams, dict):
            raise ValueError('Optimizer hyperparams must be a dict (even if empty).')
        self.optimizer_hyperparams = optimizer_hyperparams
        self.optimizer: torch.optim.Optimizer = optimizer_type(self.params(), **self.optimizer_hyperparams)
        if lr_scheduler is not None:
            self.lr_scheduler = lr_scheduler['type'](self.optimizer, 'max', **lr_scheduler['params'])
        else:
            self.lr_scheduler = None

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

        return self.optimizer.step(closure=closure)

    def update_strategy_and_evaluate_utility(
            self,
            closure = None,
            return_allocation = False,
            learning: bool = True
        ):
        """updates model and returns utility after the update."""

        if learning:
            # self.update_strategy(closure)
            self.prev_gradient = self.gradient_vector
            self.gradient_norm = 0
            self.optimizer.zero_grad()
            rewards, stopping = self._set_gradients()
        else:
            rewards, stopping = 0, 0

        reward = self.environment.get_strategy_reward(
                self.model,
                return_allocation = return_allocation,
                **self.strat_to_player_kwargs
            )

        if learning and self.lr_scheduler is not None:
            self.lr_scheduler.step(reward if not return_allocation else reward[0])

        return (rewards,) + reward + (stopping,)

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
                 lr_scheduler = None, strat_to_player_kwargs: dict = None,
                 experience: ExperienceReplay = None):
        # Create and validate optimizer
        super().__init__(model, environment,
                         optimizer_type, optimizer_hyperparams, lr_scheduler,
                         strat_to_player_kwargs, experience)

        # gradient from previous iteration
        self.prev_gradient = torch.zeros_like(parameters_to_vector(self.params())) 

        # noprm of the gradient
        self.gradient_norm = 0

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

        replay = self.experience if (self.experience is not None
            and np.random.rand() < self.experience.replay_prob
            and self.experience.n_memories > 0) else None

        rewards, epsilons = (
            torch.cat(tensors).view(self.population_size, -1)
            for tensors in zip(*(
                (
                    self.environment.get_strategy_reward(
                        model, experience=replay,
                        **self.strat_to_player_kwargs).detach().view(1),
                    epsilon
                )
                for (model, epsilon) in population
                ))
            )
        ### 4. calculate the ES-pseuogradients   ####
        # See ES_Analysis notebook in repository for more information about where
        # these choices come from.
        baseline = \
            self.environment.get_strategy_reward(self.model, **self.strat_to_player_kwargs).detach().view(1) \
                if self.baseline == 'current_reward' \
            else rewards.mean(dim=0) if self.baseline == 'mean_reward' \
            else self.baseline # a float

        denominator = self.sigma * rewards.std() if self.normalize_gradients else self.sigma**2

        if denominator == 0:
            # all candidates returned same reward and normalize is true --> stationary
            self.gradient_vector = torch.zeros_like(parameters_to_vector(self.params()))
            stopping = 0
        else:
            gradient_vectors = ((rewards - baseline)*epsilons) / denominator
            self.gradient_vector = gradient_vectors.mean(dim=0)
            stopping = stopping_criterion(gradient_vectors)

        # put gradient vector into same format as model parameters
        gradient_params = deepcopy(list(self.params()))
        vector_to_parameters(self.gradient_vector, gradient_params)

        self.gradient_norm = (sum(self.gradient_vector ** 2) ** 1/2).detach().cpu().numpy()

        ### 5. assign gradients to model gradient ####
        # We actually _add_ to existing gradient (as common in pytorch), to make it
        # possible to accumulate gradients over multiple batches.
        # When this is not desired (most of the time!), you need to flush the gradients
        # before calling this method.

        # NOTE: torch.otpimizers minimize but we use a maximization formulation
        # in the rewards, thus we need to use the negative gradient here.

        if self.experience is not None:
            if replay is None: # add observation to memories
                self.experience.append_memory(
                    self.environment,
                    priority = self.gradient_norm,
                    player_position = self.strat_to_player_kwargs['player_position']
                )
            else: # update priority of replayed memory
                self.experience.set_priority(
                    self.experience.current_memory_index,
                    priority = self.gradient_norm
                )

        for p, d_p in zip(self.params(), gradient_params):
            if p.grad is not None:
                p.grad.add_(-d_p)
            else:
                p.grad = -d_p

        return rewards, stopping

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

    def copy(self):
        return super.copy()

class DPGLearner(GradientBasedLearner):
    """Neural Self-Play with Deterministic Policy Gradients."""
    def __init__(self):
        super().__init__()
        raise NotImplementedError()

    def _set_gradients(self):
        raise NotImplementedError()
