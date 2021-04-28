"""Implements multi-agent learning rules"""

import warnings
from copy import deepcopy

from abc import ABC, abstractmethod
from typing import Tuple, Type, Callable

#for PSOLearner
import sympy.ntheory as sympy
import math
from time import perf_counter as timer

import torch
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from bnelearn.environment import Environment
from bnelearn.strategy import Strategy, NeuralNetStrategy



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
        self.writer = None
        self.cur_epoch = 0

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
        #self.optimizer.zero_grad()
        #self._set_gradients()
        #return self.optimizer.step(closure=closure)
        start_timer = timer()
        self.optimizer.zero_grad()
        self._set_gradients()
        loss = self.optimizer.step(closure=closure)
        if self.writer is not None:
            self.writer.add_scalar('learner/time_per_step', timer()-start_timer, self.cur_epoch)
        self.cur_epoch += 1
        return loss

    def update_strategy_and_evaluate_utility(self, closure = None):
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
                regularization: dict of
                    inital_strength: float, inital penaltization factor of bid value
                    regularize_decay: float, decay rate by which the regularization factor
                        is mutliplied each iteration.

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

        if 'regularization' in hyperparams:
            self.regularize = hyperparams['regularization']['inital_strength']
            self.regularize_decay = hyperparams['regularization']['regularize_decay']
        else:
            self.regularize = 0.0
            self.regularize_decay = 1.0

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
        self.regularize *= self.regularize_decay
        rewards, epsilons = (
            torch.cat(tensors).view(self.population_size, -1)
            for tensors in zip(*(
                (
                    self.environment.get_strategy_reward(
                        model, **self.strat_to_player_kwargs, regularize=self.regularize
                    ).detach().view(1),
                    epsilon
                )
                for (model, epsilon) in population
                ))
            )
        ### 4. calculate the ES-pseuogradients   ####
        # See ES_Analysis notebook in repository for more information about where
        # these choices come from.
        baseline = \
            self.environment.get_strategy_reward(
                self.model, regularize=self.regularize,
                **self.strat_to_player_kwargs
            ).detach().view(1) \
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


class PGLearner(GradientBasedLearner):
    """Neural Self-Play with directly computed Policy Gradients.

    """
    def __init__(self,
                 model: torch.nn.Module, environment: Environment, hyperparams: dict,
                 optimizer_type: Type[torch.optim.Optimizer], optimizer_hyperparams: dict,
                 strat_to_player_kwargs: dict = None):
        # Create and validate optimizer
        super().__init__(model, environment,
                         optimizer_type, optimizer_hyperparams,
                         strat_to_player_kwargs)

        if 'normalize_gradient' in hyperparams and hyperparams['normalize_gradient']:
            self.normalize_gradient = True
        else:
            self.normalize_gradients = False

        if 'baseline' in hyperparams:
            self.baseline_method = hyperparams['baseline']
            if not isinstance(self.baseline_method, float) \
                    and not self.baseline_method in ['current_reward']:
                raise ValueError('Invalid baseline provided. Should be float or '\
                    + '"current_reward"')

            if isinstance(self.baseline_method, float):
                self.baseline = self.baseline_method
                self.baseline_method = 'manual'
            else:
                 self.baseline = 0 # initial baseline

        else:
            # standard baseline
            self.baseline_method = 'current_reward'
            self.baseline = 0 # init

    def _set_gradients(self):
        self.environment.prepare_iteration()

        if self.baseline_method == 'current_reward':
            self.baseline = self.environment.get_strategy_reward(
                self.model,**self.strat_to_player_kwargs
                ).detach().view(1)
        else:
            pass # is already constant float

        loss = -self.environment.get_strategy_reward(
            self.model,**self.strat_to_player_kwargs
        )

        loss.backward()

class DPGLearner(GradientBasedLearner):
    """Implements Deterministic Policy Gradients

    http://proceedings.mlr.press/v32/silver14.pdf

    via directly calculating `dQ/da and da/d\\theta`


    """
    def __init__(self):
        raise NotImplementedError()


class _PerturbedActionModule(Strategy, torch.nn.Module):
    def __init__(self, module, epsilon):
        super().__init__()
        self.module = module
        self.epsilon = epsilon

    def forward(self, x):
        return (self.module(x) + self.epsilon).relu()

    def play(self, x):
        return self.forward(x)


class AESPGLearner(GradientBasedLearner):
    """ Implements Deterministic Policy Gradients http://proceedings.mlr.press/v32/silver14.pdf
    with ES-pseudogradients of dQ/da
    """
    def __init__(self,
                 model: NeuralNetStrategy, environment: Environment, hyperparams: dict,
                 optimizer_type: Type[torch.optim.Optimizer], optimizer_hyperparams: dict,
                 strat_to_player_kwargs: dict = None):
        # Create and validate optimizer
        super().__init__(model, environment,
                         optimizer_type, optimizer_hyperparams,
                         strat_to_player_kwargs)

        # Validate ES hyperparams
        if not set(['population_size', 'sigma']) <= set(hyperparams):
            raise ValueError(
                'Missing hyperparams for ES. Provide at least, population size, sigma.')
        if not isinstance(hyperparams['population_size'], int) or hyperparams['population_size'] < 2:
            # one is invalid because there will be zero variance, leading to div by 0 errors
            raise ValueError('Please provide a valid `population_size` parameter >=2')

        # set hyperparams
        self.population_size = hyperparams['population_size']
        self.sigma = float(hyperparams['sigma'])

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

        n_pop = self.population_size
        n_actions =  self.model.output_length
        n_batch = self.environment.batch_size

        ### 1. if required redraw valuations / perform random moves (determined by env)
        self.environment.prepare_iteration()
        ### 2. Create a population of perturbations of the original model outputs
        population = (self._perturb_model(self.model) for _ in range(n_pop))
        ### 3. let each candidate against the environment and get their utils ###

        # rewards: population_size x n_batch x 1, epsilons: n_pop x n_batch x n_action
        rewards, epsilons = (
            torch.cat(tensors)#.view(n_pop, -1)
            for tensors in zip(*(
                (
                    self.environment.get_strategy_reward(
                        model, aggregate_batch=False, **self.strat_to_player_kwargs).detach().view(1,n_batch, 1),
                    epsilon.unsqueeze(0)
                )
                for (model, epsilon) in population
                ))
            )
        ### 4. calculate the ES-pseuogradients   ####
        ## base case: current reward
        # action: batch x 1, baseline: batch
        action, baseline = self.environment.get_strategy_action_and_reward(self.model,**self.strat_to_player_kwargs)

        if self.baseline == 'mean_reward':
            baseline = rewards.mean(dim=0)
        elif isinstance(self.baseline, float):
            baseline = self.baseline

        if torch.is_tensor(baseline):
            baseline = baseline.view(n_batch, 1)

        denominator = self.sigma * rewards.std() if self.normalize_gradients else self.sigma**2

        if denominator == 0:
            # all candidates returned same reward and normalize is true --> stationary
            es_dudb = torch.zeros(n_batch, n_actions, 1)
        else:
            # mean over pop --> result is (batch), we want batch x n_actions x 1
            # this should be # batch x n_actions (TODO: test for n_actions >1)

            # pop_size x batch x 1
            scaled_rewards = (rewards - baseline)/denominator

            es_dudb = (scaled_rewards*epsilons).mean(dim=0)
            #es_dudb.unsqueeze_(-1) # batch x n_actions x 1


        ### 5. assign gradients to model gradient ####
        # should be ∇_θ π *  ∇^ES_b u
        # assuming all current `param.grad`s are zero, we set  db/da by
        #for action_loss in -torch.einsum('ba,ba->b', action, es_dudb):
        #    action_loss.div(n_batch).backward(retain_graph=True)
        loss = -torch.einsum('ba,ba->b', action, es_dudb).mean()
        loss.backward()

    def _perturb_model(self, model: NeuralNetStrategy) -> Tuple[torch.nn.Module, torch.Tensor]:
        """
        Returns a model [torch.nn.Module] perturbed via adding random noise to
        its outputs,
        as well as the noise vector used to generate the perturbation.
        """
        # for now, we'll assume model is a NeuralNetStrategy, i.e. has an attribute output_length

        noise = torch.zeros([self.environment.batch_size, model.output_length],
                             device = next(model.parameters()).device
            ).normal_(mean=0.0, std=self.sigma)

        perturbed = _PerturbedActionModule(model, noise)

        return perturbed, noise


class DDPGLearner(GradientBasedLearner):
    """Implements Deep Deterministic Policy Gradients (Lilicrap et al 2016)

       http://arxiv.org/abs/1509.02971
    """
    def __init__(self):
        raise NotImplementedError()


class DummyNonLearner(GradientBasedLearner):
    """A learner that does nothing."""

    def __init__(self,
                 model: torch.nn.Module, environment: Environment, hyperparams: dict, #pylint:disable=unused-argument
                 optimizer_type: Type[torch.optim.Optimizer], optimizer_hyperparams: dict,
                 strat_to_player_kwargs: dict = None):
        # Create and validate optimizer
        super().__init__(model, environment,
                         optimizer_type, optimizer_hyperparams,
                         strat_to_player_kwargs)

    def _set_gradients(self):
        pass


# additional libraries math, sympy.ntheory
class PSOLearner(Learner):
    """ Implements the Particle Swarm Optimization Algorithm as a Learner

        Particles represent a possible solutions to the model parameters.
        Every update step they move one step in the search space to sample a new solution point.
        They are guided by their previously best found solution (personal best position)
        and the best solution found by the entire swarm (best position)

        NOTE: dim = number of parameters in the model to be optimized

        Arguments:
            model: bnelearn.bidder
            environment: bnelearn.Environment
            hyperparams: dict
                (required:)
                    swarm_size: int
                        Number of particles in the swarm
                    topology: str
                        Defines the communication network of the swarm
                        If 'global', particles are drawn to the global best position of the swarm.
                            Neighborhood size = swarm size
                        If 'ring', particles are drawn to the best position in their neighborhood.
                            Particles form a neighborhood based on their position in the population array.
                            The first and last particles are connected to form a ring structure.
                            Neighborhood size = 3. E.g., neighborhood of particle i: particle i-1, particle i, particle i+1
                        If 'von_neumann', particles are drawn to the best position in their neighborhood.
                            Particles form a neighborhood based on their position in the population matrix.
                            A particle is connected to its left, right, upper and lower neighbor in the matrix.
                            Neighborhood size = 5
                    upper_bounds: float, List or Tensor
                        Upper search space bounds for each dimension
                        If a float is given, the value will be used for each dim
                        If bound_handling == False then only used for initialization
                    lower_bounds: float, List or Tensor
                        Lower search space bounds for each dimension
                        If a float is given, this value will be used for each dim
                        If bound_handling == False then only used for initialization
                    max_velocity: float
                        Max step size in each direction during one update step
                        If velocity_clamping == False then only used for initialization
                (optional:)
                    The default values for the inertia weight and the cognition & social ratio are commonly used values
                    performing well form most problem settings. Based on: Clerc, M., & Kennedy, J. (2002)
                    inertia_weight: float (default: 0.792)
                        Scales the impact of the old velocity on the new one.
                    cognition_ratio: float (default: 1.49445)
                        Upper limit for the impact of the personal best solution on the velocity
                    social_ratio: float (default: 1.49445)
                        Upper limit for the impact of the swarm's best solution on the velocity
                    reevaluation_frequency: int (default: None)
                        Number of epochs after which the personal and overall bests are reevaluated
                        to prevent false memory introduced by varying batch data
                    pretrain_deviation: float (default: 0)
                        If pretrain_deviation > 0 the positions will be initialized as:
                        model.parameters + U[-pretrain_deviation, pretrain_deviation]
                        otherwise positions will be initialized randomly over the whole search space
                    bound_handling: bool (default: False)
                        If true will clamp particle's positions in each dim to the interval [-max_position, max_position]
                    velocity_clamping: bool (default: True)
                        If true will clamp particle's velocities in each dim to the interval [-max_velocity, max_velocity]
                        before adding to the positions
            optimizer_type: Type[torch.optim.Optimizer]
                A class implementing torch's optimizer interface used for parameter update step.
                PSO does not need an torch optimizer to compute an parameter update step.
                -> currently only used to have an consistent interface with other learners
            optimizer_hyperparams: dict
            strat_to_player_kwargs: dict
                Dict of arguments provided to environment used for evaluating utility of current and candidate strategies.
        """

    def __init__(self,
                 model: torch.nn.Module, environment: Environment, hyperparams: dict,
                 optimizer_type: Type[torch.optim.Optimizer], optimizer_hyperparams: dict,
                 strat_to_player_kwargs: dict = None):
        self.model = model
        self.particle_evaluation_model = deepcopy(model)
        # PSO does not need gradient computation
        for param in self.particle_evaluation_model.parameters():
            param.requires_grad = False
        self.environment = environment
        self.cur_epoch = 0

        # for logging
        self.writer = None
        self.utility_eval_counter = 0

        self.strat_to_player_kwargs = strat_to_player_kwargs if strat_to_player_kwargs else {}
        # warn if weird initialization
        if 'player_position' not in self.strat_to_player_kwargs.keys():
            warnings.warn('You haven\'t specified a player_position to evaluate the model. Defaulting to position 0.')
            self.strat_to_player_kwargs['player_position'] = 0

        # validate PSO hyperparams
        if not set(['swarm_size', 'topology']) <= set(hyperparams):
            raise ValueError('Missing hyperparams for PSO. Provide at least, swarm_size, topology.')
        if not isinstance(hyperparams['swarm_size'], int) or hyperparams['swarm_size'] < 2:
            raise ValueError('Please provide a valid `swarm_size` parameter >=2')
        if hyperparams['topology'] not in ['global', 'Global', 'GLOBAL', 'von_neumann', 'von_Neumann', 'VON_NEUMANN',
                                           'ring', 'Ring', 'RING']:
            raise ValueError('Please provide a valid `topology`')

        self.topology = hyperparams['topology']

        # params needed only for initialization
        swarm_size = hyperparams['swarm_size']
        n_parameters = sum([p.numel() for p in self.model.parameters()])
        # search range
        if 'pretrain_deviation' in hyperparams:
            pretrain_deviation = float(hyperparams['pretrain_deviation'])
        else:
            pretrain_deviation = 0.0
        # model params a commonly initialized within max range [-1, 1]
        # i.e., pytorch linear: stdv = 1. / math.sqrt(input_length); self.weight.data.uniform_(-stdv,stdv)
        max_position_init = 1.0
        max_velocity = max_position_init
        max_position = max_position_init + pretrain_deviation

        # initialize non-required parameters
        if 'inertia_weight' in hyperparams:
            self.inertia = float(hyperparams['inertia_weight'])
        else:
            self.inertia = 0.729
        if 'cognition_ratio' in hyperparams:
            self.cognition = float(hyperparams['cognition_ratio'])
        else:
            self.cognition = 1.49445
        if 'social_ratio' in hyperparams:
            self.social = float(hyperparams['social_ratio'])
        else:
            self.social = 1.49445
        if 'reevaluation_frequency' in hyperparams:
            self.reevaluation_frequency = int(hyperparams['reevaluation_frequency'])
        else:
            self.reevaluation_frequency = None
        if 'bound_handling' in hyperparams and hyperparams['bound_handling']:
            self.bound_handling = True
            self.max_position = max_position
        else:
            self.bound_handling = False
        if 'velocity_clamping' in hyperparams and not hyperparams['velocity_clamping']:
            self.velocity_clamping = False
        else:
            self.velocity_clamping = True
            self.max_velocity = max_velocity

        #### --- initialize the swarm ---
        # positions
        if pretrain_deviation > 0:
            # pertubation of pretrained model params
            self.position = torch.zeros(swarm_size, n_parameters, device=torch.cuda.current_device()).normal_(mean=0.0,
                                                                                                              std=pretrain_deviation)
            self.position.add_(parameters_to_vector(self.model.parameters()))
        else:
            # random positions
            self.position = 2 * max_position * torch.rand(swarm_size, n_parameters,
                                                          device=torch.cuda.current_device()) - max_position
        # velocities
        self.velocity = 2 * max_velocity * torch.rand_like(self.position) - max_velocity
        # option for evaluation: zero velocities:
        # self.velocity = torch.zeros_like(self.position)

        # personal best fitness and positions
        self.pbest_fitness = torch.full((swarm_size,), float("Inf"), device=self.position.device)
        self.pbest_position = torch.empty_like(self.position)
        # the shape of swarm's best position and fitness depend on the topology structure
        self.best_fitness, self.best_position, self.neighborhood = self._calculate_neighborhood(swarm_size)

    def _calculate_neighborhood(self, swarm_size):
        """Initializes the swarm's best position and fitness
            and information structure (neighborhood) defining the social attractor for each particle

            Arguments:
                swarm_size: int
                    Number of particles in the swarm
            Returns:
                best_fitness: Tensor
                    The fitness value of the social attractor for each particle
                best_position: Tensor
                    The position of the social attractor for each particle
                neighborhood: Tensor
                    The indices of all particles part of the particle's neighborhood for each particle


            If a global topology is used the neighborhood size = swarm_size
            all particle remember the same best position and fitness
            only one global attractor is used as social influence
            best_position: 1 x n_params, best_fitness: 1 x 1 (single value), neighborhood: None

            If a local topology is use (ring, von Neumann) a neighborhood is defined for each particle
            each particle is attracted by the local best position and fitness of its neighborhood
            the neighborhood tensor holds the particle indices for each neighborhood
            best_position: swarm_size x n_params, best_fitness: 1 x swarm_size, neighborhood: swarm_size x neighborhood_size

        """
        if self.topology == 'global':
            # all particle use the same global position as reference -> no neighborhood indices necessary
            # the position will be set in step 0
            return torch.tensor([float("Inf")], device=self.position.device), None, None

        index = torch.unsqueeze(torch.arange(0, swarm_size, dtype=torch.long), 1)
        if self.topology == 'ring':
            # a neighborhood consists of 3 particle, the particle itself and its left and right index neighbor
            # neighborhood of particle i: particle i-1, particle i, particle i+1
            # first and last particle are connected to form a ring network

            # NOTE: torch.remainder: The remainder has the same sign as the divisor
            # neighborhood: swarm size x 3, structure: [left index, particle index, right index]
            neighborhood = index.repeat(1, 3)
            neighborhood.add_(torch.tensor([-1, 0, 1], dtype=torch.long)).remainder_(swarm_size)
        else:
            ### --- von Neumann ---
            # a neighborhood consists of 5 particle, the particle and its left, right, upper and lower index neighbor
            # particles are arranged as a matrix (n,m); size: swarm_size N = n x m ; with n,m >= 3
            # neighborhood of particle i in 1,...,N
            #       Above neighbor: N_a = (i-column) mod N; if N_a == 0, N_a = N
            #       Left neighbor: N_l = i-1; if (i-1) mod column == 0, N_l = i–1+column
            #       Right neighbor: N_r = i+1; if i mod column == 0, N_r = i+1-column
            #       Below neighbor: N_b = (i+column) mod N; if N_b == 0, N_b = N

            # NOTE: torch.remainder: The remainder has the same sign as the divisor.
            # neighborhood: swarm size x 5
            # structure: [uppper index, left index, particle index, right index, lower index]

            ### 1. calculate the size of the matrix (column length)
            if sympy.isprime(swarm_size) or swarm_size < 9:
                raise ValueError("{} is not a valid value for von neumann neighborhood size".format(swarm_size))
            if math.ceil(math.sqrt(swarm_size)) ** 2 == swarm_size:
                column = math.ceil(math.sqrt(swarm_size))
            else:
                prime = torch.Tensor(list(sympy.primerange(3, math.ceil(math.sqrt(swarm_size)))))
                column_candidates = torch.cat((torch.Tensor([4]), prime))
                swarm_dividers = torch.remainder(torch.full((column_candidates.numel(),), swarm_size, dtype=torch.long),
                                                 column_candidates) == 0
                if column_candidates[swarm_dividers].numel() == 0:
                    raise ValueError("{} is not a valid value for von neumann neighborhood size".format(swarm_size))
                column = column_candidates[swarm_dividers].max().long()

            ### 2. initialize the neighborhood index tensor
            neighborhood = index.repeat(1, 5)
            neighborhood.add_(torch.Tensor([-column, -1, 0, 1, column]).long())
            neighborhood[::column, 1].add_(column)
            neighborhood[(column - 1)::column, 3].add_(-column)
            neighborhood.remainder_(swarm_size)

        # best_position: swarm_size x n_params, best_fitness: 1 x swarm size
        return self.pbest_fitness.detach().clone(), torch.empty_like(self.position), neighborhood.to(
            device=self.position.device)

    def _calculate_fitness(self, position):
        """Let the candidate particle try against the environment and get its utility

            NOTE: PSO minimize but we use a maximization formulation in the rewards,
            thus we need to use the negative reward.

            Arguments:
                position: Tensor
                    The current particle's parameter values
            Returns:
                reward: Tensor
                    The fitness value (utility) of the current particle
        """
        vector_to_parameters(position, self.particle_evaluation_model.parameters())
        reward = self.environment.get_strategy_reward(self.particle_evaluation_model,
                                                      **self.strat_to_player_kwargs).detach()
        assert reward.numel() == 1
        self.utility_eval_counter += 1
        return -reward

    def update_strategy(self):
        # Performs one model-update to the player's strategy.
        start_time = timer()

        ### 1. if required redraw valuations / perform random moves (determined by env)
        self.environment.prepare_iteration()
        ### 2. evaluate each particles current position (solution)
        # fitness: 1 x swarm size
        fitness = torch.tensor([self._calculate_fitness(p) for p in self.position], device=self.position.device)

        # prevent stale memory: reevaluate the personal and overall best fitness
        if self.reevaluation_frequency and self.cur_epoch > 0 and not self.cur_epoch % self.reevaluation_frequency:
            old_best = self.best_fitness.detach().clone()
            self.best_fitness = torch.squeeze(
                torch.tensor([self._calculate_fitness(p) for p in self.best_position], device=self.position.device), 0)
            if not torch.equal(old_best, self.best_fitness):
                self.pbest_fitness = torch.tensor([self._calculate_fitness(p) for p in self.pbest_position],
                                                  device=self.position.device)

        ### --- best solution update ---
        ### 3. update the personal best positions:
        # check if the current sample point of the particle is a better solution than the particles previous found solution
        # -> check if the particle's current fitness is better than the particle's "personal best fitness".
        # if so, update the personal best position and fitness to the values of the current ones
        new_best = fitness < self.pbest_fitness
        if new_best.any():
            self.pbest_fitness[new_best] = fitness[new_best]
            self.pbest_position[new_best, :] = self.position[new_best, :]

        # 4. update the swarm's best position(s):
        if self.topology == 'global':
            # check if a particle found a better solution than the current global best
            #  -> check if the best fitness of all "personal best fitness" is better than the "global best fitness".
            # if so, update the global best position and fitness to the values of the best personal best
            if self.pbest_fitness.min() < self.best_fitness:
                self.best_fitness = self.pbest_fitness.min()
                self.best_position = torch.unsqueeze(self.pbest_position[self.pbest_fitness.argmin(), :], 0)

        else:
            # get the best particle of each neighborhood (best "personal best fitness" of the neighborhood)
            # check if this particle's "personal best fitness" is better than the previous "best fitness" of the neighborhood.
            # if so, update the neighborhoods best positions and fitness to the values of this particle's personal best
            best_neighbor = self.pbest_fitness[self.neighborhood].min(axis=1)
            new_best = best_neighbor.values < self.best_fitness
            if new_best.any():
                self.best_fitness[new_best] = best_neighbor.values[new_best]
                index = self.neighborhood[torch.arange(0, self.neighborhood.size()[0]), best_neighbor.indices][new_best]
                self.best_position[new_best, :] = self.pbest_position[index, :]

        ### --- move ---
        ### 5. update the velocities:
        #save current velocity before updating
        cur_velocity = self.velocity
        cur_position = self.position
        # new velocity = old velocity + cognitive component + social component
        self.velocity = self.inertia * self.velocity \
                        + self.cognition * torch.rand_like(self.position) * (self.pbest_position - self.position) \
                        + self.social * torch.rand_like(self.position) * (self.best_position - self.position)
        # clamp particles velocity values to be <= the maximal allowed velocity step size
        if self.velocity_clamping:
            self.velocity.clamp_(-self.max_velocity, self.max_velocity)
        ### 6. update the positions
        self.position += self.velocity
        # clamp particles position values to lay inside the search space bounds
        if self.bound_handling:
            self.position.clamp_(-self.max_position, self.max_position)
            # torch.max(torch.min(self.position, self.max_position), -self.max_position, out=self.position)

        assert torch.isfinite(self.best_fitness.min())
        # assign the parameters of the best particle to the model parameters
        vector_to_parameters(self.best_position[self.best_fitness.argmin(), :], self.model.parameters())
        #an sich unnötig aber so übernommen von Nils
        #wenn model sharing = on ist es eh immer 0
        #prob wenn model sharing = off dann würde sosnt für zweites model einfach angehangen ohne unterscheidung
        #kann mir für eval egal sein brauch die werte nicht für model sharing
        time_per_step = timer()-start_time
        if self.strat_to_player_kwargs == {'player_position': 0} and self.writer is not None:
            self._log_pso_params(cur_velocity, cur_position, time_per_step)
        self.cur_epoch += 1

    def update_strategy_and_evaluate_utility(self):
        self.update_strategy()
        true_best_fitness = self.environment.get_strategy_reward(self.model, **self.strat_to_player_kwargs).detach()
        if self.writer is not None:
            self.writer.add_scalar('learner/fitness_error', torch.abs(torch.neg(self.best_fitness.min())-true_best_fitness), self.cur_epoch)
        return true_best_fitness
        #return self.environment.get_strategy_reward(self.model, **self.strat_to_player_kwargs).detach()

    def _log_pso_params(self, velocity, position, time_per_step):
        position_L_2_norm = torch.linalg.norm(position - self.best_position)*(1./float(position.shape[0]))**(1/2)
        velocity_L_2_norm = torch.linalg.norm(velocity)*(1./float(velocity.shape[0]))**(1/2)

        self.writer.add_scalar('learner/util_eval_counter', self.utility_eval_counter, self.cur_epoch)
        self.writer.add_scalar('learner/velocity_L_2', velocity_L_2_norm, self.cur_epoch)
        self.writer.add_scalar('learner/position_L_2', position_L_2_norm, self.cur_epoch)
        self.writer.add_scalar('learner/best_fitness', torch.neg(self.best_fitness.min()), self.cur_epoch)
        self.writer.add_scalar('learner/time_per_step', time_per_step, self.cur_epoch)

