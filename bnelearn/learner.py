"""Implements multi-agent learning rules"""

import warnings
from copy import deepcopy

from abc import ABC, abstractmethod
from typing import Tuple, Type, Callable

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

    def update_strategy(self, closure: Callable=None,
                        opponent_model: torch.nn.Module=None) -> None or torch.Tensor: # pylint: disable=arguments-differ
        """Performs one model-update to the player's strategy.

        Params:
            closure: (optional) Callable that recomputes model loss.
                Required by some optimizers such as LBFGS. When given,
                optimizer.step() (and thus this function) return the last
                evaluated loss. (Usually evaluated BEFORE the model update).
                For correct usage see:
                https://pytorch.org/docs/stable/optim.html#optimizer-step-closure
            opponent_model: (optional) for opponent aware learning.

        Returns: None or loss evaluated by closure. (See above.)
        """

        self.optimizer.zero_grad()

        if opponent_model is None:
            self._set_gradients()
        else:
            optimizer_type = type(self.optimizer)
            opponent_model.optimizer: torch.optim.Optimizer = optimizer_type(
                opponent_model.parameters(), **self.optimizer_hyperparams)
            opponent_model.optimizer.zero_grad()
            self._set_gradients(opponent_model=opponent_model)

            # Undo comment to enable simultaneous update of parameters
            # opponent_model.optimizer.step(closure=closure)

        return self.optimizer.step(closure=closure)

    def update_strategy_and_evaluate_utility(self, closure: Callable=None,
                                             opponent_model: torch.nn.Module=None):
        """updates model and returns utility after the update."""

        self.update_strategy(closure=closure, opponent_model=opponent_model)
        reward = self.environment.get_strategy_reward(
            strategy=self.model, opponent_model=opponent_model,
            **self.strat_to_player_kwargs
        )

        if opponent_model is None:
            return reward.detach()
        else:  # Opponent utility is returned also
            return reward[0].detach()


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
        super().__init__(model=model, environment=environment,
                         optimizer_type=optimizer_type,
                         optimizer_hyperparams=optimizer_hyperparams,
                         strat_to_player_kwargs=strat_to_player_kwargs)

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
                        strategy=model,
                        **self.strat_to_player_kwargs
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
                strategy=self.model,
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


class LOLALearner(GradientBasedLearner):
    """LOLA Learner implemented with Automatic Differentiation Gradient (ADG).
    
       LOLA Algorithm Reference: https://arxiv.org/abs/1709.04326

       LOLA Gradient for player 1 is calculated as: L1/Q1 - L1/Q2 * L2/Q2Q1 * eta

    Author: Liu Wusheng.
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

        # Validate and set LOLA hyperparams
        if not set(['eta']) <= set(hyperparams):
            print('Fallback to default second order LOLA step size')
            self.eta = 3e-3
        else:
            self.eta = hyperparams['eta']

    def _set_gradients(self, opponent_model):
        self.environment.prepare_iteration()

        loss, loss_opponent = self.environment.get_strategy_reward(
            strategy=self.model, opponent_model=opponent_model,
            **self.strat_to_player_kwargs
        )

        # Reverse the sign to get losses from the returned utilities
        loss = -loss
        loss_opponent = -loss_opponent

        # Calculate gradient of self, L1/Q1
        gradient_from_own_loss = torch.autograd.grad(loss, self.model.parameters(),create_graph=True)
        own_flat = [torch.flatten(p) for p in gradient_from_own_loss]
        own_flat_grad = torch.cat(own_flat).view(-1, 1)

        # Calculate gradient of opponent,L2/Q2
        gradient_from_opp_loss = torch.autograd.grad(loss_opponent, opponent_model.parameters(),create_graph=True)
        opp_flat = [torch.flatten(p) for p in gradient_from_opp_loss]
        opp_flat_grad = torch.cat(opp_flat).view(-1, 1)

        # Calculate self gradient over opponent, L1/Q2
        gradient_from_cross_loss = torch.autograd.grad(loss, opponent_model.parameters(),create_graph=True)
        cross_flat = [torch.flatten(p) for p in gradient_from_cross_loss]
        cross_flat_grad = torch.cat(cross_flat).view(-1, 1)

        # Calculate Hessian matrix, L2/Q2Q1
        hess_params = []
        for i in range(opp_flat_grad.size(0)):
            sec_grad=torch.autograd.grad(opp_flat_grad[i], self.model.parameters(), create_graph=True)
            sec_flat = [torch.flatten(p) for p in sec_grad]
            sec_grad_flat = torch.cat(sec_flat).view(-1, 1)
            sec_grad_flat = torch.transpose(sec_grad_flat, 0, 1)
            hess_params.append(sec_grad_flat)
        opp_hessian = torch.squeeze(torch.stack(hess_params)).view(-1, cross_flat_grad.size(0))

        # Calculate LOLA gradient
        grads = own_flat_grad - self.eta*torch.matmul(opp_hessian, cross_flat_grad)

        # put gradient vector into same format as model parameters
        gradient_params = deepcopy(list(self.params()))
        vector_to_parameters(grads, gradient_params)

        ### 5. assign gradients to model gradient ####
        # We actually _add_ to existing gradient (as common in pytorch), to make it
        # possible to accumulate gradients over multiple batches.
        # When this is not desired (most of the time!), you need to flush the gradients
        # before calling this method.
        # NOTE: torch.otpimizers minimize but we use a maximization formulation
        # in the rewards, thus we need to use the negative gradient here.
        # ---Need to check the sign of the LOLA gradient

        for p, d_p in zip(self.params(), gradient_params):
            if p.grad is not None:
                p.grad.add_(d_p)
            else:
                p.grad = d_p


class LOLA_ESPGLearner(ESPGLearner):
    """LOLA implemented with mixed ADG and ESPG. First order gradient R1/Q1 is calculated with ESPG.
       For the second derivative term R2/Q2Q1, first calculate R2/Q2 with ADG, then take derivative over Q1 with ESPG.
       
       LOLA Algorithm Reference: https://arxiv.org/abs/1709.04326

       LOLA Gradient for player 1 is calculated as: -(R1/Q1 + R1/Q2 * R2/Q2Q1 * eta)
    
       ESPG is calculated as: mean( rewards - baseline) * epsilons / sigma²
            and approximates the true gradient.

    Author: Liu Wusheng.
    """
    def __init__(self,
                 model: torch.nn.Module, environment: Environment, hyperparams: dict,
                 optimizer_type: Type[torch.optim.Optimizer], optimizer_hyperparams: dict,
                 strat_to_player_kwargs: dict = None):

        # Create and validate optimizer
        super().__init__(model=model, environment=environment,
                         hyperparams=hyperparams, optimizer_type=optimizer_type,
                         optimizer_hyperparams=optimizer_hyperparams,
                         strat_to_player_kwargs=strat_to_player_kwargs)

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
                 self.baseline = 0  # initial baseline

        else:
            # standard baseline
            self.baseline_method = 'current_reward'
            self.baseline = 0  # init

        # overwrite baseline method if provided
        if 'baseline' in hyperparams:
            self.baseline_method = hyperparams['baseline']
            if not isinstance(self.baseline_method, float) \
                    and not self.baseline_method in ['current_reward', 'mean_reward']:
                raise ValueError('Invalid baseline provided. Should be float or '\
                    + 'one of "mean_reward", "current_reward"')

        # Validate and set LOLA hyperparams
        if not set(['eta']) <= set(hyperparams):
            print('Fallback to default second order LOLA step size')
            self.eta = 3e-3
        else:
            self.eta = hyperparams['eta']

    def _set_gradients(self, opponent_model):
        self.environment.prepare_iteration()

        if self.baseline_method == 'current_reward':
            self.baseline = self.environment.get_strategy_reward(
                strategy=self.model, opponent_model=opponent_model,
                **self.strat_to_player_kwargs
            )[0].detach().view(1)
        else:
            pass  # is already constant float

        population = (self._perturb_model(self.model) for _ in range(self.population_size))

        ### 3. let each candidate against the environment and get their utils ###
        # both of these as a row-matrix. i.e.
        # rewards: population_size x 1
        # epsilons: population_size x parameter_length

        # Calculate gradient of self, R1/Q1
        rewards, epsilons = (
            torch.cat(tensors).view(self.population_size, -1)
            for tensors in zip(*(
                (
                    self.environment.get_strategy_reward(
                        strategy=model, opponent_model=opponent_model,
                        **self.strat_to_player_kwargs
                    )[0].detach().view(1),
                    epsilon
                )
                for (model, epsilon) in population
                ))
            )
        baseline = \
            self.environment.get_strategy_reward(
                strategy=self.model, opponent_model=opponent_model,
                **self.strat_to_player_kwargs)[0].detach().view(1) \
                if self.baseline == 'current_reward' \
            else rewards.mean(dim=0) if self.baseline == 'mean_reward' \
            else self.baseline # a float

        denominator = self.sigma * rewards.std() if self.normalize_gradients else self.sigma**2    

        if denominator == 0:
            # all candidates returned same reward and normalize is true --> stationary
            gradient_vector = torch.zeros_like(parameters_to_vector(self.params()))
        else:
            gradient_vector = ((rewards - baseline)*epsilons).mean(dim=0) / denominator

        # Calculate self gradient over opponent, R1/Q2
        opp_population = (self._perturb_model(opponent_model) for _ in range(self.population_size))

        cross_rewards, cross_epsilons = (
            torch.cat(tensors).view(self.population_size, -1)
            for tensors in zip(*(
                (
                    self.environment.get_strategy_reward(
                        strategy=self.model, opponent_model=cross_model,
                        **self.strat_to_player_kwargs
                    )[0].detach().view(1),
                    cross_epsilon
                )
                for (cross_model, cross_epsilon) in opp_population
                ))
            )

        cross_denominator = self.sigma * cross_rewards.std() if self.normalize_gradients else self.sigma**2

        if cross_denominator == 0:
            # all candidates returned same reward and normalize is true --> stationary
            cross_gradient_vector = torch.zeros_like(parameters_to_vector(self.params()))
        else:
            cross_gradient_vector = ((cross_rewards - baseline)*cross_epsilons).mean(dim=0) / cross_denominator

        # Calculate second order derivative, R2/Q2Q1
        sec_gradient_vector = self._sec_gradient(self.model, opponent_model)

        LOLA_gradient = torch.einsum('b,bc->c', cross_gradient_vector, sec_gradient_vector)
        gradient_vector = gradient_vector + self.eta*LOLA_gradient

        # put gradient vector into same format as model parameters
        gradient_params = deepcopy(list(self.params()))
        # gradient_params = deepcopy(list(self.model.parameters()))
        vector_to_parameters(gradient_vector, gradient_params)

        ### 5. assign gradients to model gradient ####
        # We actually _add_ to existing gradient (as common in pytorch), to make it
        # possible to accumulate gradients over multiple batches.
        # When this is not desired (most of the time!), you need to flush the gradients
        # before calling this method.

        # NOTE: torch.otpimizers minimize but we use a maximization formulation
        # in the rewards, thus we need to use the negative gradient here.

        for p, d_p in zip(self.params(), gradient_params):
        # for p, d_p in zip(self.model.parameters(), gradient_params):
            if p.grad is not None:
                p.grad.add_(-d_p)
            else:
                p.grad = -d_p

    def _sec_gradient(self, model, opponent_model) -> torch.Tensor:
        # Calculate second order derivative, R2/Q1Q2
        population = (self._perturb_model(model) for _ in range(self.population_size))
        sec_rewards, sec_epsilons = (
            torch.cat(tensors).view(self.population_size, -1)
            for tensors in zip(*(
                (
                    self.opp_gradient(cross_model, opponent_model).detach(),
                    cross_epsilon
                )
                for (cross_model, cross_epsilon) in population
                ))
            )

        sec_baseline = self.opp_gradient(model, opponent_model).detach() \
                if self.baseline == 'current_reward' \
            else sec_rewards.mean(dim=0) if self.baseline == 'mean_reward' \
            else self.baseline # a float

        sec_denominator = self.sigma * sec_rewards.std() if self.normalize_gradients else self.sigma**2

        if sec_denominator == 0:
            # all candidates returned same reward and normalize is true --> stationary
            sec_gradient_vector = torch.zeros_like(parameters_to_vector(opponent_model.parameters()))
        else:
            sec_gradient_vector = torch.einsum(
                'ba,bc->bac', (sec_rewards - sec_baseline), sec_epsilons
            ).mean(dim=0) / sec_denominator

        return sec_gradient_vector

    def opp_gradient(self, model, opponent_model)-> torch.Tensor: 
        # Calculate gradient of opponent with ADG,R2/Q2
        loss_opponent = self.environment.get_strategy_reward(
            strategy=model, opponent_model=opponent_model,
            **self.strat_to_player_kwargs
        )[1]
        opp_gradient=torch.autograd.grad(loss_opponent, opponent_model.parameters(),create_graph=True)
        opp_flat = [torch.flatten(p) for p in opp_gradient]
        opp_gradient_vector = torch.squeeze(torch.cat(opp_flat).view(-1, 1), 1)
        return opp_gradient_vector


class SOSLearner(GradientBasedLearner):
    """"SOS algorithm implemented with ADG.
     
        SOS Algorithm Reference: https://arxiv.org/abs/1811.08469

        SOS Gradient for player 1 is calculated as: L1/Q1 - eta * L2/Q1 * L1/Q1Q2 - p*(eta * L1/Q2 * L2/Q2Q1)

    Author: Liu Wusheng.
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

        # Validate and set SOS hyperparams
        if not set(['eta']) <= set(hyperparams):
            print('Fallback to default second order SOS step size')
            self.eta = 3e-3
        else:
            self.eta = hyperparams['eta']

        self.epoch = 0
        self.writer = None

    def _set_gradients(self, opponent_model):

        self.environment.prepare_iteration()

        loss, loss_opponent = self.environment.get_strategy_reward(
            strategy=self.model, opponent_model=opponent_model,
            **self.strat_to_player_kwargs
        )

        # Reverse the sign to get losses from the returned utilities
        loss = -loss
        loss_opponent = -loss_opponent

        # Calculate gradient of self, L1/Q1
        gradient_from_own_loss = torch.autograd.grad(
            loss, self.model.parameters(), create_graph=True)
        own_flat = [torch.flatten(p) for p in gradient_from_own_loss]
        own_flat_grad = torch.cat(own_flat).view(-1, 1)

        # Calculate gradient of opponent, L2/Q2
        gradient_from_opp_loss = torch.autograd.grad(
            loss_opponent, opponent_model.parameters(), create_graph=True)
        opp_flat = [torch.flatten(p) for p in gradient_from_opp_loss]
        opp_flat_grad = torch.cat(opp_flat).view(-1, 1)

        # Calculate self gradient over opponent, L1/Q2
        gradient_from_cross_loss = torch.autograd.grad(
            loss, opponent_model.parameters(), create_graph=True)
        cross_flat = [torch.flatten(p) for p in gradient_from_cross_loss]
        cross_flat_grad = torch.cat(cross_flat).view(-1, 1)

        # Calculate Hessian matrix, L2/Q2Q1
        hess_params = []
        for i in range(opp_flat_grad.size(0)):
            sec_grad = torch.autograd.grad(
                opp_flat_grad[i], self.model.parameters(), create_graph=True)
            sec_flat = [torch.flatten(p) for p in sec_grad]
            sec_grad_flat = torch.cat(sec_flat).view(-1, 1)
            sec_grad_flat = torch.transpose(sec_grad_flat, 0, 1)
            hess_params.append(sec_grad_flat)
        opp_hessian = torch.squeeze(torch.stack(hess_params)).view(-1, cross_flat_grad.size(0))

        # Calculate Hessian matrix, L1/Q1Q2
        sos_hess_params = []
        for i in range(own_flat_grad.size(0)):
            sos_grad = torch.autograd.grad(
                own_flat_grad[i], opponent_model.parameters(), create_graph=True)
            sos_flat = [torch.flatten(p) for p in sos_grad]
            sos_grad_flat = torch.cat(sos_flat).view(-1, 1)
            sos_grad_flat = torch.transpose(sos_grad_flat, 0, 1)
            sos_hess_params.append(sos_grad_flat)
        sos_hessian = torch.squeeze(torch.stack(sos_hess_params)).view(-1, opp_flat_grad.size(0))

        # Calculate LOLA second order correction and SOS LookAhead second order correction term
        LOLA_gradient = torch.matmul(opp_hessian, cross_flat_grad)
        SOS_gradient = torch.matmul(sos_hessian, opp_flat_grad)

        # Assign naive gradient
        gradient_vector = own_flat_grad

        # SOS Algorithm gradient
        xi_0 = gradient_vector - self.eta*SOS_gradient
        chi = LOLA_gradient
        a = 0.5
        b = 0.1
        dot = torch.einsum('ij,ij->j', -self.eta*chi, xi_0)
        p1 = 1 if dot >= 0 else min(1, -a*torch.norm(xi_0)**2/dot)
        xi_norm = torch.norm(gradient_vector)
        p2 = xi_norm**2 if xi_norm < b else 1
        p = min(p1, p2)
        gradient_vector = xi_0 - p*self.eta*chi

        # Logging P-value
        if self.strat_to_player_kwargs == {'player_position': 0}:
            self.epoch += 1
            self.writer.add_scalar('learner/dot', dot, self.epoch)
            self.writer.add_scalar('learner/xi_norm', xi_norm, self.epoch)
            self.writer.add_scalar('learner/p1_value', p1, self.epoch)
            self.writer.add_scalar('learner/p2_value', p2, self.epoch)
            self.writer.add_scalar('learner/p_value', p, self.epoch)

        # put gradient vector into same format as model parameters
        gradient_params = deepcopy(list(self.params()))
        vector_to_parameters(gradient_vector, gradient_params)

        ### 5. assign gradients to model gradient ####
        # We actually _add_ to existing gradient (as common in pytorch), to make it
        # possible to accumulate gradients over multiple batches.
        # When this is not desired (most of the time!), you need to flush the gradients
        # before calling this method.

        for p, d_p in zip(self.params(), gradient_params):
            if p.grad is not None:
                p.grad.add_(d_p)
            else:
                p.grad = d_p


class SOS_ESPGLearner_Mixed(ESPGLearner):
    """SOS implementation with Mixed ADG and ESPG.
    
       SOS Algorithm Reference: https://arxiv.org/abs/1811.08469

       SOS Gradient for player 1 is calculated as: -（R1/Q1 + eta * R2/Q1 * R1/Q1Q2 + p*(eta * R1/Q2 * R2/Q2Q1)）
     
       ESPG is calculated as: mean( rewards - baseline) * epsilons / sigma²
            and approximates the true gradient.

    Author: Liu Wusheng.
    """
    def __init__(self,
                 model: torch.nn.Module, environment: Environment, hyperparams: dict,
                 optimizer_type: Type[torch.optim.Optimizer], optimizer_hyperparams: dict,
                 strat_to_player_kwargs: dict = None):
        # Create and validate optimizer
        super().__init__(model=model, environment=environment,
                         hyperparams=hyperparams, optimizer_type=optimizer_type,
                         optimizer_hyperparams=optimizer_hyperparams,
                         strat_to_player_kwargs=strat_to_player_kwargs)

        # Note: There are two SOS versions 1 (default) and 2 (simultaneous updating, faster?)
        self.simultaneous = False

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

        # Validate and set SOS hyperparams
        if not set(['eta']) <= set(hyperparams):
            print('Fallback to default second order SOS step size')
            self.eta = 3e-3
        else:
            self.eta = hyperparams['eta']

        self.epoch = 0
        self.writer = None

    def _set_gradients(self, opponent_model):

        ### 1. if required redraw valuations / perform random moves (determined by env)
        self.environment.prepare_iteration()

        ### 2. Create a population of perturbations of the original model
        population = (self._perturb_model(self.model) for _ in range(self.population_size))
        population2 = (self._perturb_model(self.model) for _ in range(self.population_size))
        population3 = (self._perturb_model(opponent_model) for _ in range(self.population_size))

        ### 3. let each candidate against the environment and get their utils ###
        # both of these as a row-matrix. i.e.
        # rewards: population_size x 1
        # epsilons: population_size x parameter_length

        # Calculate gradient of self, R1/Q1
        rewards, epsilons = (
            torch.cat(tensors).view(self.population_size, -1)
            for tensors in zip(*(
                (
                    self.environment.get_strategy_reward(
                        strategy=model, opponent_model=opponent_model,
                        **self.strat_to_player_kwargs)[0].detach().view(1),
                    epsilon
                )
                for (model, epsilon) in population
                ))
            )

        # Calculate self gradient over opponent, R1/Q2
        cross_gradient_vector = self.cross_gradient_vector(
            strategy=self.model, opponent_model=opponent_model)

        # Calculate second order derivative, R2/Q2Q1
        sec_rewards, sec_epsilons = (
            torch.cat(tensors).view(self.population_size, -1)
            for tensors in zip(*(
                (
                    self.opp_gradient(
                        strategy=model, opponent_model=opponent_model).detach(),
                    epsilon
                )
                for (model, epsilon) in population2
                ))
            )

        # Calculate gradient of opponent,R2/Q2
        opp_gradient_vector = self.opp_gradient_vector(
            strategy=self.model, opponent_model=opponent_model
        ).detach()

        # Calculate second order derivative, R1/Q1Q2
        sos_rewards, sos_epsilons = (
            torch.cat(tensors).view(self.population_size, -1)
            for tensors in zip(*(
                (
                    self.self_gradient(
                        model, opponent_model).detach(),
                    epsilon
                )
                for (model, epsilon) in population3
                ))
            )

        ### 4. calculate the ES-pseuogradients   ####
        baseline = \
            self.environment.get_strategy_reward(
                strategy=self.model, opponent_model=opponent_model,
                **self.strat_to_player_kwargs
            )[0].detach().view(1) if self.baseline == 'current_reward' \
            else rewards.mean(dim=0) if self.baseline == 'mean_reward' \
            else self.baseline # a float

        sec_baseline = \
            self.opp_gradient(strategy=self.model, opponent_model=opponent_model).detach() \
                if self.baseline == 'current_reward' \
            else sec_rewards.mean(dim=0)

        sos_baseline = \
            self.self_gradient(strategy=self.model, opponent_model=opponent_model).detach() \
                if self.baseline == 'current_reward' \
            else sos_rewards.mean(dim=0)

        denominator = self.sigma * rewards.std() if self.normalize_gradients else self.sigma**2
        sec_denominator = self.sigma * sec_rewards.std() if self.normalize_gradients else self.sigma**2
        sos_denominator = self.sigma * sos_rewards.std() if self.normalize_gradients else self.sigma**2

        if denominator == 0:
            # all candidates returned same reward and normalize is true --> stationary
            gradient_vector = torch.zeros_like(parameters_to_vector(self.params()))
        else:
            gradient_vector = ((rewards - baseline)*epsilons).mean(dim=0) / denominator

        if sec_denominator == 0:
            # all candidates returned same reward and normalize is true --> stationary
            sec_gradient_vector = torch.zeros_like(
                parameters_to_vector(self.params()),
                parameters_to_vector(self.params()))
        else:
            sec_gradient_vector = torch.einsum(
                'ba,bc->bac', (sec_rewards - sec_baseline), sec_epsilons
            ).mean(dim=0) / sec_denominator

        if sos_denominator == 0:
            # all candidates returned same reward and normalize is true --> stationary
            sos_gradient_vector = torch.zeros_like(
                parameters_to_vector(self.params()),
                parameters_to_vector(self.params()))
        else:
            sos_gradient_vector = torch.einsum(
                'ba,bc->bac', (sos_rewards - sos_baseline), sos_epsilons
            ).mean(dim=0) / sos_denominator

        # Calculate LOLA second order correction and SOS LookAhead second order correction term for self
        LOLA_gradient = torch.einsum('b,bc->c', cross_gradient_vector, sec_gradient_vector)
        SOS_gradient = torch.einsum('b,bc->c', opp_gradient_vector, sos_gradient_vector)

        a = 0.5
        b = 0.1
        if self.simultaneous:
            # Calculate opponent gradient over self, L2/Q1
            opp_cross_gradient_vector = self.opp_cross_gradient_vector(self.model, opponent_model)

            # Calculate LOLA second order correction and SOS LookAhead second order correction term for opponent
            opp_LOLA_gradient = torch.einsum('b,bc->c', opp_cross_gradient_vector, sos_gradient_vector)
            opp_SOS_gradient = torch.einsum('b,bc->c', gradient_vector, sec_gradient_vector)

            # SOS Algorithm gradient
            xi_0 = torch.cat(
                [
                    gradient_vector + self.eta*SOS_gradient,
                    opp_gradient_vector + self.eta*opp_SOS_gradient
                ],
                0
            )
            chi = torch.cat([LOLA_gradient, opp_LOLA_gradient], 0)

            dot = torch.dot(self.eta*chi, xi_0)
            p1 = 1 if dot >= 0 else min(1, -a*torch.norm(xi_0)**2/dot)
            xi_norm = torch.norm(torch.cat([gradient_vector, opp_gradient_vector], 0))
            p2 = xi_norm**2 if xi_norm < b else 1
            p = min(p1, p2)

            gradient_vector, opp_gradient_vector = (xi_0 + p*self.eta*chi).view(2, -1)

        else:
            # SOS Algorithm gradient
            xi_0 = gradient_vector + self.eta*SOS_gradient
            chi = LOLA_gradient
            dot = torch.dot(self.eta*chi, xi_0)
            p1 = 1 if dot >= 0 else min(1, -a*torch.norm(xi_0)**2/dot)
            xi_norm = torch.norm(gradient_vector)
            p2 = xi_norm**2 if xi_norm < b else 1
            p = min(p1, p2)
            gradient_vector = xi_0+p*self.eta*chi

        # Logging P-value
        if self.strat_to_player_kwargs == {'player_position': 0}:
            self.epoch += 1
            self.writer.add_scalar('learner/dot', dot, self.epoch)
            self.writer.add_scalar('learner/xi_norm', xi_norm, self.epoch)
            self.writer.add_scalar('learner/p1_value', p1, self.epoch)
            self.writer.add_scalar('learner/p2_value', p2, self.epoch)
            self.writer.add_scalar('learner/p_value', p, self.epoch)

        # put gradient vector into same format as model parameters
        gradient_params = deepcopy(list(self.params()))
        vector_to_parameters(gradient_vector, gradient_params)
        if self.simultaneous:
            opp_gradient_params = deepcopy(list(opponent_model.parameters()))
            vector_to_parameters(opp_gradient_vector, opp_gradient_params)

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

        if self.simultaneous:
            for p, d_p in zip(opponent_model.parameters(), opp_gradient_params):
                if p.grad is not None:
                    p.grad.add_(-d_p)
                else:
                    p.grad = -d_p

    def self_gradient(self, model, opponent_model)-> torch.Tensor: 
        # Calculate gradient of opponent with ADG,R1/Q1
        loss = self.environment.get_strategy_reward(
            strategy=model, opponent_model=opponent_model,
            **self.strat_to_player_kwargs
        )[0]
        gradient=torch.autograd.grad(loss, model.parameters(),create_graph=True)
        self_flat = [torch.flatten(p) for p in gradient]
        gradient_vector = torch.squeeze(torch.cat(self_flat).view(-1, 1), 1)
        return gradient_vector

    def opp_gradient(self, model, opponent_model)-> torch.Tensor: 
        # Calculate gradient of opponent with ADG,R2/Q2
        loss_opponent = self.environment.get_strategy_reward(
            strategy=model, opponent_model=opponent_model,
            **self.strat_to_player_kwargs
        )[1]
        opp_gradient=torch.autograd.grad(loss_opponent, opponent_model.parameters(),create_graph=True)
        opp_flat = [torch.flatten(p) for p in opp_gradient]
        opp_gradient_vector = torch.squeeze(torch.cat(opp_flat).view(-1, 1), 1)
        return opp_gradient_vector

    def opp_gradient_vector(self, strategy, opponent_model) -> torch.Tensor:
        """Calculate gradient of opponent, R2/Q2"""
        opp_population = (self._perturb_model(opponent_model) for _ in range(self.population_size))
        opp_rewards, opp_epsilons = (
            torch.cat(tensors).view(self.population_size, -1)
            for tensors in zip(*(
                (
                    self.environment.get_strategy_reward(
                        strategy=strategy, opponent_model=opp_model,
                        **self.strat_to_player_kwargs)[1].detach().view(1),
                    opp_epsilon
                )
                for (opp_model, opp_epsilon) in opp_population
                ))
            )

        opp_baseline = \
            self.environment.get_strategy_reward(
                strategy=strategy, opponent_model=opponent_model,
                **self.strat_to_player_kwargs
            )[1].detach().view(1) if self.baseline == 'current_reward' \
            else opp_rewards.mean(dim=0) if self.baseline == 'mean_reward' \
            else self.baseline # a float

        opp_denominator = self.sigma * opp_rewards.std() if self.normalize_gradients else self.sigma**2

        if opp_denominator == 0:
            # all candidates returned same reward and normalize is true --> stationary
            opp_gradient_vector = torch.zeros_like(parameters_to_vector(self.params()))
        else:
            opp_gradient_vector = ((opp_rewards - opp_baseline)*opp_epsilons).mean(dim=0) / opp_denominator
        return opp_gradient_vector

    def cross_gradient_vector(self, strategy, opponent_model) -> torch.Tensor:
        """Calculate self gradient over opponent, R1/Q2 """
        opp_population = (self._perturb_model(opponent_model) for _ in range(self.population_size))

        cross_rewards, cross_epsilons = (
            torch.cat(tensors).view(self.population_size, -1)
            for tensors in zip(*(
                (
                    self.environment.get_strategy_reward(
                        strategy=strategy, opponent_model=cross_model,
                        **self.strat_to_player_kwargs)[0].detach().view(1),
                    cross_epsilon
                )
                for (cross_model, cross_epsilon) in opp_population
                ))
            )

        baseline = \
            self.environment.get_strategy_reward(
                strategy=strategy, opponent_model=opponent_model,
                **self.strat_to_player_kwargs
            )[0].detach().view(1) if self.baseline == 'current_reward' \
            else cross_rewards.mean(dim=0) if self.baseline == 'mean_reward' \
            else self.baseline # a float

        cross_denominator = self.sigma * cross_rewards.std() if self.normalize_gradients else self.sigma**2
        if cross_denominator == 0:
            # all candidates returned same reward and normalize is true --> stationary
            cross_gradient_vector = torch.zeros_like(parameters_to_vector(self.params()))
        else:
            cross_gradient_vector = ((cross_rewards - baseline)*cross_epsilons).mean(dim=0) / cross_denominator

        return cross_gradient_vector

    def opp_cross_gradient_vector(self, strategy, opponent_model)-> torch.Tensor:
        """Calculate self gradient over opponent, R2/Q1

        Only required for simultaneous version of this learner.
        """
        population = (self._perturb_model(strategy) for _ in range(self.population_size))
        cross_rewards, cross_epsilons = (
            torch.cat(tensors).view(self.population_size, -1)
            for tensors in zip(*(
                (
                    self.environment.get_strategy_reward(
                        strategy=cross_model, opponent_model=opponent_model,
                        **self.strat_to_player_kwargs)[1].detach().view(1),
                    cross_epsilon
                )
                for (cross_model, cross_epsilon) in population
                ))
            )

        baseline = \
            self.environment.get_strategy_reward(
                strategy=strategy, opponent_model=opponent_model,
                **self.strat_to_player_kwargs
            )[1].detach().view(1) if self.baseline == 'current_reward' \
            else cross_rewards.mean(dim=0) if self.baseline == 'mean_reward' \
            else self.baseline # a float

        cross_denominator = self.sigma * cross_rewards.std() if self.normalize_gradients else self.sigma**2
        if cross_denominator == 0:
            # all candidates returned same reward and normalize is true --> stationary
            opp_cross_gradient_vector = torch.zeros_like(parameters_to_vector(self.params()))
        else:
            opp_cross_gradient_vector = ((cross_rewards - baseline)*cross_epsilons).mean(dim=0) / cross_denominator

        return opp_cross_gradient_vector

class SOS_ESPGLearner(ESPGLearner):
    """SOS implementation with ESPG.
    
       SOS Algorithm Reference: https://arxiv.org/abs/1811.08469

       SOS Gradient for player 1 is calculated as: -（R1/Q1 + eta * R2/Q1 * R1/Q1Q2 + p*(eta * R1/Q2 * R2/Q2Q1)

       First order gradient R1/Q1 is calculated with ESPG. For the second derivative term R2/Q2Q1, first calculate R2/Q2 with ADG, then take   
       derivative over Q1 with ESPG. For the second derivative term R1/Q1Q2, first calculate R1/Q1 with ADG, then take   
       derivative over Q2 with ESPG.

       ESPG is calculated as: mean( rewards - baseline) * epsilons / sigma²
            and approximates the true gradient.

    Author: Liu Wusheng.
    """
    def __init__(self,
                 model: torch.nn.Module, environment: Environment, hyperparams: dict,
                 optimizer_type: Type[torch.optim.Optimizer], optimizer_hyperparams: dict,
                 strat_to_player_kwargs: dict = None):
        # Create and validate optimizer
        super().__init__(model=model, environment=environment,
                         hyperparams=hyperparams, optimizer_type=optimizer_type,
                         optimizer_hyperparams=optimizer_hyperparams,
                         strat_to_player_kwargs=strat_to_player_kwargs)

        # Note: There are two SOS versions 1 (default) and 2 (simultaneous updating, faster?)
        self.simultaneous = False

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

        # Validate and set SOS hyperparams
        if not set(['eta']) <= set(hyperparams):
            print('Fallback to default second order SOS step size')
            self.eta = 3e-3
        else:
            self.eta = hyperparams['eta']

        self.epoch = 0
        self.writer = None

    def _set_gradients(self, opponent_model):

        ### 1. if required redraw valuations / perform random moves (determined by env)
        self.environment.prepare_iteration()

        ### 2. Create a population of perturbations of the original model
        population = (self._perturb_model(self.model) for _ in range(self.population_size))
        population2 = (self._perturb_model(self.model) for _ in range(self.population_size))
        population3 = (self._perturb_model(self.model) for _ in range(self.population_size))

        ### 3. let each candidate against the environment and get their utils ###
        # both of these as a row-matrix. i.e.
        # rewards: population_size x 1
        # epsilons: population_size x parameter_length

        # Calculate gradient of self, R1/Q1
        rewards, epsilons = (
            torch.cat(tensors).view(self.population_size, -1)
            for tensors in zip(*(
                (
                    self.environment.get_strategy_reward(
                        strategy=model, opponent_model=opponent_model,
                        **self.strat_to_player_kwargs)[0].detach().view(1),
                    epsilon
                )
                for (model, epsilon) in population
                ))
            )

        # Calculate self gradient over opponent, R1/Q2
        cross_gradient_vector = self.cross_gradient_vector(
            strategy=self.model, opponent_model=opponent_model)

        # Calculate second order derivative, R2/Q1Q2
        sec_rewards, sec_epsilons = (
            torch.cat(tensors).view(self.population_size, -1)
            for tensors in zip(*(
                (
                    self.opp_gradient_vector(
                        strategy=model, opponent_model=opponent_model).detach(),
                    epsilon
                )
                for (model, epsilon) in population2
                ))
            )

        # Calculate gradient of opponent,R2/Q2
        opp_gradient_vector = self.opp_gradient_vector(
            strategy=self.model, opponent_model=opponent_model
        ).detach()

        # Calculate second order derivative, R1/Q2Q1
        sos_rewards, sos_epsilons = (
            torch.cat(tensors).view(self.population_size, -1)
            for tensors in zip(*(
                (
                    self.cross_gradient_vector(
                        model, opponent_model).detach(),
                    epsilon
                )
                for (model, epsilon) in population3
                ))
            )

        ### 4. calculate the ES-pseuogradients   ####
        baseline = \
            self.environment.get_strategy_reward(
                strategy=self.model, opponent_model=opponent_model,
                **self.strat_to_player_kwargs
            )[0].detach().view(1) if self.baseline == 'current_reward' \
            else rewards.mean(dim=0) if self.baseline == 'mean_reward' \
            else self.baseline # a float

        sec_baseline = \
            self.opp_gradient_vector(strategy=self.model, opponent_model=opponent_model).detach() \
                if self.baseline == 'current_reward' \
            else sec_rewards.mean(dim=0)

        sos_baseline = \
            self.cross_gradient_vector(strategy=self.model, opponent_model=opponent_model).detach() \
                if self.baseline == 'current_reward' \
            else sos_rewards.mean(dim=0)

        denominator = self.sigma * rewards.std() if self.normalize_gradients else self.sigma**2
        sec_denominator = self.sigma * sec_rewards.std() if self.normalize_gradients else self.sigma**2
        sos_denominator = self.sigma * sos_rewards.std() if self.normalize_gradients else self.sigma**2

        if denominator == 0:
            # all candidates returned same reward and normalize is true --> stationary
            gradient_vector = torch.zeros_like(parameters_to_vector(self.params()))
        else:
            gradient_vector = ((rewards - baseline)*epsilons).mean(dim=0) / denominator

        if sec_denominator == 0:
            # all candidates returned same reward and normalize is true --> stationary
            sec_gradient_vector = torch.zeros_like(
                parameters_to_vector(self.params()),
                parameters_to_vector(self.params()))
        else:
            sec_gradient_vector = torch.einsum(
                'ba,bc->bac', (sec_rewards - sec_baseline), sec_epsilons
            ).mean(dim=0) / sec_denominator

        if sos_denominator == 0:
            # all candidates returned same reward and normalize is true --> stationary
            sos_gradient_vector = torch.zeros_like(
                parameters_to_vector(self.params()),
                parameters_to_vector(self.params()))
        else:
            sos_gradient_vector = torch.einsum(
                'ba,bc->bac', (sos_rewards - sos_baseline), sos_epsilons
            ).mean(dim=0) / sos_denominator

        # Calculate LOLA second order correction and SOS LookAhead second order correction term for self
        LOLA_gradient = torch.einsum('b,bc->c', cross_gradient_vector, sec_gradient_vector)
        SOS_gradient = torch.einsum('b,bc->c', opp_gradient_vector, sos_gradient_vector)

        a = 0.5
        b = 0.1
        if self.simultaneous:
            # Calculate opponent gradient over self, L2/Q1
            opp_cross_gradient_vector = self.opp_cross_gradient_vector(self.model, opponent_model)

            # Calculate LOLA second order correction and SOS LookAhead second order correction term for opponent
            opp_LOLA_gradient = torch.einsum('b,bc->c', opp_cross_gradient_vector, sos_gradient_vector)
            opp_SOS_gradient = torch.einsum('b,bc->c', gradient_vector, sec_gradient_vector)

            # SOS Algorithm gradient
            xi_0 = torch.cat(
                [
                    gradient_vector + self.eta*SOS_gradient,
                    opp_gradient_vector + self.eta*opp_SOS_gradient
                ],
                0
            )
            chi = torch.cat([LOLA_gradient, opp_LOLA_gradient], 0)

            dot = torch.dot(self.eta*chi, xi_0)
            p1 = 1 if dot >= 0 else min(1, -a*torch.norm(xi_0)**2/dot)
            xi_norm = torch.norm(torch.cat([gradient_vector, opp_gradient_vector], 0))
            p2 = xi_norm**2 if xi_norm < b else 1
            p = min(p1, p2)

            gradient_vector, opp_gradient_vector = (xi_0 + p*self.eta*chi).view(2, -1)

        else:
            # SOS Algorithm gradient
            xi_0 = gradient_vector + self.eta*SOS_gradient
            chi = LOLA_gradient
            dot = torch.dot(self.eta*chi, xi_0)
            p1 = 1 if dot >= 0 else min(1, -a*torch.norm(xi_0)**2/dot)
            xi_norm = torch.norm(gradient_vector)
            p2 = xi_norm**2 if xi_norm < b else 1
            p = min(p1, p2)
            gradient_vector = xi_0+p*self.eta*chi

        # Logging P-value
        if self.strat_to_player_kwargs == {'player_position': 0}:
            self.epoch += 1
            self.writer.add_scalar('learner/dot', dot, self.epoch)
            self.writer.add_scalar('learner/xi_norm', xi_norm, self.epoch)
            self.writer.add_scalar('learner/p1_value', p1, self.epoch)
            self.writer.add_scalar('learner/p2_value', p2, self.epoch)
            self.writer.add_scalar('learner/p_value', p, self.epoch)

        # put gradient vector into same format as model parameters
        gradient_params = deepcopy(list(self.params()))
        vector_to_parameters(gradient_vector, gradient_params)
        if self.simultaneous:
            opp_gradient_params = deepcopy(list(opponent_model.parameters()))
            vector_to_parameters(opp_gradient_vector, opp_gradient_params)

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

        if self.simultaneous:
            for p, d_p in zip(opponent_model.parameters(), opp_gradient_params):
                if p.grad is not None:
                    p.grad.add_(-d_p)
                else:
                    p.grad = -d_p

    def opp_gradient_vector(self, strategy, opponent_model) -> torch.Tensor:
        """Calculate gradient of opponent, R2/Q2"""
        opp_population = (self._perturb_model(opponent_model) for _ in range(self.population_size))
        opp_rewards, opp_epsilons = (
            torch.cat(tensors).view(self.population_size, -1)
            for tensors in zip(*(
                (
                    self.environment.get_strategy_reward(
                        strategy=strategy, opponent_model=opp_model,
                        **self.strat_to_player_kwargs)[1].detach().view(1),
                    opp_epsilon
                )
                for (opp_model, opp_epsilon) in opp_population
                ))
            )

        opp_baseline = \
            self.environment.get_strategy_reward(
                strategy=strategy, opponent_model=opponent_model,
                **self.strat_to_player_kwargs
            )[1].detach().view(1) if self.baseline == 'current_reward' \
            else opp_rewards.mean(dim=0) if self.baseline == 'mean_reward' \
            else self.baseline # a float

        opp_denominator = self.sigma * opp_rewards.std() if self.normalize_gradients else self.sigma**2

        if opp_denominator == 0:
            # all candidates returned same reward and normalize is true --> stationary
            opp_gradient_vector = torch.zeros_like(parameters_to_vector(self.params()))
        else:
            opp_gradient_vector = ((opp_rewards - opp_baseline)*opp_epsilons).mean(dim=0) / opp_denominator
        return opp_gradient_vector

    def cross_gradient_vector(self, strategy, opponent_model) -> torch.Tensor:
        """Calculate self gradient over opponent, R1/Q2 """
        opp_population = (self._perturb_model(opponent_model) for _ in range(self.population_size))

        cross_rewards, cross_epsilons = (
            torch.cat(tensors).view(self.population_size, -1)
            for tensors in zip(*(
                (
                    self.environment.get_strategy_reward(
                        strategy=strategy, opponent_model=cross_model,
                        **self.strat_to_player_kwargs)[0].detach().view(1),
                    cross_epsilon
                )
                for (cross_model, cross_epsilon) in opp_population
                ))
            )

        baseline = \
            self.environment.get_strategy_reward(
                strategy=strategy, opponent_model=opponent_model,
                **self.strat_to_player_kwargs
            )[0].detach().view(1) if self.baseline == 'current_reward' \
            else cross_rewards.mean(dim=0) if self.baseline == 'mean_reward' \
            else self.baseline # a float

        cross_denominator = self.sigma * cross_rewards.std() if self.normalize_gradients else self.sigma**2
        if cross_denominator == 0:
            # all candidates returned same reward and normalize is true --> stationary
            cross_gradient_vector = torch.zeros_like(parameters_to_vector(self.params()))
        else:
            cross_gradient_vector = ((cross_rewards - baseline)*cross_epsilons).mean(dim=0) / cross_denominator

        return cross_gradient_vector

    def opp_cross_gradient_vector(self, strategy, opponent_model)-> torch.Tensor:
        """Calculate self gradient over opponent, R2/Q1

        Only required for simultaneous version of this learner.
        """
        population = (self._perturb_model(strategy) for _ in range(self.population_size))
        cross_rewards, cross_epsilons = (
            torch.cat(tensors).view(self.population_size, -1)
            for tensors in zip(*(
                (
                    self.environment.get_strategy_reward(
                        strategy=cross_model, opponent_model=opponent_model,
                        **self.strat_to_player_kwargs)[1].detach().view(1),
                    cross_epsilon
                )
                for (cross_model, cross_epsilon) in population
                ))
            )

        baseline = \
            self.environment.get_strategy_reward(
                strategy=strategy, opponent_model=opponent_model,
                **self.strat_to_player_kwargs
            )[1].detach().view(1) if self.baseline == 'current_reward' \
            else cross_rewards.mean(dim=0) if self.baseline == 'mean_reward' \
            else self.baseline # a float

        cross_denominator = self.sigma * cross_rewards.std() if self.normalize_gradients else self.sigma**2
        if cross_denominator == 0:
            # all candidates returned same reward and normalize is true --> stationary
            opp_cross_gradient_vector = torch.zeros_like(parameters_to_vector(self.params()))
        else:
            opp_cross_gradient_vector = ((cross_rewards - baseline)*cross_epsilons).mean(dim=0) / cross_denominator

        return opp_cross_gradient_vector

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
                strategy=self.model, **self.strat_to_player_kwargs
                ).detach().view(1)
        else:
            pass # is already constant float

        loss = -self.environment.get_strategy_reward(
            strategy=self.model, **self.strat_to_player_kwargs
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
                        strategy=model, aggregate_batch=False,
                        **self.strat_to_player_kwargs
                    ).detach().view(1,n_batch, 1),
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
