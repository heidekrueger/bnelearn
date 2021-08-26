import random
import torch
import os
import gpytorch

import numpy as np

from pathlib import Path
from math import ceil


from bnelearn.experiment.configuration_manager import ConfigurationManager
from bnelearn.parameter_estimators.evaluation_module import Evaluation_Module
from bnelearn.strategy import NeuralNetStrategy

class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

class Bayesian_Optimizer:

    def __init__(self, 
                 evaluation_module: Evaluation_Module, 
                 log_root_dir: str, 
                 gpu: int, 
                 initial_samples: int = 10, 
                 param_bounds: float = [0, 1], 
                 n_players: int = 2, 
                 iterations: int = 15,
                 maximization: bool = False):

        self.initial_samples = initial_samples
        self.param_bounds = param_bounds
        self.log_root_dir = log_root_dir
        self.gpu = gpu
        self.n_players = n_players
        self.evaluation_module = evaluation_module

        self.alpha = 1e-5
        self.n_restarts_optimizer = 10
        self.iterations = iterations

        self.grid_size = 2**9 - 2*+5

        self.n_players = n_players

        self.maximization = maximization

        # Method parameter
        self.performances = torch.zeros(self.initial_samples + self.iterations)
        self.params = torch.zeros((self.initial_samples + self.iterations, len(self.param_bounds)))
        self.xi = 0.01

    def _generate_parameter_grid(self):
        
        dims = len(self.param_bounds)

        # for each dim, calculate weight 
        diffs = [self.param_bounds[k][1] - self.param_bounds[k][0] for k in self.param_bounds]
        weights = [d / sum(diffs) for d in diffs]

        dims = sum(w > 0 for w in weights)

        equi_distance = ceil(self.grid_size / dims)

        # default values
        if weights[4] > 0:
            defaults = [torch.ones(1), torch.zeros(1), torch.zeros(1), torch.ones(1), torch.zeros(1)] # Standardize loss-eta to 1
        else:
            defaults = [torch.ones(1), torch.zeros(1), torch.zeros(1), torch.zeros(1), torch.zeros(1)]

        grid_lines = [torch.linspace(*self.param_bounds[k], equi_distance) if weights[i] > 0 else defaults[i] for i, k in enumerate(self.param_bounds)]
        grid = torch.stack(torch.meshgrid(grid_lines), dim=-1).reshape(equi_distance**dims, len(self.param_bounds)).cuda()

        return grid

    def _sample_next_query(self, model):

        """
        Query acquisition function (expected improvement) to obtain the most promising query that should be sampled
        """

        # Create Grid
        param_grid = self._generate_parameter_grid()

        # Get predictoins of surrogate model
        f_preds = model(param_grid)
        mu = f_preds.mean
        sigma = f_preds.stddev

        # Calculate expected improvement criterion
        if self.maximization:
            Z = lambda m, s: (m - self.best - self.xi) / s
            ei = (mu - self.best - self.xi) * torch.distributions.Normal(0, 1).cdf(Z(mu, sigma)) + \
                sigma * torch.exp(torch.distributions.Normal(0, 1).log_prob(Z(mu, sigma)))
        else:
            Z = lambda m, s: (self.best - m - self.xi) / s
            ei = (self.best - mu - self.xi) * torch.distributions.Normal(0, 1).cdf(Z(mu, sigma)) + \
                sigma * torch.exp(torch.distributions.Normal(0, 1).log_prob(Z(mu, sigma)))

        ei[sigma == 0] = 0

        best_index = torch.argmax(ei)

        return param_grid[best_index]

    def _fit_gp(self, model, likelihood, training_iter, X, y):
        model.train()
        likelihood.train()

        # Use the adam optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=0.1)  # Includes GaussianLikelihood parameters

        # "Loss" for GPs - the marginal log likelihood
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

        for i in range(training_iter):
            # Zero gradients from previous iteration
            optimizer.zero_grad()
            # Output from model
            output = model(X)
            # Calc loss and backprop gradients
            loss = -mll(output, y)
            loss.backward(retain_graph=True)
            print('Iter %d/%d - Loss: %.3f   lengthscale: %.3f   noise: %.3f' % (
                i + 1, training_iter, loss.item(),
                model.covar_module.base_kernel.lengthscale.item(),
                model.likelihood.noise.item()
            ))
            optimizer.step()

        model.eval()
        likelihood.eval()

        return model, likelihood

    def _run_eval_experiment(self, risk, regret_beta, regret_gamma, loss_eta, loss_lambda, i):

        experiment_config, experiment_class = ConfigurationManager(experiment_type='single_item_symmetric_uniform_all_pay', n_runs=1, n_epochs=3500) \
                .set_setting(n_players=self.n_players, risk=risk, regret=[regret_beta, regret_gamma], loss=[loss_eta, loss_lambda]) \
                .set_learning(pretrain_iters = 500, batch_size=2**15) \
                .set_logging(log_root_dir=self.log_root_dir, eval_batch_size=2**15, util_loss_grid_size=2**10, util_loss_batch_size=2**12, 
                             util_loss_frequency=100000, stopping_criterion_frequency=100000, save_models=True) \
                .set_hardware(specific_gpu=self.gpu).get_config()

        # Run experiment
        experiment = experiment_class(experiment_config)
        experiment.run()
        torch.cuda.empty_cache()

        # Evaluate performance
        ## Get and load model 
        ## TODO: Generalize for asymmetric models & settings without model sharing
        [*model_file] = Path(self.log_root_dir).rglob("*.pt")
            
        model = NeuralNetStrategy(input_length=1, hidden_nodes=experiment_config.learning.hidden_nodes,
                                  hidden_activations=experiment_config.learning.hidden_activations)

        model.load_state_dict(torch.load([*model_file][0]))
        model.eval()

        it = i - self.initial_samples

        model_performance = self.evaluation_module.evaluate(model, it)
        model_params = torch.tensor([risk, regret_beta, regret_gamma, loss_eta, loss_lambda])
            
        # store performances and params
        self.performances[i] = model_performance
        self.params[i] = model_params

        ## Delete model file
        os.remove([*model_file][0])

    def optimize(self):

        # Step 1: Setup problem
        ## Get initial samples
        for i in range(self.initial_samples):

            ### Randomly sample parameters
            risk = random.uniform(*self.param_bounds["risk"])
            regret_beta = random.uniform(*self.param_bounds["regret_beta"])
            regret_gamma = random.uniform(*self.param_bounds["regret_gamma"])
            loss_eta = random.uniform(*self.param_bounds["loss_eta"])
            loss_lambda = random.uniform(*self.param_bounds["loss_lambda"])

            self._run_eval_experiment(risk, regret_beta, regret_gamma, loss_eta, loss_lambda, i)

        ## Define and train surrogate model

        self.performances = self.performances.cuda()
        self.params = self.params.cuda()

        self.best = torch.min(self.performances[self.performances > 0])

        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        model = ExactGPModel(self.params[0:self.initial_samples, ], self.performances[0:self.initial_samples, ], likelihood)
        model.cuda()

        # fit surrogate
        model, likelihood = self._fit_gp(model, likelihood, 50, self.params[0:self.initial_samples, ], self.performances[0:self.initial_samples, ])

        for i in range(self.iterations):
            # Step 2: Query acquisition function to obtain a promising sample
            next_query = self._sample_next_query(model)

            # Step 3: Evaluate performance of sample
            self._run_eval_experiment(*next_query.tolist(), self.initial_samples + i)
            self.best = torch.min(self.performances[self.performances > 0])

            ## Plot progress

            # Step 4: Re-optimize surrogate
            model.set_train_data(self.params[0:self.initial_samples + i, ], self.performances[0:self.initial_samples + i, ], strict=False)
            model, likelihood = self._fit_gp(model, likelihood, 50, self.params[0:self.initial_samples + i, ], self.performances[0:self.initial_samples + i, ])

        # Step 5: Return optimal parameters
        best = torch.argmin(self.performances)
        print(self.params[best])

        return [self.performances.cpu().detach().numpy(), self.params.cpu().detach().numpy()]

class MultiObjectiveBayesianOptimizer:


    def __init__(self, param_bounds: float, eval_module: Evaluation_Module, model_sharing: int, iterations: int = 15, initial_iterations: int = 8, num_bidder_types: int = 1,
                 gpu: int = 4):
        
        self.iterations = iterations
        self.initial_iterations = initial_iterations
        self.num_bidder_types = num_bidder_types
        self.param_bounds = param_bounds
        self.eval_module = eval_module
        self.model_sharing = model_sharing

        # Derive variables that are necessary for the conduction of a single experiment run
        self.model_sharing = False if(num_bidder_types > 1) else True

        # TODO: number of players has to fit also 
        #=> da sind die vergleichsbasis nicht mehr stimmt. summe über alle spieler muss = 4 oder 2 sein..., anzahl gewichten anhand der cluster-größen

    def _setup_models(self):
        pass
        
    def _run_eval_experiment(self, risk, regret_beta, regret_gamma, loss_eta, loss_lambda, i):

        experiment_config, experiment_class = ConfigurationManager(experiment_type='single_item_symmetric_uniform_all_pay', n_runs=1, n_epochs=1) \
                .set_setting(n_players=self.num_bidder_types, risk=risk, regret=[regret_beta, regret_gamma], loss=[loss_eta, loss_lambda]) \
                .set_learning(pretrain_iters = 500, batch_size=2**15, model_sharing=self.model_sharing) \
                .set_logging(log_root_dir=self.log_root_dir, eval_batch_size=2**15, util_loss_grid_size=2**10, util_loss_batch_size=2**12, 
                             util_loss_frequency=100000, stopping_criterion_frequency=100000, save_models=True) \
                .set_hardware(specific_gpu=self.gpu).get_config()

        # Run experiment
        experiment = experiment_class(experiment_config)
        experiment.run()
        torch.cuda.empty_cache()

        # Evaluate performance
        ## Get and load model 
        ## TODO: Generalize for asymmetric models & settings without model sharing
        [*model_file] = Path(self.log_root_dir).rglob("*.pt")
            
        model = NeuralNetStrategy(input_length=1, hidden_nodes=experiment_config.learning.hidden_nodes,
                                  hidden_activations=experiment_config.learning.hidden_activations)

        model.load_state_dict(torch.load([*model_file][0]))
        model.eval()

        it = i - self.initial_samples

        model_performance = self.evaluation_module.evaluate(model, it)
        model_params = torch.tensor([risk, regret_beta, regret_gamma, loss_eta, loss_lambda])
            
        # store performances and params
        self.performances[i] = model_performance
        self.params[i] = model_params

        ## Delete model file
        os.remove([*model_file][0])

        pass

    def optimize(self):

        # Step 1: Setup problem structure
        ## Get initial samples
        for i in range(self.initial_iterations):

            ### Randomly sample parameters
            risk = [random.uniform(*self.param_bounds["risk"][t]) for t in range(self.num_bidder_types)]
            regret_beta = [random.uniform(*self.param_bounds["regret_beta"][t]) for t in range(self.num_bidder_types)]
            regret_gamma = [random.uniform(*self.param_bounds["regret_gamma"][t]) for t in range(self.num_bidder_types)]
            loss_eta = [random.uniform(*self.param_bounds["loss_eta"][t]) for t in range(self.num_bidder_types)]
            loss_lambda = [random.uniform(*self.param_bounds["loss_lambda"][t]) for t in range(self.num_bidder_types)]

            self._run_eval_experiment(risk, regret_beta, regret_gamma, loss_eta, loss_lambda, i)

        ## Define and fit surrogate models 

        for i in range(self.iterations):
            print(i)

            # Step 2: Query acquisition function

            # Step 3: Evaluate performance of sample

            # Step 4: Re-optimize surrogate model

        # Step 5: Finalize & return optimal parameters


        