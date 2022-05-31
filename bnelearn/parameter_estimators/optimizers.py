from multiprocessing.sharedctypes import Value
import random
import torch
import os
import gpytorch

import numpy as np
import matplotlib.pyplot as plt

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
                bo_params: dict,
                log_root_dir: str, 
                gpu: int):

        self.initial_samples = bo_params['initial_samples']
        self.behavioral_model = bo_params['behavioral_model']
        self.param_bounds = self.behavioral_model['param_bounds']
        self.experiment = bo_params['experiment']
        self.epochs = bo_params['epochs']
        self.n_players = bo_params['n_players']
        self.objective = bo_params['objective']
        self.learner = bo_params['learner']
        self.closed_form = bo_params['closed_form']
        self.payment_rule = bo_params['payment_rule']
        self.u_hi = bo_params['u_hi']

        self.mixed_strategy = 'normal' if self.learner == 'PGLearner' else None


        self.log_root_dir = log_root_dir
        self.gpu = gpu

        self.evaluation_module = bo_params['evaluation_module']

        self.alpha = 1e-5
        self.n_restarts_optimizer = 10
        self.iterations = bo_params['iterations']

        self.grid_size = 2**9 # tbd -> max?
        self.xi = 0.01

        # logging
        self.params = None
        self.performances = None
        self.models = []

    def _generate_parameter_grid(self):
        
        dims = len(self.param_bounds)

        # for each dim, calculate weight 
        # TBD: Here we assume that each behavioral parameter is one dimensional and we only investigate a single one
        # diffs = [self.param_bounds[k][1] - self.param_bounds[k][0] for k in self.param_bounds]
        # weights = [d / sum(diffs) for d in diffs]

        # dims = sum(w > 0 for w in weights)

        # equi_distance = ceil(self.grid_size / dims)

        # # default values
        # defaults = [torch.ones(1), torch.zeros(1), torch.zeros(1)] # Standardize loss-eta to 1

        # # add white noise (sigma^2 = 0.005)
        # grid_lines = [torch.linspace(*self.param_bounds[k], equi_distance) + torch.rand(equi_distance) * 0.005 if weights[i] > 0 else defaults[i] for i, k in enumerate(self.param_bounds)]
        # grid = torch.stack(torch.meshgrid(grid_lines), dim=-1).reshape(equi_distance**dims, len(self.param_bounds)).cuda()

        if self.behavioral_model['type'] == 'risk':
            grid = torch.linspace(self.param_bounds[0], self.param_bounds[1], self.grid_size, device=self.gpu) + torch.rand(self.grid_size, device=self.gpu) * self.xi
        else:
            distances = ceil(self.grid_size / dims)
            grid_lines = [torch.linspace(self.param_bounds[k][0], self.param_bounds[k][1], distances, device=self.gpu) + torch.rand(distances, device=self.gpu) * self.xi for k in range(dims)]
            grid =  torch.stack(torch.meshgrid(grid_lines), dim=-1).reshape(distances**dims, dims).cuda()
        

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
        if self.objective == torch.max:
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
            loss.backward()
            #äloss.backward(retain_graph=True)
            # print('Iter %d/%d - Loss: %.3f   lengthscale: %.3f   noise: %.3f' % (
            #     i + 1, training_iter, loss.item(),
            #     model.covar_module.base_kernel.lengthscale.item(),
            #     model.likelihood.noise.item()
            # ))
            optimizer.step()

        model.eval()
        likelihood.eval()

        return model, likelihood

    def _run_eval_experiment(self, param, i):

        if self.closed_form is not None:
            print(2) # tbd: use closed form expression of the equilibrium strategy 
        else:

            success = False

            while not success:

                experiment_config, experiment_class = ConfigurationManager(experiment_type=self.experiment, n_runs=1, n_epochs=self.epochs)\
                    .set_setting(n_players=self.n_players, **param, payment_rule=self.payment_rule, u_hi=self.u_hi) \
                    .set_logging(log_root_dir=self.log_root_dir, util_loss_frequency=1000, plot_frequency=self.epochs) \
                    .set_hardware(specific_gpu=self.gpu) \
                    .set_learning(pretrain_iters=750, use_valuation=True, batch_size=2 ** 18, mixed_strategy=self.mixed_strategy, learner_type=self.learner) \
                    .get_config()

                # Run experiment
                experiment = experiment_class(experiment_config)
                success, model = experiment.run()

            # Evaluate performance
            performance = self.evaluation_module.evaluate(model, i)

            if self.performances is None:
                self.performances = performance
            else:
                self.performances = torch.vstack((self.performances, performance))
            
            # finalize experiment
            self.models.append(model)
            torch.cuda.empty_cache()

    def _pack_param(self, param):
        if self.behavioral_model['type'] == 'risk':
            param = {'risk': param}
        elif self.behavioral_model['type'] == 'regret':
            param = {'regret': param}
        else:
            raise ValueError("Behavioral Model not found")
        
        return param
            

    def _sample_initial_params(self):

        if self.behavioral_model['type'] == 'regret':
            param = [random.uniform(*self.param_bounds[0]), random.uniform(*self.param_bounds[1])]
        else:
            param = random.uniform(*self.param_bounds)

        if self.params is None:
            self.params = torch.tensor(param, device=self.gpu, dtype=torch.float64)
        else:
            self.params = torch.vstack((self.params, torch.tensor(param, device=self.gpu, dtype=torch.float64)))
            
        param = self._pack_param(param)

        # tbd: add other behavioral models

        return param  

    def optimize(self):

        # Step 1: Setup problem
        ## Get initial (random) samples
        for i in range(self.initial_samples):
            random_param = self._sample_initial_params()
            self._run_eval_experiment(param=random_param, i=i)

        # store best seen sample so far
        self.best = self.objective(self.performances)

        # define surrogate
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        model = ExactGPModel(self.params.squeeze(), self.performances.squeeze(), likelihood)
        model.cuda()

        # fit surrogate
        model, likelihood = self._fit_gp(model, likelihood, 100, self.params.squeeze(), self.performances.squeeze())

        for i in range(self.iterations):
            # Step 2: Query acquisition function to obtain a promising sample
            next_query = self._sample_next_query(model)
            self.params = torch.vstack((self.params, next_query))
            if next_query.dim == 0:
                next_query = self._pack_param(next_query.item())
            else:
                next_query = self._pack_param(next_query.tolist())

            # Step 3: Evaluate performance of sample
            self._run_eval_experiment(next_query, self.initial_samples + i)
            self.best = self.objective(self.performances[self.performances != 0])

            ## Plot progress

            # Step 4: Re-optimize surrogate
            model.set_train_data(self.params.squeeze(), self.performances.squeeze(), strict=False)
            model, likelihood = self._fit_gp(model, likelihood, 100, self.params.squeeze(), self.performances.squeeze())

        # Step 5: Return optimal parameters
        if self.objective == torch.min:
            best = torch.argmin(self.performances) 
        else:
            best = torch.argmax(self.performances)

        if self.params.shape[1] == 1:
            return (self.performances[best].item(), self.params[best].item(), self.models[best])
        else: 
            return (self.performances[best].item(), self.params[best].tolist(), self.models[best])

