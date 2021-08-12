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

    def __init__(self, evaluation_module: Evaluation_Module, log_root_dir: str, gpu: int,  initial_samples: int = 10, param_bounds: float = [0, 1], n_players: int = 2, iterations: int = 15):

        self.initial_samples = initial_samples
        self.param_bounds = param_bounds
        self.log_root_dir = log_root_dir
        self.gpu = gpu
        self.n_players = n_players
        self.evaluation_module = evaluation_module

        self.alpha = 1e-5
        self.n_restarts_optimizer = 10
        self.iterations = iterations

        self.grid_size = 2**10

        # Method parameter
        self.performances = torch.zeros(self.initial_samples)
        self.params = torch.zeros((self.initial_samples, len(self.param_bounds)))

    def _sample_next_query(self):

        """
        Query acquisition function (expected improvement) to obtain the most promising query that should be sampled
        """

        # Create Grid
        dims = len(self.param_bounds)
        n_points_per_dim = ceil(self.grid_size ** (1/dims))

        lines = [torch.linspace(*self.param_bounds[key], n_points_per_dim) 
                for key in self.param_bounds]

        # todo => passt noch nicht so ganz hier

        grid = torch.stack(torch.meshgrid(lines), dim=-1).view(-1, dims)


        # create equidistant lines along the support in each dimension
        #lines = [torch.linspace(bounds[d][0], bounds[d][1], n_points_per_dim,
        #                        device=device, dtype=dtype)
        #         for d in range(dims)]
        #grid = torch.stack(torch.meshgrid(lines), dim=-1).view(-1, dims)



        pass

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

            experiment_config, experiment_class = ConfigurationManager(experiment_type='single_item_symmetric_uniform_all_pay', n_runs=1, n_epochs=10) \
                .set_setting(n_players=2, risk=risk, regret=[regret_beta, regret_gamma], loss=[loss_eta, loss_lambda]) \
                .set_learning(pretrain_iters = 500, batch_size=2**15) \
                .set_logging(log_root_dir=self.log_root_dir, save_tb_events_to_csv_detailed=True, eval_batch_size=2**15, util_loss_grid_size=2**10, 
                             util_loss_batch_size=2**12, util_loss_frequency=1000, stopping_criterion_frequency=100000,
                             save_models=True) \
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

            model_performance = self.evaluation_module.evaluate(model)
            model_params = torch.tensor([risk, regret_beta, regret_gamma, loss_eta, loss_lambda])
            
            # store performances and params
            self.performances[i] = model_performance
            self.params[i] = model_params

            ## Delete model file
            os.remove([*model_file][0])

        ## Define and train surrogate model

        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        model = ExactGPModel(self.params, self.performances, likelihood)

        # fit surrogate
        self._fit_gp(model, likelihood, 150, self.params, self.performances)

        for i in range(self.iterations):
            # Step 2: Query acquisition function to obtain a promising sample
            next_query = self._sample_next_query()

            # Step 3: Evaluate performance of sample

            # Step 4: Re-optimize surrogate

        # Step 5: Return optimal parameters

        return

class Evolutionary_Optimizer:

    def __init__(self):
        pass