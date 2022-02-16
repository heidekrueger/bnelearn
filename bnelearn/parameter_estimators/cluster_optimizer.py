from operator import matmul
import os
import pandas as pd
import numpy as np
from scipy import interpolate
from scipy.interpolate import splrep, BSpline
from sklearn.cluster import KMeans
from patsy import dmatrix
import torch
import gpytorch


class Cluster_Optimizer:
    
    def __init__(self, data_location: str):
        
        assert os.path.isfile(data_location)

        # Read data 
        self.file = pd.read_csv(data_location)

        # variables
        self.Lambda = {}
        self.Phi = {}
        self.sigmas = {}

    def _get_basis(self, spline):
        

        
        return 

    def _fit_b_splines(self, key):
        """
        Fits a smoothed regression model given a dataset using B-Splines    
        """

        data = self.file[self.file["idCRD"]==key].sort_values("value")

        train_x = torch.tensor(data["value"].values)
        train_y = torch.tensor(data["bid"].values)

        class ExactGPModel(gpytorch.models.ExactGP):
            def __init__(self, train_x, train_y, likelihood):
                super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
                self.mean_module = gpytorch.means.ConstantMean()
                self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

            def forward(self, x):
                mean_x = self.mean_module(x)
                covar_x = self.covar_module(x)
                return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


        # initialize likelihood and model
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        model = ExactGPModel(train_x, train_y, likelihood)

        # Find optimal model hyperparameters
        model.train()
        likelihood.train()

        # Use the adam optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=0.1)  # Includes GaussianLikelihood parameters

        # "Loss" for GPs - the marginal log likelihood
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

        for i in range(50):
            # Zero gradients from previous iteration
            optimizer.zero_grad()
            # Output from model
            output = model(train_x)
            # Calc loss and backprop gradients
            loss = -mll(output, train_y)
            loss.backward()
            print('Iter %d/%d - Loss: %.3f   lengthscale: %.3f   noise: %.3f' % (
                i + 1, 50, loss.item(),
                model.covar_module.base_kernel.lengthscale.item(),
                model.likelihood.noise.item()
            ))
            optimizer.step()

        model.eval()
        likelihood.eval()

        eval_data = torch.range(0, 100, 0.01).double()
        f_preds = model(eval_data)

              
        #t, c, k = splrep(data["value"], data["bid"], xb=0, xe=100, k=2)
        #spline = BSpline(t, c, k, extrapolate=False)
        #B = spline(data["value"])/spline.c[spline.c!=0]

        return f_preds.mean.detach().numpy()

    def _fit_curves(self):
        """
        Fit quadratic regression model for each bidder 
        -> we do not use B-splines or any fancy smoothing method here, because we do not 
        want to overfit the model
        """

        self.bidder_ids = pd.unique(self.file["idCRD"])
        self.bidder_ids = self.bidder_ids[~np.isnan(self.bidder_ids)]

        eval_values = [self._fit_b_splines(k) for k in self.bidder_ids]

        return eval_values


    def _k_means(self, k: int):
        """
        Runs KMeans algorithm for given k clusters
        """
        kmeans = KMeans(n_clusters=k).fit(self.curves)
        
        print(2)
        

        pass

    def get_clusters(self, k):

        # get eval values
        eval_values = self._fit_curves()

        kmeans = KMeans(n_clusters=k).fit(eval_values)
        print(kmeans)


        pass


    def _distance(self):

        i = 0

        pass        


    def _optimal_cluster(self, max_clusters: int = 10):
        """
        Find optimal number of clusters using the gap statistics 
        """
       
        pass