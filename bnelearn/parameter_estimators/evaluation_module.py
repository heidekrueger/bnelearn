import os
import torch
import pandas as pd
import numpy as np

from abc import ABC, abstractmethod

class Evaluation_Module(ABC):

    def __init__(self, data_location: str):

        assert os.path.isfile(data_location)

        # Read data 
        self.file = pd.read_csv(data_location)
    
    @abstractmethod
    def evaluate(self):
        pass

class CurveEvaluation(Evaluation_Module):

    def __init__(self, data_location: str, method: str = "Tobit"):

        assert method == "GLM" or method == "Tobit"

        super().__init__(data_location=data_location)

        self.method = method

        # Estimate data using quadratic regression models
        self._estimate_model()

    def _estimate_model(self):
        
        # Add value^2 to data set for the subsequent estimation
        self.file = self.file.assign(value_squared = lambda dataframe: dataframe['value'] ** 2)

        if self.method == "GLM":
            # Estimate GLM
            self.model = np.poly1d(np.polyfit(self.file["value"], self.file["bid"], 2))
        elif self.method == "Tobit":
            # Estimate Tobit model
            ## Since there is no good python package for this purpose, use the estimation in R for the moment
            self.model = lambda x : -3.423469 - 0.478134 * x + 0.013032 * (x ** 2) 
        

    def evaluate(self, eq_model):
        """
        Function to evaluate the performance of a given equilibrium at point v.

        v: The point at which the evaluation should be performed
        b_hat: The equilibrium estimation at this point

        returns the RMSE of the estimation and the model

        """

        values = torch.tensor(np.arange(0, 100, 0.1), dtype = torch.float)
        values = values.reshape(len(values), 1)

        b_hat_eq = eq_model(values)
        b_hat_reg = self.model(values)

        return torch.sqrt(torch.mean((b_hat_eq - b_hat_reg) ** 2))
        
