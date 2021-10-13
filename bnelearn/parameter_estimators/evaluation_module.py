import os
from tensorboard.compat.proto.tensor_shape_pb2 import TensorShapeProto
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from torch.utils.tensorboard import SummaryWriter, writer
from abc import ABC, abstractmethod

class Evaluation_Module(ABC):

    def __init__(self, data_location: str, writer: SummaryWriter):

        assert os.path.isfile(data_location)

        # Read data 
        self.file = pd.read_csv(data_location)

        # Tensorboard writer
        self.writer = writer
    
    @abstractmethod
    def evaluate(self):
        pass

class CurveEvaluation(Evaluation_Module):

    def __init__(self, n_players: int, data_location: str, full_feedback: bool = False, method: str = "Tobit", writer: SummaryWriter = None):

        assert method == "GLM" or method == "Tobit"

        super().__init__(data_location=data_location, writer = writer)

        self.method = method
        self.n_players = n_players,
        self.full_feedback = full_feedback

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
            if self.n_players == 4:
            ## Since there is no good python package for this purpose, use the estimation in R for the moment
                self.model = lambda x : -0.03422 - 0.47817 * x + 1.30317 * (x ** 2) 
            elif self.n_players == 2 and self.full_feedback:
                self.model = lambda x : -0.09283 + 0.48325 * x + 0.35525 * (x ** 2)
            else:
                self.model = lambda x : -0.12925 + 0.29804 * x + 0.51243 * (x ** 2)
                

    def evaluate(self, eq_model, iteration: int):
        """
        Function to evaluate the performance of a given equilibrium at point v.

        v: The point at which the evaluation should be performed
        b_hat: The equilibrium estimation at this point
        returns the RMSE of the estimation and the model

        """

        values = torch.tensor(np.arange(0, 1.001, 0.001), dtype = torch.float)
        values = values.reshape(len(values), 1)

        b_hat_eq = eq_model(values)
        b_hat_reg = self.model(values)

        if iteration > 0:
            # plot differences
            tb_group = "TIME SERIES"
            figure_name = "bid_function"

            fig = plt.figure()
            plt.scatter(self.file["value"], self.file["bid"])
            plt.plot(values.detach().numpy(), b_hat_eq.detach().numpy(), color="darkred", label="Estimated Equilibrium")
            plt.plot(values.detach().numpy(), b_hat_reg.detach().numpy(), color="darkgreen", label="Regression Model")
            plt.legend(loc="upper left")

            self.writer.add_figure(f'{tb_group}/{figure_name}', fig, iteration)

            plt.close()

        return torch.sqrt(torch.mean((b_hat_eq - b_hat_reg) ** 2)), b_hat_eq, values
        

class StaticsEvaluation(Evaluation_Module):

        def __init__(self, data_location: str, writer: SummaryWriter = None):
            super().__init__(data_location=data_location, writer = writer)

            # calculate total sum of squares 
            self.TSS = np.sum((self.file["bid"] - np.average(self.file["bid"])) ** 2)

        def evaluate(self, eq_model, iteration: int):
            
            # Get prediction from equilibrium model
            values = torch.tensor(self.file["value"]).float()
            values = values.reshape(len(values), 1)

            y_hat = eq_model(values)
            y_hat_tensor = y_hat.clone()
            y_hat = y_hat.detach().numpy().squeeze()

            RSS = np.sum((self.file["bid"] - y_hat) ** 2)

            pseudo_R2 = 1 - RSS/self.TSS

            if iteration > 0:
            # plot differences
                tb_group = "TIME SERIES"
                figure_name = "bid_function"

                fig = plt.figure()
                plt.scatter(self.file["value"], self.file["bid"])
                plt.plot(values.detach().numpy(), y_hat, color="darkred", label="Estimated Equilibrium")
                plt.legend(loc="upper left")

                self.writer.add_figure(f'{tb_group}/{figure_name}', fig, iteration)
                plt.close()

            return pseudo_R2, y_hat_tensor, values
