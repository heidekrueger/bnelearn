from functools import partial
from typing import Callable
from abc import ABC, abstractmethod
import torch

############################################################################################
## EQUILIBRIUM FUNCTIONS 
############################################################################################
def regret_equilibrium(risk, beta, gamma):

    if gamma == 0:
        # partial feedback
        eq = lambda x: torch.relu((x**2)/(2+2 * beta * (1-x)))
    elif beta == 0:
        eq = lambda x: torch.relu(((1+gamma) * (1+torch.exp(x*gamma)*(x*gamma-1)))/(gamma**2 * torch.exp(x*gamma)))
    elif beta == gamma:
        eq = lambda x: torch.relu(((1+beta)(-x*beta + (1+beta)(torch.log(1+beta)-torch.log(1+beta-x*beta))))/(beta**2))
    elif 2*beta == gamma:
        eq = lambda x: torch.relu(((1+2*beta)(x*beta+(-1-beta+beta*x)*torch.log(1+beta)+(1+beta-beta*x)*torch.log(1+beta-beta*x)))/(beta**2))
    else:
        eq = lambda x: torch.relu((1+beta-x*beta)**(gamma/beta-1)*(1+gamma)*((1+beta)**(2-gamma/beta)+(1+beta-beta*x)**(1-gamma/beta)*(x*gamma-1-(1+x)*beta))/((gamma-2*beta)*(gamma-beta)))

    return eq

class EquilibriumEstimator(ABC):

    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def get_equilibrium(self):
        pass

class AlgorithmicEstimator(EquilibriumEstimator):

    def __init__(self) -> None:
        pass


class ClosedEstimator(EquilibriumEstimator):

    """
    Estimator for equilibria, where the closed form solution is known in advance
    """

    def __init__(self, szenario: str):
        
        if szenario == "Regret":
            self.equilibrium = regret_equilibrium
        else:
            raise NotImplementedError


    def get_equilibrium(self, risk, regret_beta, regret_gamma):

        eq =  self.equilibrium(risk, regret_beta, regret_gamma)

        return eq
