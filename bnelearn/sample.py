from abc import ABC, abstractmethod
import torch
from torch.distributions import Distribution
import chaospy

class generate_sampler:
    """
    generic class, used to create sampler
    """
    def __init__(self,distribution:Distribution, dim2: int, rule : str = "pseudorandom", device = "cuda", antithetic : bool = False, inplace_sampling :bool = False, scramble : bool = True):
        if rule not in ["pseudorandom", "sobol", "latin_hypercube","halton"]:
            raise "Rule should be one of the following : pseudorandom, sobol, latin_hypercube and halton "
        self.factory = {"sobol" : sobol_sampler, "pseudorandom" : pseudorandom_sampler, "halton" : halton_sampler, "sobol": sobol_sampler, "latin_hypercube":latin_hypercube_sampler}
        self.rule = rule
        self.distribution= distribution
        self.dim2 = dim2
        self.antithetic = antithetic
        self.inplace_sampling = inplace_sampling
        self.device = device
        self.scramble = scramble
    def create_sampler(self) : 
        return self.factory[self.rule](distribution = self.distribution, dim2 = self.dim2, device = self.device, antithetic = self.antithetic, inplace_sampling=self.inplace_sampling, scramble = self.scramble)
        

class sampler(ABC) : 
    def __init__(self,distribution:Distribution , dim2: int, device = "cuda", antithetic : bool = False, inplace_sampling :bool = False, scramble: bool = True) :
        """

        Args:
            distribution (torch.distributions.Distribution): 
            rule (str, optional): Defaults to "pseudorandom", can be either "pseudorandom" or "sobol" or "halton", or "latin_hypercube"
            device (str, optional): Defaults to "cuda".
            antithetic (bool, optional) : Defaults to False
            inplace_sampling (bool, optional) : Defaults to False
            scramble(bool, optional) : only relevant for sobol, use randomized sobol sequences 
        """
        self.distribution = distribution
        self.device = device
        self.dim2 = dim2
        self.inplace_sampling = inplace_sampling
        self.antithetic = antithetic
        self.scramble = scramble
    @abstractmethod
    def draw_uniform_random(self, dim1):
        raise NotImplementedError

    def sample(self, dim1): 
        if not self.antithetic: 
            return self.distribution.icdf(self.draw_uniform_random(dim1))
        else: 
            size = torch.Size([dim1, self.dim2]) if self.dim2>=1 else torch.Size([dim1])
            if self.dim2 >= 1 :
                
                if dim1 % 2 == 0 : 
                    uniform = torch.zeros(size, device = self.device)
                    half_uniform = self.draw_uniform_random(dim1 // 2 + dim1 %2)
                    uniform[::2,:] = half_uniform[:,:]
                    uniform[1::2,:] =1-half_uniform[:,:]
                else: 
                    uniform = torch.zeros(size, device = self.device)
                    half_uniform = self.draw_uniform_random(dim1 // 2 + dim1 %2)
                    uniform[::2,:] = half_uniform[:,:]
                    uniform[1::2,:] =1-half_uniform[: dim1 // 2 + dim1 % 2 -1,:]

            else :
                if dim1 % 2 == 0 : 
                    uniform = torch.zeros(size, device = self.device)
                    half_uniform = self.draw_uniform_random(dim1 // 2 + dim1%2)
                    uniform[::2,] = half_uniform[:]
                    uniform[1::2] =1-half_uniform[:]
                else: 
                    uniform = torch.zeros(size, device = self.device)
                    half_uniform = self.draw_uniform_random(dim1 // 2 + dim1 % 2)
                    uniform[::2] = half_uniform[:]
                    uniform[1::2] =1-half_uniform[:dim1 // 2 + dim1 % 2 -1]

            return self.distribution.icdf(uniform)


 
class sobol_sampler(sampler):
    def __init__(self,distribution:Distribution, dim2: int, device = "cuda", antithetic : bool = False, inplace_sampling :bool = False, scramble:bool = False):
        super().__init__(distribution = distribution , dim2 = dim2, device = device, antithetic = antithetic, inplace_sampling = inplace_sampling, scramble = scramble)
        self.inplace_sampling=False
        sobol_dim = max(self.dim2, 1)
        self.sobol_engine = torch.quasirandom.SobolEngine(sobol_dim, scramble = self.scramble)

    def draw_uniform_random(self, dim1):
        if self.dim2 == 0 : 
             return self.sobol_engine.draw(dim1).reshape(dim1).to(device=self.device)
        else :
            return self.sobol_engine.draw(dim1).to(device=self.device)


class halton_sampler(sampler):
    def __init__(self,distribution:Distribution, dim2: int, device = "cuda", antithetic : bool = False, inplace_sampling :bool = False, scramble : bool = False):
        super().__init__(distribution = distribution, dim2 = dim2, device = device, antithetic = antithetic, inplace_sampling = inplace_sampling, scramble = scramble)
        self.inplace_sampling=False

    def draw_uniform_random(self, dim1):
        if self.dim2 == 0 : 
             return torch.from_numpy(chaospy.create_halton_samples(dim1, dim=1).T.reshape(dim1)).to(self.device)
        else :
            return torch.from_numpy(chaospy.create_halton_samples(dim1, dim=self.dim2).T).to(device = self.device)

class latin_hypercube_sampler(sampler):
    def __init__(self,distribution:Distribution, dim2: int, device = "cuda", antithetic : bool = False, inplace_sampling :bool = False, scramble : bool = False):
        super().__init__(distribution = distribution, dim2 = dim2, device = device, antithetic = antithetic, inplace_sampling = inplace_sampling, scramble = scramble)
        self.inplace_sampling=False

    def draw_uniform_random(self, dim1):
        if self.dim2 == 0: 
             return torch.from_numpy(chaospy.create_latin_hypercube_samples(dim1, dim=1).T.reshape(dim1)).to(self.device)
        else :
            return torch.from_numpy(chaospy.create_latin_hypercube_samples(dim1, dim=self.dim2).T).to(device = self.device)

class pseudorandom_sampler(sampler):
    def __init__(self,distribution:Distribution, dim2: int, device = "cuda", antithetic : bool = False, inplace_sampling :bool = False, scramble:bool = False):
        super().__init__(distribution = distribution, dim2 = dim2, device = device, antithetic = antithetic, inplace_sampling = inplace_sampling, scramble = scramble)
        if (not isinstance(self.distribution, torch.distributions.uniform.Uniform)) and (not isinstance(self.distribution, torch.distributions.normal.Normal)) : 
            self.inplace_sampling = False
        

    def sample(self,dim1):
        if not self.antithetic: 
            if self.inplace_sampling:
                size = torch.Size([dim1,self.dim2]) if self.dim2 >0 else torch.Size([dim1])
                inplace = torch.zeros(size, device = self.device)
                if isinstance(self.distribution, torch.distributions.uniform.Uniform):
                    return inplace.uniform_(self.distribution.low, self.distribution.high)
                else : 
                    return inplace.normal_(mean = self.distribution.loc, std = self.distribution.scale)
            else: 
                return self.distribution.icdf(self.draw_uniform_random(dim1))
        else: 
            size = torch.Size([dim1, self.dim2]) if self.dim2>=1 else torch.Size([dim1])
            if self.dim2 >= 1 : 
                if dim1 % 2 == 0 : 
                    uniform = torch.zeros(size, device = self.device)
                    half_uniform = self.draw_uniform_random(dim1 // 2 + dim1 %2)
                    uniform[::2,:] = half_uniform[:,:]
                    uniform[1::2,:] =1-half_uniform[:,:]
                else: 
                    uniform = torch.zeros(size, device = self.device)
                    half_uniform = self.draw_uniform_random(dim1 // 2 + dim1 %2)
                    uniform[::2,:] = half_uniform[:,:]
                    uniform[1::2,:] =1-half_uniform[: dim1 // 2 + dim1 % 2 -1,:]

            else :
                if dim1 % 2 == 0 : 
                    uniform = torch.zeros(size, device = self.device)
                    half_uniform = self.draw_uniform_random(dim1 // 2 + dim1%2)
                    uniform[::2,] = half_uniform[:]
                    uniform[1::2] =1-half_uniform[:]
                else: 
                    uniform = torch.zeros(size, device = self.device)
                    half_uniform = self.draw_uniform_random(dim1 // 2 + dim1 % 2)
                    uniform[::2] = half_uniform[:]
                    uniform[1::2] =1-half_uniform[:dim1 // 2 + dim1 % 2 -1]

            return self.distribution.icdf(uniform)


    def draw_uniform_random(self, dim1):
        if self.dim2 ==0 : 
            return torch.distributions.uniform.Uniform(0,1).rsample(torch.Size([dim1])).to(self.device)
        else : 
            return torch.distributions.uniform.Uniform(0,1).rsample(torch.Size([dim1,self.dim2])).to(self.device)