from abc import ABC, abstractmethod
import torch
import numpy as np
import math
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib
# from skimage.metrics import peak_signal_noise_ratio
# from scipy.ndimage.filters import convolve
# from scipy import ndimage
from scipy import signal
from scipy.signal import convolve2d
__CONDITIONING_METHOD__ = {}

def register_conditioning_method(name: str):
    def wrapper(cls):
        if __CONDITIONING_METHOD__.get(name, None):
            raise NameError(f"Name {name} is already registered!")
        __CONDITIONING_METHOD__[name] = cls
        return cls
    return wrapper

def get_conditioning_method(name: str, operator, noiser, **kwargs):
    if __CONDITIONING_METHOD__.get(name, None) is None:
        raise NameError(f"Name {name} is not defined!")
    return __CONDITIONING_METHOD__[name](operator=operator, noiser=noiser, **kwargs)


def scale_image(im, vlow, vhigh, ilow=None, ihigh=None):
    if ilow is None or ihigh is None:
        ilow = im.min()
        ihigh = im.max()
    imo = (im - ilow) / (ihigh - ilow) * (vhigh - vlow) + vlow
    return imo

class ConditioningMethod(ABC):
    def __init__(self, operator, noiser, **kwargs):
        self.operator = operator
        self.noiser = noiser
    
    def project(self, data, noisy_measurement, **kwargs):
        return self.operator.project(data=data, measurement=noisy_measurement, **kwargs)

    
    def grad_and_value(self, x_prev, x_0_hat, measurement, H_funcs, **kwargs):
        if self.noiser.__name__ == 'gaussian':
            difference = measurement - H_funcs.H(x_0_hat)
            norm = torch.linalg.norm(difference)
            norm_grad = torch.autograd.grad(outputs=norm, inputs=x_prev)[0]
        
        elif self.noiser.__name__ == 'poisson':
            Ax = H_funcs.H(x_0_hat)
            difference = measurement-Ax
            norm = torch.linalg.norm(difference) / measurement.abs()
            norm = norm.mean()
            norm_grad = torch.autograd.grad(outputs=norm, inputs=x_prev)[0]

        else:
            raise NotImplementedError
             
        return norm_grad, norm
   
    @abstractmethod
    def conditioning(self, x_t, measurement, noisy_measurement=None, **kwargs):
        pass
    
@register_conditioning_method(name='vanilla')
class Identity(ConditioningMethod):
    # just pass the input without conditioning
    def conditioning(self, x_t):
        return x_t

@register_conditioning_method(name='projection')
class Projection(ConditioningMethod):
    def conditioning(self, x_t, noisy_measurement, **kwargs):
        x_t = self.project(data=x_t, noisy_measurement=noisy_measurement)
        return x_t

@register_conditioning_method(name='MAP')
class PosteriorSampling_meng(ConditioningMethod):
    def __init__(self, operator, noiser, **kwargs):
        super().__init__(operator, noiser)
        self.coffc = kwargs.get('coffc', 205)
            

    def conditioning(self, x_prev, x_t, x_0_hat, measurement, H_funcs, alpha_t, **kwargs):
        mat = H_funcs.Ht(measurement - H_funcs.H(x_0_hat)).detach()
        mat_x = (mat * x_0_hat.reshape(-1)).sum()
        norm_grad = torch.autograd.grad(outputs = mat_x, inputs=x_prev)[0]

        coeff = (1-alpha_t)/np.sqrt(alpha_t)*self.coffc 
        x_t += coeff * norm_grad

        return x_t