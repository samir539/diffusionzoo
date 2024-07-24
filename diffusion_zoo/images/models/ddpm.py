import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from einops import rearrange

from collections.abc import Callable
from typing import Optional, Union
import math


class DDPMForwardProcess():
    _noising_space: jax.Array

    def __init__(self,start,stop,timesteps,):
       self._betas = jnp.linspace(start,stop,timesteps) 
       self._alpha = 1. - self._betas
       self._alpha_cumprod = jnp.cumprod(self._alpha,axis=0)
       self._alpha_cumprod_prev = jnp.concatenate([jnp.array([1.0]),self._alpha_cumprod[:-1]])
       self._sqrt_recip_alpha = jnp.sqrt(1.0/self._alpha)
       self._sqrt_alpha_cumprod = jnp.sqrt(self._alpha_cumprod)
       self._sqrt_one_minus_alpha_cumprod = jnp.sqrt(1.0 - self._alpha_cumprod)
       self._posterior_variance = self._betas * (1.0 - self._alpha_cumprod_prev)/(1. - self._alpha_cumprod)
       

    def _get_index_from_list(self,vals,t,x_shape):
        """
        Gathers specific values from a list based on batch indices and reshapes 
        the result for broadcasting.

        This function takes a tensor of values `vals`, a tensor of indices `t`, 
        and a target shape `x_shape`. It gathers the values from `vals` based 
        on the indices in `t`, which should correspond to the batch dimension.
        The gathered values are then reshaped to match the batch size and to be 
        compatible with `x_shape` for broadcasting purposes.
        """
        batch_size = t.shape[0]
        out = jnp.take(vals,t,axis=-1)
        new_shape = (batch_size, *([1] * (len(x_shape)-1)))
        out = jnp.reshape(out,new_shape)
        return out
    
    def forward_diffusion_sample(self,x_0,t,key):
        """Take an image and a timestep and return the noisy version of that image at the given timestep"""
        noise = jax.random.normal(key, shape=x_0.shape) 
        sqrt_alphas_cumprod_t = self._get_index_from_list(self._sqrt_alpha_cumprod,t,x_0.shape)
        sqrt_one_minus_alphas_cumprod_t = self._get_index_from_list(self._sqrt_one_minus_alpha_cumprod,t,x_0.shape)
        
        return sqrt_alphas_cumprod_t * x_0 + sqrt_one_minus_alphas_cumprod_t * noise, noise
        
    
    

# vals = jnp.array([0.5, 0.75, 1.0, 1.25])
# t = jnp.array([2, 0, 3, 1])
# x_shape = (4, 3, 32, 32)         
# ddpm = DDPMForwardProcess(1,2,10)
# print(ddpm._get_index_from_list(vals,t,x_shape))