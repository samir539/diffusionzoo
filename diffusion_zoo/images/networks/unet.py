#implementation of a simple unet
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from einops import rearrange

from collections.abc import Callable
from typing import Optional, Union
import math

#position embedding
class sinusoidalPositionEmbeddngs(eqx.Module):
    embedding: jax.Array
    
    def __init__(self,dim):
        half_dim = dim//2
        embedding = math.log(10000)/(half_dim -1)
        self._embedding = jnp.exp(jnp.arange(half_dim)*-embedding)
        
    def __call__(self,x):
        embedding = x*self._embedding
        embedding = jnp.concatenate((jnp.sin(embedding),jnp.cos(embedding)),axis=-1)
        return embedding


