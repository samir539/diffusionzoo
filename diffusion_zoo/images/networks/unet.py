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
    _embedding: jax.Array
    
    def __init__(self,dim):
        half_dim = dim//2
        embedding = math.log(10000)/(half_dim -1)
        self._embedding = jnp.exp(jnp.arange(half_dim)*-embedding)
        
    def __call__(self,x):
        embedding = x*self._embedding
        embedding = jnp.concatenate((jnp.sin(embedding),jnp.cos(embedding)),axis=-1)
        return embedding


class Block(eqx.Module):
    _transform: eqx.nn.Conv2d
    _up: bool
    _time_mlp: eqx.nn.Linear
    _conv1: eqx.nn.Conv2d
    _conv2: eqx.nn.Conv2d
    _bnorm1: eqx.nn.BatchNorm
    _bnorm2: eqx.nn.BatchNorm

    
    def __init__(self, in_channels,out_channels,time_embed_dim,up=False,*,key):
        self._time_mlp = eqx.nn.Linear(time_embed_dim,out_channels,key=key)
        self._conv1 = eqx.nn.Conv2d(in_channels,out_channels,3,padding=1,key=key)
        if up:
            # self._conv1 = eqx.nn.Conv2d(2*in_channels,out_channels,3,padding=1,key=key)
            self._transform = eqx.nn.ConvTranspose2d(out_channels, out_channels, 4,2,1,key=key)
        else:
            self._transform = eqx.nn.Conv2d(out_channels,out_channels,4,2,1,key=key)
        self._conv2 = eqx.nn.Conv2d(out_channels,out_channels,3,padding=1,key=key)
        self._bnorm1 = eqx.nn.BatchNorm(out_channels)
        self._bnorm2 = eqx.nn.BatchNorm(out_channels)
        
    def __call__(self,x,t):
        h = self._bnorm1(jax.nn.relu(self._conv1(x)))   #bcwh
        time_embedding = jax.nn.relu(self._time_mlp(t)) #dim [batch,outchannels]
        time_embedding = time_embedding[(...,)+[None,]*2] #prep for broadcasting
        h = h + time_embedding
        h = self._bnorm2(jax.nn.relu(self._conv2(h)))
        return self._transform(h)
        
        
class SimpleUnet(eqx.Module):
    _time_mlp: eqx.nn.Sequential
    _conv0: eqx.nn.Conv2d
    _down_path: list
    _up_path: list
    _output: eqx.nn.Conv2d
    
    def __init__(self,key):
        image_channels = 3
        down_channels = (64,128,256,512,1024)
        up_channels = (1024,512,256,128,64)
        out_dim = 3
        time_embed_dim = 32
        
        self._time_mlp = eqx.nn.Sequential([sinusoidalPositionEmbeddngs(time_embed_dim),eqx.nn.Linear(time_embed_dim,time_embed_dim,key=key), jax.nn.relu])
        
        self._conv0 = eqx.nn.Conv2d(image_channels,down_channels[0],3,padding=1,key=key)
        keys = jax.random.split(key, len(down_channels) + len(up_channels)-2)
        
        self._down_path = [Block(down_channels[i],down_channels[i+1],time_embed_dim,key=keys[i]) for i in range(len(down_channels)-1)]
        self._up_path = [Block(up_channels[i],up_channels[i+1],time_embed_dim,up=True,key=keys[i])for i in range(len(up_channels)-1)]
        
        self._output = eqx.nn.Conv2d(up_channels[-1],out_dim,1,key=keys[-1])
        
        
    def __call__(self,x,timestep):
        t = self._time_mlp(timestep)
        x = self._conv0(x)
        residual_inputs = []
        for down in self._down_path:
            x = down(x,t)
            residual_inputs.append(x)
        for up in self._up_path:
            residual_x = residual_inputs.pop()
            x = jnp.concatenate((x,residual_x),axis=1)
            x = up(x,t)
        return self._output(x)
        
        
        
        
        
        
        
    
