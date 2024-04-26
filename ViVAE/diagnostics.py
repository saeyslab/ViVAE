"""
Copyright 2024 David Novak

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from typing import List, Union
import math

import numpy as np
import torch
from torch.utils.data import DataLoader


from scipy.spatial import Delaunay

from .network import Autoencoder
from .geometry import jacobian, metric_tensor, EncoderIndicatome, DecoderIndicatome

def decoder_jacobian_determinants(model: Autoencoder, x: Union[np.ndarray, torch.Tensor], batch_size: int = 256) -> np.ndarray:
    """Compute point-wise Jacobian determinants

    Args:
        model (Autoencoder): Trained autoencoder model.
        x (numpy.ndarray or torch.Tensor): Input dataset.
        batch_size (int, optional): Batch size for forward pass of `x` through `model`. Defaults to 256.

    Returns:
        numpy.ndarray: Array of scaled Jacobian determinant values per row in `x`.
    """

    if isinstance(x, np.ndarray):
        x = torch.FloatTensor(x)
    dets = torch.Tensor([])
    loader = DataLoader(x, batch_size=batch_size, shuffle=False)
    for x in loader:
        x = x[0]
        ## Get latent activations
        if model.variational:
            act, _, _ = model.encode(x)
        else:
            act = model.encode(x)
        ## Compute metric tensor
        jac = jacobian(model.decoder, act)
        metric = metric_tensor(jac)
        jac = torch.func.vmap(torch.func.jacfwd(model.decoder), in_dims=(0,))(act)
        ## Compute determinants
        batch_dets = torch.linalg.det(metric)
        dets = torch.hstack((dets, batch_dets))
    ## Scale determinants
    dets = torch.log10(dets)
    # dets = dets[torch.argwhere(~torch.isnan(dets)).squeeze()]
    rel_dets = dets/torch.abs(torch.mean(dets))
    rel_dets = rel_dets-1
    rel_dets = rel_dets.detach().numpy()
    return rel_dets

def encoder_indicatrices(
    model: Autoencoder,
    x: Union[np.ndarray, torch.Tensor],
    batch_size: int = 256,
    radius: float = 1e-3,
    n_steps: int = 20,
    n_polygon: int = 100
) -> EncoderIndicatome:
    """Compute encoder indicatrices

    Args:
        model (Autoencoder): Trained Autoencoder object.
        x (Union[np.ndarray, torch.Tensor]): Input data to `model`.
        batch_size (int, optional): Batch size for forward pass of `x` through `model`. Defaults to 256.
        radius (float, optional): Hypersphere radius in ambient space. Defaults ot 1e-3.
        n_steps (int, optional): Number of steps along each axis of latent representation to generate grid of indicatrices. Defaults to 20.
        n_polygon (int, optional): Number of points in each polygon approximating the unit sphere in ambient space. Defaults to 100.

    Returns:
        EncoderIndicatome: Set of encoder indicatrices.
    """
    ei = EncoderIndicatome(model=model, x=x)
    ei.compute_indicatrices(r=radius, batch_size=batch_size, n_steps=n_steps, n_polygon=n_polygon)
    return ei

def decoder_indicatrices(
    model: Autoencoder,
    x: Union[np.ndarray, torch.Tensor],
    batch_size: int = 256,
    n_steps: int = 20,
    n_polygon: int = 50
) -> DecoderIndicatome:
    """Compute decoder indicatrices

    Args:
        model (Autoencoder): Trained Autoencoder object.
        x (Union[np.ndarray, torch.Tensor]): Input data to `model`.
        batch_size (int, optional): Batch size for forward pass of `x` through `model`. Defaults to 256.
        n_steps (int, optional): Number of steps along each axis of latent representation to generate grid of indicatrices. Defaults to 20.
        n_polygon (int, optional): Number of points in each polygon approximating the unit sphere in pullback metric. Defaults to 50.

    Returns:
        DecoderIndicatome: Set of decoder indicatrices.
    """
    di = DecoderIndicatome(model=model, x=x)
    di.compute_indicatrices(batch_size=batch_size, n_steps=n_steps, n_polygon=n_polygon)
    return di