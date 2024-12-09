"""
Decoder indicatrix and Jacobian matrix computation adopted from:

Nazari, P., Damrich, S. and Hamprecht, F.A.. (2023).
Geometric Autoencoders - What You See is What You Decode.
Proceedings of the 40th International Conference on Machine Learning,
in Proceedings of Machine Learning Research 202:25834-25857
Available from https://proceedings.mlr.press/v202/nazari23a.html.

Copyright (c) 2022 Philipp Nazari, Sebastian Damrich, Fred Hamprecht

---

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

from ViVAE import torch, DEVICE, DEVICE_NAME

from scipy.spatial import Delaunay

from torch.utils.data import DataLoader
from .mps import MPSDataLoader
from .network import Autoencoder
from .geometry import jacobian, metric_tensor, EncoderIndicatome, DecoderIndicatome

def decoder_jacobian_determinants(model: Autoencoder, x: Union[np.ndarray, torch.tensor], batch_size: int = 256) -> np.ndarray:
    """Compute point-wise scaled Jacobian determinants for decoder

    For each point in our latent representation (lower-dimensional data
    embedding), we define an immersion function, which is the decoder (but can
    be any differentiable function that embeds the latent representation in a
    higher-dimensional ambient space).

    We then compute the Jacobian matrix, which is a linearisation of the
    immersion function, for each point. We proceed to compute the determinant
    of a metric tensor (computed from Jacobian matrices) and apply a log10
    scaling and centering around the mean.

    By quantifying how much the latent space is stretched locally by the
    decoder, the determinant values reflect, indirectly, the local distortions
    of the latent space. This is, however, under the assumption that the
    reconstruction of input data by the decoder is good.

    The determinants are not transferrable between different immersion functions.
    This means that the extent of their size and shape distortion cannot be
    compared between embeddings from different autoencoder models. On the other
    hand, being a local distortion measure, the distortion levels in different
    parts of the embedding can be compared. This can show, for instance, that a
    specific group of points (eg. biological population) seems embedded less
    faithfully than another one.

    Jacobian determinant computation is adopted from the following publication:

    Nazari, P., Damrich, S. and Hamprecht, F.A.. (2023).
    Geometric Autoencoders - What You See is What You Decode.
    Proceedings of the 40th International Conference on Machine Learning,
    in Proceedings of Machine Learning Research 202:25834-25857
    Available from https://proceedings.mlr.press/v202/nazari23a.html.

    Args:
        model (Autoencoder): Trained autoencoder model.
        x (numpy.ndarray or torch.tensor): Input dataset.
        batch_size (int, optional): Batch size for forward pass of `x` through `model`. Defaults to 256.

    Returns:
        numpy.ndarray: Array of scaled Jacobian determinant values per row in `x`.
    """

    if isinstance(x, np.ndarray):
        x = torch.tensor(x)
    dets = torch.tensor([], device=DEVICE)
    if DEVICE_NAME=='mps':
        loader = MPSDataLoader(dataset=x, batch_size=batch_size, shuffle=False)
    else:
        loader = DataLoader(x, batch_size=batch_size, shuffle=False)
    for xx in loader:
        xx = xx[0]
        ## Get latent activations
        if model.variational:
            act, _, _ = model.encode(xx)
        else:
            act = model.encode(xx)
        ## Compute metric tensor
        jac = jacobian(model.decoder, act)
        metric = metric_tensor(jac)
        ## Compute determinants
        batch_dets = torch.linalg.det(metric)
        dets = torch.hstack((dets, batch_dets))
    ## Scale determinants
    dets = torch.log10(dets)
    # dets = dets[torch.argwhere(~torch.isnan(dets)).squeeze()]
    rel_dets = dets/torch.abs(torch.mean(dets))
    rel_dets = rel_dets-1
    rel_dets = rel_dets.detach().cpu().numpy()
    return rel_dets

def encoder_indicatrices(
    model: Autoencoder,
    x: Union[np.ndarray, torch.tensor],
    batch_size: int = 256,
    radius: float = 1e-3,
    n_steps: int = 20,
    all_points: bool = False,
    n_polygon: int = 100
) -> EncoderIndicatome:
    """Compute encoder indicatrices

    To compute encoder indicatrices, we first sample some points in our dataset,
    such that they lie more or less on a grid in our latent space. We proceed to
    define a submersion function for each point, which is the encoder (but can be
    any differentiable function that embeds our points in a lower-dimensional
    latent space). We also compute the Jacobian matrix for each point and its
    submersion. This is a linearisation of the submersion function.
    
    The Jacobian matrix corresponding to any given point and its submersion allows
    us to define a horizontal tangent space for that point. The horizontal tangent
    space is a subspace of the ambient high-dimensional space, and it is the plane
    which gets projected into the 2-dimensional latent space by the encoder.
    
    Next, we imagine a small circular 2-dimensional disc lying on the respective
    horizontal tangent space of each of our points, centered around the given
    point. We sample points from the circumference of each the disc.

    We use the corresponding submersion function to project each disc into our
    2-dimensional latent space again. If superimposed over the embedding, the
    latent representation of these discs show the nature of distortion (change
    in size and shape) introduced by the encoder, for a small neighbourhood of
    each selected point. These latent representations of the sampled circular
    discs are encoder indicatrices. Each indicatrix serves as a local indicator
    of distortion of the latent space (versus the original high-dimensional
    space.)
    
    Encoder indicatrices are not transferrable between different submersion
    functions. This means that the extent of their size and shape distortion
    cannot be compared between embeddings from different models. On the other
    hand, being a local distortion measure, the distortion levels in different
    parts of the embedding can be compared. This can show, for instance, that
    a specific group of points (eg. biological population) is embedded less
    faithfully than another one.
    
    Additionally, if an embedding contains a sparse region (with very few
    cells) separating two compact islands of points, the indicatrices in this
    sparse region will often show that the gap between the islands is
    artificially stretched out. This is because single-cell biological data
    often reflects gradual changes in cellular states, rather than abrupt jumps,
    and the discretisation of populations frequently introduced by some
    dimension-reduction methods (eg. typically t-SNE) is often artificial.

    Args:
        model (Autoencoder): Trained Autoencoder object.
        x (Union[np.ndarray, torch.tensor]): Input data to `model`.
        batch_size (int, optional): Batch size for forward pass of `x` through `model`. Defaults to 256.
        radius (float, optional): Circular disc radius in ambient space. Defaults ot 1e-3.
        n_steps (int, optional): Number of steps along each axis of latent representation to generate grid of indicatrices. Defaults to 20.
        all_points (bool, optional): Overrides `n_steps` and uses all points, instead of grid-sampling. Can be computationally intensive. Defaults to False.
        n_polygon (int, optional): Number of points in each polygon approximating the circular disc in ambient space. Defaults to 100.

    Returns:
        EncoderIndicatome: Set of encoder indicatrices.
    """
    ei = EncoderIndicatome(model=model, x=x)
    ei.compute_indicatrices(r=radius, batch_size=batch_size, n_steps=n_steps, all_points=all_points, n_polygon=n_polygon)
    return ei

def decoder_indicatrices(
    model: Autoencoder,
    x: Union[np.ndarray, torch.tensor],
    batch_size: int = 256,
    n_steps: int = 20,
    n_polygon: int = 50
) -> DecoderIndicatome:
    """Compute decoder indicatrices

    Args:
        model (Autoencoder): Trained Autoencoder object.
        x (Union[np.ndarray, torch.tensor]): Input data to `model`.
        batch_size (int, optional): Batch size for forward pass of `x` through `model`. Defaults to 256.
        n_steps (int, optional): Number of steps along each axis of latent representation to generate grid of indicatrices. Defaults to 20.
        n_polygon (int, optional): Number of points in each polygon approximating the unit sphere in pullback metric. Defaults to 50.

    Returns:
        DecoderIndicatome: Set of decoder indicatrices.
    """
    di = DecoderIndicatome(model=model, x=x)
    di.compute_indicatrices(batch_size=batch_size, n_steps=n_steps, n_polygon=n_polygon)
    return di