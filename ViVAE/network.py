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

from collections import OrderedDict
from typing import List, Dict, Union, Optional

import numpy as np

import torch
torch.use_deterministic_algorithms(True)
import torch.nn as nn
from torch.utils.data import DataLoader

from .losses import KLDivLoss, MDSLoss, GeometricLoss, EncoderGeometricLoss

class Autoencoder(nn.Module):
    """Autoencoder model"""
    def __init__(self, input_dim: int, latent_dim: int = 2, hidden_dims: List[int] = [100, 100, 100], variational: bool = False, activation=nn.GELU()):
        """Autoencoder model

        This is a generalised class for a vanilla autoencoder (AE) or a variational autoencoder (VAE).
        The difference is governed by the argument `variational`.

        Reconstruction error is computed as the mean squared error (MSE).
        MSE is chosen due to its sensitivity to outliers (relative to binary cross-entropy), which encourages robustness.

        Default activation function is GELU, chosen as an as established smoother alternative to ReLU.

        Args:
            input_dim (int): Dimension of input data.
            latent_dim (int, optional): Dimension of latent space. Defaults to 2.
            hidden_dims (List[int], optional): Dimensions of hidden layers (order for encoder; decoder takes reverse). Defaults to [100, 100, 100].
            variational (bool, optional): Whether to use a VAE with isotropic Gaussian latent prior. Defaults to False.
            activation (optional): Activation function (instantiated `torch` module). Defaults to `torch.nn.GELU()`.
        """
        super().__init__()

        self.variational = variational

        self.latent_dim = latent_dim
        self.input_dim = input_dim

        ## Put together an encoder using an OrderedDict
        enc_layers = OrderedDict()
        prev_dim = input_dim
        for idx, hidden_dim in enumerate(hidden_dims):
            enc_layers[f'{(idx+1):02d}EncoderPotential'] = nn.Linear(prev_dim, hidden_dim)
            enc_layers[f'{(idx+1):02d}EncoderActivation'] = activation
            prev_dim = hidden_dim
        if self.variational:
            self.mu = nn.Linear(prev_dim, latent_dim)
            self.logvar = nn.Linear(prev_dim, latent_dim)
            
            self.kldiv_error = KLDivLoss()
        else:
            enc_layers[f'{(idx+2):02d}EncoderPotential'] = nn.Linear(prev_dim, latent_dim)

        ## Put together a decoder using an OrderedDict
        dec_layers = OrderedDict()
        prev_dim = latent_dim
        for idx, hidden_dim in enumerate(reversed(hidden_dims)):
            dec_layers[f'{(idx+1):02d}DecoderPotential'] = nn.Linear(prev_dim, hidden_dim)
            dec_layers[f'{(idx+1):02d}DecoderActivation'] = activation
            prev_dim = hidden_dim
        dec_layers[f'{(idx+2):02d}DecoderPotential'] = nn.Linear(prev_dim, input_dim)
        
        self.encoder = nn.Sequential(enc_layers)
        self.decoder = nn.Sequential(dec_layers)

        ## Specify reconstruction error computed using MSE
        self.reconstruction_error = nn.MSELoss()
        
        ## Specify geometric and encoder-geometric error (experimental)
        self.geometric_error = GeometricLoss()
        self.encoder_geometric_error = EncoderGeometricLoss()

        ## Specify MDS error computed using the stochastic-MDS loss algorithm
        self.mds_error = MDSLoss()

    def reparameterise(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterisation trick

        Obtains an indirect sample from a VAE latent space.
        Maintains a probabilistic latent space.

        Args:
            mu (torch.Tensor): Latent means.
            logvar (torch.Tensor): Latent log-variances.

        Returns:
            torch.Tensor: Latent space sample.
        """
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        res = mu+eps*std
        return res

    def encode(self, x: torch.Tensor):
        """Encode data

        Uses a trained encoder module to yield a latent-space embedding of data from input space.
        This data should come from the same domain as the original training data.

        If the model is a vanilla AE, a single latent representation (embedding) of data is returned.
        If the model is a VAE, this returns a list of
        - latent means (`mu`),
        - latent log-variances (`sigma`),
        - the latent-space sample (`z`).

        Note that `z` contains Gaussian-sampled noise.
        `z` is used in training to propagate error through the probabilistic model.
        To obtain an embedding of data for visualisation or downstream processing, use `mu`, which is non-noisy.

        Args:
            x (torch.Tensor): Input.

        Returns:
            - if `variational` is `False`: torch.Tensor object
            - if `variational` is `True`: a list of 3 torch.Tensor objects (`mu`, `sigma`, `z`)
        """
        enc = self.encoder(x)
        if not self.variational:
            return enc
        mu = self.mu(enc)
        logvar = self.logvar(enc)
        z = self.reparameterise(mu, logvar)
        return mu, logvar, z
    
    def submersion(self, x: torch.Tensor) -> torch.Tensor:
        """Submersion function

        Smooth mapping between manifolds (higher-dimensional to lower-dimensional) with surjective derivative.
        This is the encoder in its functional form.

        Args:
            x (torch.Tensor): Points on higher-dimensional manifold.

        Returns:
            torch.Tensor: Projection of `x` onto lower-dimensional manifold.
        """
        enc = self.encoder(x)
        if not self.variational:
            return enc
        mu = self.mu(enc)
        return mu

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent representation

        Uses a trained decoder module to yield a reconstruction of points from latent space in the original input space.
        If the model is a vanilla AE, the latent space is deterministic and its behaviour outside the domain of previously encoded points may be quite odd.
        If the model is a VAE, the latent space is probabilistic and its behaviour outside this domain may be interesting.

        Args:
            z (torch.Tensor): Latent representation (embedding).

        Returns:
            torch.Tensor: Reconstruction of original data.
        """
        z = self.decoder(z)
        return z
    
    def immersion(self, z: torch.Tensor) -> torch.Tensor:
        """Immersion function

        Smooth mapping between manifolds (lower-dimensional to higher-dimensional) with injective derivative.
        This is the decoder in its functional form.

        Args:
            z (torch.Tensor): Points on lower-dimensional manifold.

        Returns:
            torch.Tensor: Projection of `z` onto higher-dimensional manifold.
        """
        return self.decoder(z)

    def forward(
            self,
            x: torch.Tensor,
            lam_recon: float = 1.,
            lam_kldiv: float = 1.,
            lam_geom: float = 0.,
            lam_egeom: float = 0.,
            lam_mds: float = 0.,
            mds_distf_hd: str = 'euclidean',
            mds_distf_ld: str = 'euclidean',
            mds_nsamp: int = 4
        ) -> Dict:
        """Run forward pass

        Runs a forward pass through an autoencoder model.
        This returns a dictionary of all the different loss terms, which are differentiable and can therefore be used in back-propagation to train the model.
        A weight value of 0 or more is assigned to each term, which governs its coefficient in the aggregate loss function.
        
        If a weight of 0 is assigned to any loss term, its computation is skipped, speeding up the forward pass.

        Specifically, if a weight of 0 is assigned to the reconstruction error (`lam_recon=0.`), the decoder of the model is not trained at all.

        The weight for divergence from latent prior (`lam_kldiv`) is ignored if the autoencoder is not variational.

        Args:
            x (torch.Tensor): Input.
            lam_recon (float, optional): Weight of reconstruction loss term. Defaults to 1.
            lam_kldiv (float, optional): Weight of KL divergence from latent prior. Defaults to 1.
            lam_geom (float, optional): Weight of geometric loss term. Defaults to 0.
            lam_egeom (float, optional): Weight of encoder-geometric loss term. Defaults to 0.
            lam_mds (float, optional): Weight of MDS loss term. Defaults to 0.
            mds_distf_hd (str, optional): Input-space distance function to be used by MDS loss. Either 'euclidean' or 'cosine'. Defaults to 'euclidean'.
            mds_distf_ld (str, optional): Latent-space distance function to be used by MDS loss. Either 'euclidean' or 'cosine'. Defaults to 'euclidean'.
            mds_nsamp (int, optional): Repeat-sampling count for computation of MDS loss. Defaults to 4.
        
        Returns:
            Dict: Losses.
        """

        ## Encode input data
        l_kldiv = None
        if self.variational:
            mu, logvar, z = self.encode(x)
            l_kldiv = lam_kldiv*self.kldiv_error(mu, logvar)
        else:
            z = self.encode(x)
        
        ## Decode latent representation if needed
        if lam_recon>0.:
            xhat = self.decode(z)

        l_recon = None
        l_geom = None
        l_egeom = None
        l_mds = None
        if lam_recon>0.:
            l_recon = lam_recon*self.reconstruction_error(x, xhat)
        if lam_geom>0.:
            l_geom = lam_geom*self.geometric_error(self.immersion, z)
        if lam_egeom>0.:
            l_egeom = lam_egeom*self.encoder_geometric_error(self.submersion, x)
        if lam_mds>0.:
            l_mds = lam_mds*self.mds_error(x, z, distf_hd=mds_distf_hd, distf_ld=mds_distf_ld, n_sampling=mds_nsamp)
        return {'recon': l_recon, 'kldiv': l_kldiv, 'geom': l_geom, 'egeom': l_egeom, 'mds': l_mds}
    
    def embed(self, x: Union[np.ndarray, torch.Tensor], batch_size: Optional[int] = 256) -> np.ndarray:
        """Generate an embedding of data

        Uses trained encoder to creates an embedding of input data in the model latent space.
        The input data should come from the same domain as the data on which the model was trained.

        The data can be passed through the encoder in batches or all at once.

        Args:
            x (Union[np.ndarray, torch.Tensor]): Input data.
            batch_size (Optional[int]): Optional batch size (or None to transform all at once). Defaults to 256.

        Returns:
            np.ndarray: Embedding in latent space.
        """
        if isinstance(x, np.ndarray):
            x = torch.Tensor(x)

        if batch_size is not None:
            enc = None
            loader = DataLoader(x, batch_size=batch_size, shuffle=False)
            for x in loader:
                if self.variational:
                    batch_enc, _, _ = self.encode(x)
                else:
                    batch_enc = self.encode(x)
                enc = batch_enc if enc is None else torch.vstack((enc, batch_enc))
        else:
            if self.variational:
                enc,_,_ = self.encode(x)
            else:
                enc = self.encode(x)
        enc = enc.detach().numpy()
        return enc