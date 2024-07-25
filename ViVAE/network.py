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

        dec_layers = OrderedDict()
        prev_dim = latent_dim
        for idx, hidden_dim in enumerate(reversed(hidden_dims)):
            dec_layers[f'{(idx+1):02d}DecoderPotential'] = nn.Linear(prev_dim, hidden_dim)
            dec_layers[f'{(idx+1):02d}DecoderActivation'] = activation
            prev_dim = hidden_dim
        dec_layers[f'{(idx+2):02d}DecoderPotential'] = nn.Linear(prev_dim, input_dim)
        
        self.encoder = nn.Sequential(enc_layers)
        self.decoder = nn.Sequential(dec_layers)

        self.reconstruction_error = nn.MSELoss()
        
        self.geometric_error = GeometricLoss()
        self.encoder_geometric_error = EncoderGeometricLoss()

        self.mds_error = MDSLoss()

    def reparameterise(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterisation trick

        Obtains indirect sample from latent space.

        Args:
            mu (torch.Tensor): Latent means.
            logvar (torch.Tensor): Latent log-variance.

        Returns:
            torch.Tensor: Latent space sample.
        """
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        res = mu+eps*std
        return res

    def encode(self, x: torch.Tensor):
        """Encode data

        Args:
            x (torch.Tensor): Input.

        Returns:
            torch.Tensor: Single latent representation (embedding) if not variational, otherwise latent means, latent log-variances and latent space sample.
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
            mds_distf: str = 'euclidean',
            mds_nsamp: int = 5
        ) -> Dict:
        """Run forward pass

        Args:
            x (torch.Tensor): Input.
            lam_recon (float, optional): Weight of reconstruction loss term. Defaults to 1.
            lam_kldiv (float, optional): Weight of KL divergence from latent prior. Defaults to 1.
            lam_geom (float, optional): Weight of geometric loss term. Defaults to 0.
            lam_egeom (float, optional): Weight of encoder-geometric loss term. Defaults to 0.
            lam_mds (float, optional): Weight of MDS loss term. Defaults to 0.
            mds_distf (str, optional): Distance function to be used by MDS loss. Either 'euclidean' or 'cosine'. Defaults to 'euclidean'.
            mds_nsamp (int, optional): Repeat-sampling count for computation of MDS loss. Defaults to 5.
        
        Returns:
            Dict: Losses.
        """
        l_kldiv = None
        if self.variational:
            mu, logvar, z = self.encode(x)
            l_kldiv = lam_kldiv*self.kldiv_error(mu, logvar)
        else:
            z = self.encode(x)
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
            l_mds = lam_mds*self.mds_error(x, z, distf=mds_distf, n_sampling=mds_nsamp)
        return {'recon': l_recon, 'kldiv': l_kldiv, 'geom': l_geom, 'egeom': l_egeom, 'mds': l_mds}
    
    def embed(self, x: Union[np.ndarray, torch.Tensor], batch_size: Optional[int] = 256) -> np.ndarray:
        """Generate embedding of data

        Args:
            x (Union[np.ndarray, torch.Tensor]): Input data.
            batch_size (Optional[int]): Optional batch size if the data should be passed through the model in batches (or None to pass at once). Defaults to 256.

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