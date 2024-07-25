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

from typing import Optional, Union, List, Dict

import random
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
torch.use_deterministic_algorithms(True)

from .network import Autoencoder
from .diagnostics import decoder_jacobian_determinants, EncoderIndicatome, DecoderIndicatome, encoder_indicatrices, decoder_indicatrices

class ViVAE:
    """ViVAE dimension-reduction model"""
    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 2,
        hidden_dims: List[int] = [32,64,128,32],
        variational: bool = True,
        activation = torch.nn.GELU(),
        random_state: Optional[int] = None
    ):
        """ViVAE dimension-reduction model

        Args:
            input_dim (int): Dimension of input data.
            latent_dim (int, optional): Dimension of latent space. Defaults to 2.
            hidden_dims (List[int], optional): Dimensions of hidden layers (order for encoder; decoder takes reverse). Defaults to [32,64,128,32].
            variational (bool, optional): Whether to use a VAE with isotropic Gaussian latent prior. Defaults to False.
            activation (optional): Activation function (instantiated `torch` module). Defaults to `torch.nn.GELU()`.
            random_state (int, optional): Random state to use for reproducibility.
        """
        self.net = Autoencoder(
            input_dim=input_dim, latent_dim=latent_dim, hidden_dims=hidden_dims,
            variational=variational, activation=activation
        )
        self.data_loader = None
        self.optimizer = None

        self.trained = False
        self.decoder_active = False

        self.random_state = random_state

    def __repr__(self):
        return f'ViVAE(input_dim={self.net.input_dim}, latent_dim={self.net.latent_dim})'
    
    def __str__(self):
        trained = 'not trained'
        if self.trained:
            trained = 'trained'
        return f'ViVAE model {self.net.input_dim}->{self.net.latent_dim}, {trained}'

    def train_epoch(
        self,
        lam_recon: float = 1.,
        lam_kldiv: float = 1.,
        lam_geom: float = 0.,
        lam_egeom: float = 0.,
        lam_mds: float = 100.,
        mds_distf: str = 'euclidean',
        mds_nsamp: int = 4
    ) -> Dict:
        """Train for one epoch

        Args:
            lam_recon (float, optional): Weight of reconstruction loss term. Defaults to 1.
            lam_kldiv (float, optional): Weight of KL divergence from latent prior. Defaults to 1.
            lam_geom (float, optional): Weight of geometric loss term. Defaults to 0.
            lam_egeom (float, optional): Weight of encoder-geometric loss term. Defaults to 0.
            lam_mds (float, optional): Weight of MDS loss term. (We recommend 100 for scRNA-seq data and 10 for cytometry data.) Defaults to 100.
            mds_distf (str, optional): Distance function to be used by MDS loss. Either 'euclidean' or 'cosine'. Defaults to 'euclidean'.
            mds_nsamp (int, optional): Repeat-sampling count for computation of MDS loss. Defaults to 4.

        Returns:
            Dict: Losses.
        """

        if self.data_loader is None:
            raise ValueError('No data to train on: use the `fit` or `fit_transform` method')

        recon_error = 0.
        kldiv_error = 0.
        geom_error = 0.
        egeom_error = 0.
        mds_error = 0.

        for _, x in enumerate(self.data_loader):
            x = x[0]
            self.net.train()
            model_loss = self.net(
                x, lam_recon=lam_recon, lam_kldiv=lam_kldiv, lam_geom=lam_geom,
                lam_egeom=lam_egeom, lam_mds=lam_mds, mds_distf=mds_distf,
                mds_nsamp=mds_nsamp
            )

            recon = model_loss['recon']
            if recon is not None:
                recon_error += recon.item()
                if self.net.variational:
                    kldiv = model_loss['kldiv']
                    kldiv_error += kldiv.item()
                    loss = recon + kldiv
                else:
                    loss = recon
            else:
                loss = 0.
            if model_loss['geom'] is not None:
                geom = model_loss['geom']
                geom_error += geom.item()
                loss += geom
            if model_loss['egeom'] is not None:
                egeom = model_loss['egeom']
                egeom_error += egeom.item()
                loss += egeom
            if model_loss['mds'] is not None:
                mds = model_loss['mds']
                mds_error += mds.item()
                loss += mds

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        return {
            'recon': recon_error, 'kldiv': kldiv_error, 'geom': geom_error,
            'egeom': egeom_error, 'mds': mds_error
        }

    def fit(
        self,
        X: np.ndarray,
        n_epochs: int = 50,
        batch_size: int = 256,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        lam_recon: float = 1.,
        lam_kldiv: float = 1.,
        lam_geom: float = 0.,
        lam_egeom: float = 0.,
        lam_mds: float = 100.,
        mds_distf: str = 'euclidean',
        mds_nsamp: int = 4,
        verbose: bool = True
    ):
        """Fit model to data

        Args:
            X (np.ndarray): Input data coordinates.
            n_epochs (int, optional): Number of training epochs. Defaults to 50.
            batch_size (int, optional): Number of points in each training mini-batch. Defaults to 256.
            learning_rate (float, optional): Adam optimiser learning rate. Defaults to 1e-3.
            weight_decay (float, optional): Adam optimiser weight decay rate. Defaults to 1e-4.
            lam_recon (float, optional): Weight of reconstruction loss term. Defaults to 1.
            lam_kldiv (float, optional): Weight of KL divergence from latent prior. Defaults to 1.
            lam_geom (float, optional): Weight of geometric loss term. Defaults to 0.
            lam_egeom (float, optional): Weight of encoder-geometric loss term. Defaults to 0.
            lam_mds (float, optional): Weight of MDS loss term. (We recommend 100 for scRNA-seq data and 10 for cytometry data.) Defaults to 100.
            mds_distf (str, optional): Distance function to be used by MDS loss. Either 'euclidean' or 'cosine'. Defaults to 'euclidean'.
            mds_nsamp (int, optional): Repeat-sampling count for computation of MDS loss. Defaults to 4.
            verbose (bool, optional): Whether to print training progress info. Defaults to True.
        """
        
        if X.shape[1]!= self.net.input_dim:
            raise ValueError('Mismatch between data dimensionality and network shape')

        if lam_geom>0. and lam_recon==0.:
            raise ValueError('Decoder geometric loss can only be used if reconstruction loss is used')

        g = torch.Generator()

        if self.random_state is not None:
            np.random.seed(self.random_state)
            torch.manual_seed(self.random_state)
            random.seed(self.random_state)
            g.manual_seed(self.random_state)
        
        self.data_loader = DataLoader(
            TensorDataset(torch.Tensor(X)),
            batch_size=batch_size,
            shuffle=True,
            generator=g
        )
        self.optimizer = torch.optim.Adam(
            self.net.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        for epoch in range(1, n_epochs+1):
            losses = self.train_epoch(
                lam_recon=lam_recon, lam_kldiv=lam_kldiv, lam_geom=lam_geom,
                lam_egeom=lam_egeom, lam_mds=lam_mds, mds_distf=mds_distf,
                mds_nsamp=mds_nsamp
            )
            if verbose:
                print(f"Epoch {epoch}/{n_epochs}\trecon: {losses['recon']/epoch:.4f}\tkldiv: {losses['kldiv']/epoch:.4f}\tgeom: {losses['geom']/epoch:.4f}\tegeom: {losses['egeom']/epoch:.4f}\tmds: {losses['mds']/epoch:.4f}")

        self.trained = True
        if lam_recon>0.:
            self.decoder_active = True

    def transform(
        self,
        X: np.ndarray,
        batch_size: int = 256
    ) -> np.ndarray:
        """Generate embedding

        Args:
            X (np.ndarray): Input data coordinates.
            batch_size (int, optional): Number of points in each transform mini-batch. Defaults to 256.

        Returns:
            np.ndarray: Latent representation of `X`.
        """
        return self.net.embed(X, batch_size=batch_size)

    def fit_transform(
        self,
        X: np.ndarray,
        n_epochs: int = 50,
        batch_size: int = 256,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        lam_recon: float = 1.,
        lam_kldiv: float = 1.,
        lam_geom: float = 0.,
        lam_egeom: float = 0.,
        lam_mds: float = 100.,
        mds_distf: str = 'euclidean',
        mds_nsamp: int = 4,
        verbose: bool = True
    ):
        """Fit model to data and transform data

        Args:
            X (np.ndarray): Input data coordinates.
            n_epochs (int, optional): Number of training epochs. Defaults to 50.
            batch_size (int, optional): Number of points in each training and transform mini-batch. Defaults to 256.
            learning_rate (float, optional): Adam optimiser learning rate. Defaults to 1e-3.
            weight_decay (float, optional): Adam optimiser weight decay rate. Defaults to 1e-4.
            lam_recon (float, optional): Weight of reconstruction loss term. Defaults to 1.
            lam_kldiv (float, optional): Weight of KL divergence from latent prior. Defaults to 1.
            lam_geom (float, optional): Weight of geometric loss term. Defaults to 0.
            lam_egeom (float, optional): Weight of encoder-geometric loss term. Defaults to 0.
            lam_mds (float, optional): Weight of MDS loss term. (We recommend 100 for scRNA-seq data and 10 for cytometry data.) Defaults to 100.
            mds_distf (str, optional): Distance function to be used by MDS loss. Either 'euclidean' or 'cosine'. Defaults to 'euclidean'.
            mds_nsamp (int, optional): Repeat-sampling count for computation of MDS loss. Defaults to 4.
            verbose (bool, optional): Whether to print training progress info. Defaults to True.
        """
        self.fit(
            X, n_epochs=n_epochs, batch_size=batch_size, learning_rate=learning_rate,
            weight_decay=weight_decay, lam_recon=lam_recon, lam_kldiv=lam_kldiv,
            lam_geom=lam_geom, lam_egeom=lam_egeom, lam_mds=lam_mds, mds_distf=mds_distf,
            mds_nsamp=mds_nsamp, verbose=verbose
        )
        return self.transform(X, batch_size=batch_size)

    def decoder_jacobian_determinants(
        self,
        X: np.ndarray,
        batch_size: int = 256
    ):
        """Compute point-wise Jacobian determinants for decoder

        Args:
            X (np.ndarray): Input data coordinates.
            batch_size (int, optional): Batch size for forward pass of `X` through model. Defaults to 256.

        Returns:
            numpy.ndarray: Array of scaled Jacobian determinant values per row in `X`.
        """
        
        return decoder_jacobian_determinants(model=self.net, x=X, batch_size=batch_size)

    def encoder_indicatrices(
        self,
        X: np.ndarray,
        batch_size: int = 256,
        radius: float = 1e-4,
        n_steps: int = 50,
        all_points: bool = False,
        n_polygon: int = 100
    ) -> EncoderIndicatome:
        """Compute encoder indicatrices

        Args:
            model (Autoencoder): Trained Autoencoder object.
            X (np.ndarray): Input data coordiantes.
            batch_size (int, optional): Batch size for forward pass of `X` through model. Defaults to 256.
            radius (float, optional): Hypersphere radius in ambient space. Defaults ot 1e-4.
            n_steps (int, optional): Number of steps along each axis of latent representation to generate grid of indicatrices. Defaults to 50.
            all_points (bool, optional): Overrides `n_steps` and uses all points. Computationally intensive. Do not do this unless completely sure. Defaults to False.
            n_polygon (int, optional): Number of points in each polygon approximating the hypersphere in ambient space. Defaults to 100.

        Returns:
            EncoderIndicatome: Set of encoder indicatrices.
        """
        return encoder_indicatrices(
            model=self.net, x=X, batch_size=batch_size, radius=radius, n_steps=n_steps,
            all_points=all_points, n_polygon=n_polygon
        )
    
    def decoder_indicatrices(
        self,
        X: np.ndarray,
        batch_size: int = 256,
        n_steps: int = 50,
        n_polygon: int = 100
    ) -> DecoderIndicatome:
        """Compute decoder indicatrices

        Args:
            X (np.ndarray): Input data coordinates.
            batch_size (int, optional): Batch size for forward pass of `X` through model. Defaults to 256.
            n_steps (int, optional): Number of steps along each axis of latent representation to generate grid of indicatrices. Defaults to 50.
            n_polygon (int, optional): Number of points in each polygon approximating the unit sphere in pullback metric. Defaults to 50.

        Returns:
            DecoderIndicatome: Set of decoder indicatrices.
        """

        if not self.decoder_active:
            raise RuntimeError('Cannot compute decoder indicatrices if decoder was not active in training')

        return decoder_indicatrices(
            model=self.net, x=X, batch_size=batch_size, n_steps=n_steps, n_polygon=n_polygon
        )
    
