"""
Stochastic group MDS loss adopted from:

Lambert, P., de Bodt, C., Verleysen, M and Lee, John A.. (2022).
SQuadMDS: A lean Stochastic Quartet MDS improving global structure preservation in neighbor embedding like t-SNE and UMAP.
Neurocomputing 503:17-27
Available from https://www.sciencedirect.com/science/article/pii/S0925231222008402

---

Geometric loss adopted from:

Nazari, P., Damrich, S. and Hamprecht, F.A.. (2023).
Geometric Autoencoders - What You See is What You Decode.
Proceedings of the 40th International Conference on Machine Learning,
in Proceedings of Machine Learning Research 202:25834-25857
Available from https://proceedings.mlr.press/v202/nazari23a.html.

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

from itertools import combinations
from typing import Callable, List, Union, Optional

import numpy as np
import torch
torch.use_deterministic_algorithms(True)

from .geometry import jacobian, metric_tensor

class KLDivLoss():
    """ KL-divergece loss

    Computes Kullback-Leibler divergence from latent prior.
    """
    def __init__(self):
        self.eps_std = torch.Tensor([1e-2])
        self.eps_sq = self.eps_std ** 2

    def __call__(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """KL-divergence loss

        Args:
            mu (torch.Tensor): Latent means.
            logvar (torch.Tensor): Latent log-variances.

        Returns:
            torch.Tensor: Averaged KL divergence.
        """
        res = -0.5*torch.mean(logvar+torch.log(self.eps_sq)-torch.square(mu)-self.eps_sq*logvar.exp())
        res /= mu.shape[0]
        return res


class MDSLoss():
    """Stochastic quartet MDS loss
    """
    @staticmethod
    def euclidean_distance(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Euclidean distance

        Args:
            x (torch.Tensor): Row-wise coordinates (batch).
            y (torch.Tensor): Row-wise coordinates (batch).

        Returns:
            torch.Tensor: Euclidean distance (L2) norm.
        """
        res = torch.sqrt(torch.maximum(torch.sum(torch.square(x-y), dim=1, keepdim=False), torch.Tensor([1e-9])))
        return res

    @staticmethod
    def cosine_distance(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Cosine distance

        Args:
            x (torch.Tensor): Row-wise coordinates (batch).
            y (torch.Tensor): Row-wise coordinates (batch).

        Returns:
            torch.Tensor: Euclidean distance (L2) norm.
        """
        res = 1.-torch.nn.functional.cosine_similarity(x, y, dim=1)
        return res

    @staticmethod
    def quartet_norm_dist(x: List[torch.Tensor], z: List[torch.Tensor], i: int, j: int, xd: torch.Tensor, zd: torch.Tensor, distf) -> torch.Tensor:
        """Quartet-normalised distance

        Args:
            x (List[torch.Tensor]): Input points of indices 1-4 in their respective quartets.
            z (List[torch.Tensor]): Embeddings of `x`.
            i (int): First point index across quartets.
            j (int): Second point index across quartets.
            xd (torch.Tensor): Denominator (normaliser) for `x`.
            zd (torch.Tensor): Denominator (normaliser) for `z`.
            distf: Distance function (static method of `MDSLoss`).

        Returns:
            torch.Tensor: Quartet-normalised distances.
        """
        f = distf
        dx = f(x[i], x[j]) / xd
        dz = f(z[i], z[j]) / zd
        D = torch.pow(dx-dz, 2.)
        return D
    
    @staticmethod
    def quartet_norm_factor(x: List[torch.Tensor], distf):
        """Quartet normalisation factor

        Args:
            x (List[torch.Tensor]): Points of indices 1-4 in their respective quartets.
            distf: Distance function (static method of `MDSLoss`).

        Returns:
            torch.Tensor: Normalisation factors.
        """
        f = distf
        res = f(x[0],x[1])+f(x[1],x[2])+f(x[2],x[3])+f(x[0],x[2])+f(x[0],x[3])+f(x[1],x[3])
        return res

    @staticmethod
    def quartet_cost(xq: List[torch.Tensor], zq: List[torch.Tensor], distf) -> torch.Tensor:
        """Quartet cost

        Args:
            xq (List[torch.Tensor]): Input points of indices 1-4 in their respective quartets.
            zq (List[torch.Tensor]): Embeddings of `x`.
            distf: Distance function (static method of `MDSLoss`).

        Returns:
            torch.Tensor: Quartet costs, given by distortion of quartet-normalised distances.
        """
        xd = MDSLoss.quartet_norm_factor(x=xq, distf=distf)
        zd = MDSLoss.quartet_norm_factor(x=zq, distf=distf)
        def d(i, j):
            return MDSLoss.quartet_norm_dist(x=xq, z=zq, i=i, j=j, xd=xd, zd=zd, distf=distf)
        res = torch.mean(d(0,1)+d(1,2)+d(2,3)+d(0,2)+d(0,3)+d(1,3))
        return res

    def __call__(self, x: torch.Tensor, z: torch.Tensor, distf: str = 'euclidean', n_sampling: int = 1) -> torch.Tensor:
        """Stochastic quartet MDS loss

        Penalises change in relative positions of points within random groups of 4 (from SQuadMDS).

        Args:
            x (torch.Tensor): Input.
            z (torch.Tensor): Encoding.
            distf (str, optional): Distance function to be used by MDS loss. Either 'euclidean' or 'cosine'. Defaults to 'euclidean'.
            n_sampling (int, optional): How many times groups are (re-)sampled. Defaults to 1.
        Returns:
            torch.Tensor: Averaged group cost.
        """
        
        n = x.shape[0]
        nq = n//4
        
        if distf=='euclidean':
            distf = MDSLoss.euclidean_distance
        elif distf=='cosine':
            distf = MDSLoss.cosine_distance
        else:
            raise ValueError('Invalid distance function specification for MDS loss')

        loss = torch.FloatTensor([0.])

        for _ in range(n_sampling):
            idcs = torch.randperm(x.size()[0])
        
            idcs = [0, nq, nq*2, nq*3]
            xq = [x[np.arange(idx, idx+nq)] for idx in idcs]
            zq = [z[np.arange(idx, idx+nq)] for idx in idcs]

            res = torch.multiply(MDSLoss.quartet_cost(xq=xq, zq=zq, distf=distf), n)
            res = torch.divide(res, nq)
            loss += res
        
        loss = torch.mean(torch.divide(loss, n_sampling))
        return loss


class GeometricLoss():
    """Geometric loss for decoder

    Penalises local stretching of latent space, quantified via Jacobian of the immersion function defined by decoder.
    """
    def __call__(self, immersion: Callable, z: torch.Tensor) -> torch.Tensor:
        """Geometric loss for decoder

        Args:
            f (Callable): Immersion function (decoder).
            z (torch.Tensor): Latent representation of input to model.

        Returns:
            torch.Tensor: loss
        """

        ## Compute Jacobian matrix
        jac = torch.squeeze(jacobian(immersion, z))
        
        ## Compute metric tensor
        metric = metric_tensor(jac, rev=False)
        
        ## Compute log10 of determinant and prevents NaNs
        log_dets = torch.logdet(metric)
        torch.nan_to_num(log_dets, nan=1., posinf=1., neginf=1.)
        
        ## Compute variance of logdets
        loss = torch.var(log_dets)
        
        return loss
    
class EncoderGeometricLoss():
    """Geometric loss for encoder

    Penalises local stretching of latent space, quantified via Jacobian of the encoder.
    """
    def __call__(self, submersion: Callable, x: torch.Tensor) -> torch.Tensor:
        """Geometric loss for encoder

        Args:
            f (Callable): Submersion function (encoder).
            z (torch.Tensor): Input to model.

        Returns:
            torch.Tensor: loss
        """

        ## Compute Jacobian matrix
        jac = torch.squeeze(jacobian(submersion, x))
        
        ## Compute metric tensor
        metric = metric_tensor(jac, rev=True)
        
        ## Compute log10 of determinant and prevents NaNs
        log_dets = torch.logdet(metric)
        torch.nan_to_num(log_dets, nan=1., posinf=1., neginf=1.)
        
        ## Compute variance of logdets
        loss = torch.var(log_dets)
        
        return loss