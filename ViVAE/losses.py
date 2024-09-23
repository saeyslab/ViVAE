"""
Stochastic-MDS loss is conceptually based on:

Lambert, P., de Bodt, C., Verleysen, M., et al.. (2021).
Stochastic quartet approach for fast multidimensional scaling.
European Symposium on Artificial Neural Networks, Computational Intelligence and Machine Learning, pp 417-422
Available from https://dial.uclouvain.be/pr/boreal/object/boreal:252550

GitHub repo: PierreLambert3/SQuaD-MDS

---

Geometric loss is conceptually based on:

Nazari, P., Damrich, S. and Hamprecht, F.A.. (2023).
Geometric Autoencoders - What You See is What You Decode.
Proceedings of the 40th International Conference on Machine Learning,
in Proceedings of Machine Learning Research 202:25834-25857
Available from https://proceedings.mlr.press/v202/nazari23a.html.

GitHub repo: phnazari/GeometricAutoencoder

---

We have previously studied a generalisation of the stochastic-MDS loss as 'group loss'.
See the corresponding publication and GitHub repo for a TensorFlow implementation of group loss:

Novak, D., Van Gassen, S. and Saeys, Y.. (2023).
GroupEnc: encoder with group loss for global structure preservation.
Presented at BNAIC/BeNeLearn 2023.
Available from https://arxiv.org/pdf/2309.02917.

GitHub repo: saeyslab/GroupEnc

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

from .geometry import jacobian, metric_tensor

from typing import Callable, List, Union, Optional

import numpy as np
import torch

## We use deterministic variants of PyTorch algorithms where applicable.
## This may be slower, but helps with reproducibility.
torch.use_deterministic_algorithms(True)

class ImitationLoss():
    """Imitation loss

    Encourages encoding to become similar to a reference embedding by minimising L2 distances.
    This can be used for initialising an autoencoder network for increased stability.
    """
    def __init__(self):
        pass
    def __call__(self, x: torch.Tensor, z: torch.Tensor, ref_model: Callable):
        """Compute imitation loss

        Args:
            x (torch.Tensor): Input points.
            z (torch.Tensor): Embedding of input points.
            ref_model (Callable): Reference model that transforms `x`.
        
        Returns:
            torch.Tensor: Loss value averaged across batch.
        """

        z_ref = ref_model(x)
        l2 = torch.sqrt(torch.maximum(torch.sum(torch.square(z-z_ref), dim=1, keepdim=False), torch.Tensor([1e-9])))
        res = torch.mean(l2)
        return res

class FirstNDimsExtractor():
    """First-n-dimensions extractor
    
    A trivial reference model that extracts the first few features of a dataset as its embedding.
    If the input data is PCA-reduced, this amounts to extracting top principal components of data.
    """
    def __init__(self, latent_dim: int):
        self.latent_dim = latent_dim

    def __call__(self, x):
        return x[:,range(self.latent_dim)]

class KLDivLoss():
    """KL divergece loss

    KL divergence from latent prior distribution (isotropic Gaussian) in a VAE.
    """
    def __init__(self):
        self.eps_std = torch.Tensor([1e-2])
        self.eps_sq = self.eps_std ** 2

    def __call__(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Kullback-Leiber divergence loss

        Computes Kullback-Leibler divergence from latent prior.
        Uses mean (not sum) across batch size.

        Args:
            mu (torch.Tensor): Latent means.
            logvar (torch.Tensor): Latent log-variances.

        Returns:
            torch.Tensor: KL divergence.
        """
        res = -0.5*torch.mean(logvar+torch.log(self.eps_sq)-torch.square(mu)-self.eps_sq*logvar.exp())
        res /= mu.shape[0]
        return res


class MDSLoss():
    """Stochastic-MDS loss

    Stochastic multi-dimensional scaling (MDS) loss for multi-scale structure preservation in any differentiable encoder or autoencoder model.
    """
    @staticmethod
    def euclidean_distance(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """L2 (Euclidean) distance

        Scales to multiple pairs of points (row-wise).

        Args:
            x (torch.Tensor): Row-wise coordinates (batch).
            y (torch.Tensor): Row-wise coordinates (batch).

        Returns:
            torch.Tensor: Distances.
        """
        res = torch.sqrt(torch.maximum(torch.sum(torch.square(x-y), dim=1, keepdim=False), torch.Tensor([1e-9])))
        return res

    @staticmethod
    def cosine_distance(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Cosine distance

        Scales to multiple pairs of points (row-wise).

        Args:
            x (torch.Tensor): Row-wise coordinates (batch).
            y (torch.Tensor): Row-wise coordinates (batch).

        Returns:
            torch.Tensor: Distances.
        """
        res = 1.-torch.nn.functional.cosine_similarity(x, y, dim=1)
        return res

    @staticmethod
    def quartet_dissimilarity(x: List[torch.Tensor], z: List[torch.Tensor], i: int, j: int, xd: torch.Tensor, zd: torch.Tensor, distf_hd, distf_ld) -> torch.Tensor:
        """Quartet-normalised dissimilarity

        Computes a differentiable dissimilarity value for a pair of points in input and embedding space.
        The pairwise distance between them is normalised against all other distances within their quartet.

        Scales to multiple quartets (i.e., multiple pairs of points with the same intra-quartet indices but coming from different quartets).

        Args:
            x (List[torch.Tensor]): Input points of indices 1 to 4 in their respective quartets.
            z (List[torch.Tensor]): Embeddings of `x`.
            i (int): First point index across quartets.
            j (int): Second point index across quartets.
            xd (torch.Tensor): Denominator (normaliser) for `x`.
            zd (torch.Tensor): Denominator (normaliser) for `z`.
            distf_hd: Distance function to use in input space (static method of `MDSLoss`).
            distf_ld: Distance function to use in embedding space (static method of `MDSLoss`).

        Returns:
            torch.Tensor: Dissimilarities.
        """
        dx = distf_hd(x[i], x[j]) / xd
        dz = distf_ld(z[i], z[j]) / zd
        D = torch.pow(dx-dz, 2.)
        return D
    
    @staticmethod
    def quartet_norm_factor(x: List[torch.Tensor], distf):
        """Quartet-normalisation factor

        Computes a normalisation value for pairwise point distances w.r.t. their quartets.
        This enables propagation of pairwise distance error across the quartet and, consequently, the joint optimisation of all positions within the quartet.

        Scales to multiple quartets.

        Args:
            x (List[torch.Tensor]): Points of indices 1-4 in their respective quartets.
            distf: Distance function (static method of `MDSLoss`).

        Returns:
            torch.Tensor: Normalisation values.
        """
        f = distf
        res = f(x[0],x[1])+f(x[1],x[2])+f(x[2],x[3])+f(x[0],x[2])+f(x[0],x[3])+f(x[1],x[3])
        return res

    @staticmethod
    def quartet_cost(xq: List[torch.Tensor], zq: List[torch.Tensor], distf_hd, distf_ld) -> torch.Tensor:
        """Quartet cost

        Computes a quartet cost value as the average quartet dissimilarity for all 6 pairwise distances within it.

        Scales to multiple quartets.

        Args:
            xq (List[torch.Tensor]): Input points of indices 1-4 in their respective quartets.
            zq (List[torch.Tensor]): Embeddings of `x`.
            distf_hd: Distance function to use in input space (static method of `MDSLoss`).
            distf_ld: Distance function to use in embedding space (static method of `MDSLoss`).

        Returns:
            torch.Tensor: Cost.
        """
        xd = MDSLoss.quartet_norm_factor(x=xq, distf=distf_hd)
        zd = MDSLoss.quartet_norm_factor(x=zq, distf=distf_ld)
        def d(i, j):
            return MDSLoss.quartet_dissimilarity(x=xq, z=zq, i=i, j=j, xd=xd, zd=zd, distf_hd=distf_hd, distf_ld=distf_ld)
        res = torch.mean(d(0,1)+d(1,2)+d(2,3)+d(0,2)+d(0,3)+d(1,3))
        return res

    def __call__(self, x: torch.Tensor, z: torch.Tensor, distf_hd: str = 'euclidean', distf_ld: str = 'euclidean', n_sampling: int = 1) -> torch.Tensor:
        """Stochastic-MDS loss

        Penalises change in relative positions of points within randomly sampled quartets (groups of 4).
        This assumes that inputs (`x` and `z`) are shuffled.

        The quartets are, by default, sampled multiple times to prevent overfitting.
        We use Euclidean distances in both ambient space and embedding by default.
        If embedding very high-dimensional data, the expected inputs `x` would come from a reasonable (100-dimensional) PCA reduction thereof.
        However, cosine distances can be used as an alternative.
        This can be helpful for instance when training directly on extremely high-dimensional and/or sparse data.

        A scaling step is included to account for potential variable batch size (if a data loader is used with `drop_last` set to `False`), since the final smaller batch will be less informative.
        The scale of the final value is invariant to
        - chosen batch size,
        - number of re-samplings,
        - group size (which is hard-coded to 4 in this quartet implementation).

        Args:
            x (torch.Tensor): Input.
            z (torch.Tensor): Encoding.
            distf_hd (str, optional): Distance function to be used by MDS loss for input space. Either 'euclidean' or 'cosine'. Defaults to 'euclidean'.
            distf_ld (str, optional): Distance function to be used by MDS loss for embedding space. Either 'euclidean' or 'cosine'. Defaults to 'euclidean'.
            n_sampling (int, optional): How many times groups are (re-)sampled. Defaults to 1.
        Returns:
            torch.Tensor: Loss.
        """
        
        ## Determine batch size and quartet count
        n = x.shape[0]
        nq = n//4
        
        ## Resolve distance functions
        if distf_hd=='euclidean':
            distf_hd = MDSLoss.euclidean_distance
        elif distf_hd=='cosine':
            distf_hd = MDSLoss.cosine_distance
        else:
            raise ValueError('Invalid distance function specification for MDS loss')
        if distf_ld=='euclidean':
            distf_ld = MDSLoss.euclidean_distance
        elif distf_ld=='cosine':
            distf_ld = MDSLoss.cosine_distance
        else:
            raise ValueError('Invalid distance function specification for MDS loss')

        ## Initialise loss
        loss = torch.FloatTensor([0.])

        ## Iterate over sampling repeats
        for _ in range(n_sampling):
        
            ## Divide data into quartets
            idcs = [0, nq, nq*2, nq*3]
            xq = [x[np.arange(idx, idx+nq)] for idx in idcs] # input space
            zq = [z[np.arange(idx, idx+nq)] for idx in idcs] # embedding space

            ## Account for variable batch size
            res = torch.multiply(MDSLoss.quartet_cost(xq=xq, zq=zq, distf_hd=distf_hd, distf_ld=distf_ld), n)
            res = torch.divide(res, nq)

            loss += res
        
        ## Average across sampling repeats and batch size
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