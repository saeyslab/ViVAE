"""
Jacobian matrix computation adopted from:

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

from typing import Callable, Union, Optional

import math
import numpy as np

from scipy.spatial import Delaunay

import torch
import torch.func
from torch.utils.data import DataLoader

from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon

def jacobian(f: Callable, x: torch.Tensor) -> torch.Tensor:
    """Compute Jacobian matrices

    Args:
        f (Callable): Immersion function.
        x (torch.Tensor): Input to `f`.

    Returns:
        torch.Tensor: Jacobian matrices.
    """
    try:
        jac = torch.func.vmap(torch.func.jacfwd(f), in_dims=(0,))(x)
    except NotImplementedError:
        jac = torch.func.vmap(torch.func.jacrev(f), in_dims=(0,))(x)
    return jac

def metric_tensor(jac: torch.Tensor, rev: bool = False) -> torch.Tensor:
    """Compute metric tensor from Jacobian

    Args:
        jac (torch.Tensor): Jacobian matrix.
        rev (bool, optional): Swap terms of the equation to work with the encoder (submersion) case.

    Returns:
        torch.Tensor: metric
    """
    if not rev:
        res = torch.matmul(torch.transpose(jac, 1, 2), jac)
    else:
        res = torch.matmul(jac, torch.transpose(jac, 1, 2))
    return res

class EncoderIndicatome():
    """Set of encoder indicatrices"""

    def __init__(self, model, x: Union[np.ndarray, torch.Tensor]):
        """Initialise encoder indicatome

        Args:
            model (Autoencoder): Trained Autoencoder object.
            x (Union[np.ndarray, torch.Tensor]): Input data to `model`.
        """
        self.model = model
        if isinstance(x, np.ndarray):
            x = torch.FloatTensor(x)
        self.x = x
        self.act = None
        self.stepsize = None
        self.zr = None
        self.zp = None
        self.r = None

    @staticmethod
    def gridpoint_idcs(ref: torch.Tensor, xgrid: torch.Tensor, ygrid: torch.Tensor, padding: float = 0.1, minpts: Optional[int] = None) -> np.ndarray:
        """Find points on 2-d grid

        Finds points in reference 2-d data closest to each point on a grid over the data.
        Some points on the grid may remain unmatched.

        Args:
            ref (torch.Tensor): Reference dataset coordinates.
            xgrid (torch.Tensor): Grid x-coordinates.
            ygrid (torch.Tensor): Grid y-coordinates.
            padding (float, optional): Part grid tile around point that gets excluded from search. Deaults to 0.1.
            minpts (int, optional): Minimum number of points per tile for the tile not to be excluded. Defaults to None.

        Returns:
            np.ndarray: Indices of points.
        """
        idcs = []
        xstep = xgrid[1]-xgrid[0]
        ystep = ygrid[1]-ygrid[0]
        for i in range(xgrid.shape[0]):
            for j in range(ygrid.shape[0]):

                xlim, ylim = [xgrid[i]-(xstep/2*(1-padding)), xgrid[i]+(xstep/2*(1-padding))], [ygrid[j]-(ystep/2*(1-padding)), ygrid[j]+(ystep/2*(1-padding))]
                
                xpool = torch.bitwise_and(ref[:,0]>xlim[0], ref[:,0]<=xlim[1])
                ypool = torch.bitwise_and(ref[:,1]>ylim[0], ref[:,1]<=ylim[1])
                pool = torch.where(np.bitwise_and(xpool, ypool))[0]

                if pool.shape[0]>0:
                    if minpts is not None and pool.shape[0]<minpts:
                        idx = None
                    else:
                        center = torch.Tensor([(xlim[0]+xlim[1])/2, (ylim[0]+ylim[1])/2])
                        idx = torch.argmin(torch.sum((center-ref[pool])**2, axis=1)) # nearest                    
                        idcs.append(pool[idx])
                    
        return np.array(idcs)

    @staticmethod
    def circle_on_2manifold(origin: torch.Tensor, t1: torch.Tensor = torch.Tensor([1., 0.]), t2: torch.Tensor = torch.Tensor([0., 1.]), r: float = 0.1, n: int = 50) -> torch.Tensor:
        """Circle on 2-manifold

        Constructs a circle of radius `r` on 2-manifold embedded in (possibly higher-dimensional) ambient space.
        (More accurately, this is a 2-dimensional circular disc, if the ambient space is higher-dimensional.)
        Samples `n` points from the circumference to approximate it as a polygon.
        Tangent vectors `t1` and `t2` define the 2-manifold.

        Args:
            origin (torch.Tensor): Center of circle.
            t1 (torch.Tensor, optional): First tangent vector. Defaults to (1,0).
            t2 (torch.Tensor, optional): Second tangent vector. Defaults to (0,1).
            r (float, optional): Radius. Defaults to 1e-15.
            n (int, optional): Number of sampled points on circumference of circle. Defaults to 50.

        Returns:
            torch.Tensor: `n` points from the circumference of the circle.
        """

        t1_norm = t1/torch.linalg.norm(t1)
        t2_norm = t2/torch.linalg.norm(t2)
        if torch.dot(t1_norm, t2_norm)>1e-6:
            raise ValueError('Vectors `t1` and `t2` must be orthogonal.')
        angles = torch.arange(0, 1, step=1/n)*2*torch.pi
        a = np.newaxis
        points = origin+r*np.cos(angles[:,a])*t1_norm[a,:]+r*np.sin(angles[:,a])*t2_norm[a,:]
        return points

    @staticmethod
    def scale_ellipse(circum: torch.Tensor, origin: torch.Tensor, factor: float = 10.) -> torch.Tensor:
        """Scale a 2-d ellipse

        Args:
            circum (torch.Tensor): Points from circumference of ellipse.
            origin (torch.Tensor): Origin (center) of ellipse.
            factor (float, optional): _description_. Factor by which to scale. Defaults to 10..

        Returns:
            torch.Tensor: Corresponding points from circumference of re-scaled ellipse.
        """
        vecs = circum-origin
        vecs *= factor
        res = origin+vecs
        return res

    def compute_indicatrices(self, r: float = 1e-3, batch_size: int = 256, n_steps: int = 20, all_points: bool = False, n_polygon: int = 100):
        """Compute encoder indicatrices

        Args:
            r (float, optional): Hypersphere radius in ambient space. Defaults ot 1e-3.
            batch_size (int, optional): Batch size for forward pass of `self.x` through `self.model`. Defaults to 256.
            n_steps (int, optional): Number of steps along each axis of latent representation to generate grid of indicatrices. Defaults to 20.
            all_points (bool, optional): Overrides `n_steps` and uses all points, instead of grid-sampling. Can be computationally intensive. Defaults to False.
            n_polygon (int, optional): Number of points in each polygon approximating the unit sphere in pullback metric. Defaults to 100.
        """

        self.r = r

        ## Get latent activations
        act = None
        loader = DataLoader(self.x, batch_size=batch_size, shuffle=False)
        for batch in loader:
            if isinstance(batch, list):
                batch = batch[0]
            if self.model.variational:
                batch_act,_,_ = self.model.encode(batch)
            else:
                batch_act = self.model.encode(batch)
            act = batch_act if act is None else torch.vstack((act, batch_act))
        act = act.detach().cpu()
        self.act = act

        ## Extract gridpoints from embedding
        xmin, xmax = torch.min(self.act[:,0]).item(), torch.max(self.act[:,0]).item()
        ymin, ymax = torch.min(self.act[:,1]).item(), torch.max(self.act[:,1]).item()
        if not all_points:
            nsteps_x = n_steps
            nsteps_y = math.ceil((ymax-ymin)/(xmax-xmin)*nsteps_x)
            stepsize_x = (xmax-xmin)/(nsteps_x)
            stepsize_y = (ymax-ymin)/(nsteps_y)
            stepsize = min(stepsize_x, stepsize_y)
            self.stepsize = stepsize
            xs = torch.linspace(xmin, xmax, steps=nsteps_x)
            ys = torch.linspace(ymin, ymax, steps=nsteps_y)
            idcs = EncoderIndicatome.gridpoint_idcs(ref=self.act, xgrid=xs, ygrid=ys)
        else:
            idcs = np.arange(0, self.act.shape[0], 1)

        ## Trace back to ambient space
        xr = self.x[idcs]
        zr = self.model.submersion(xr)
        self.zr = zr

        ## Compute Jacobians
        jac = torch.func.vmap(torch.func.jacfwd(self.model.submersion), in_dims=(0,))(xr)

        ## Find horizontal tangents
        _, _, v = torch.svd(jac, some=False)
        htv = v[:, :2]

        ## Create hyperspheres
        xp = []
        for i in range(xr.shape[0]):
            origin = xr[i]
            t1 = htv[i,0,:]
            t2 = htv[i,1,:]
            circle = EncoderIndicatome.circle_on_2manifold(origin, t1, t2, r, n_polygon)
            xp.append(circle)

        ## Submerge hyperspheres
        zp = [self.model.submersion(this_xp) for this_xp in xp]
        self.zp = zp

    def get_embedding(self) -> np.ndarray:
        """Get latent space embedding

        Returns:
            np.ndarray: Embedding coordinates.
        """
        return self.act

    def get_polygons(self, scale_factor: float = 1e-2) -> PatchCollection:
        """Get polygonal approximations of indicatrices
        Args:
            scale_factor (float, optional): Scaling factor for polygons. Defaults to 1e-2.
        
        Returns:
            matplotlib.collections.PatchCollection: Collection of polygons.
        """
        zp_scaled = []
        for i in range(self.zr.shape[0]):
            res = EncoderIndicatome.scale_ellipse(self.zp[i], self.zr[i], scale_factor)
            zp_scaled.append(res)
        polygons = [Polygon(tuple(vec.tolist()), closed=True) for vec in zp_scaled]
        p = PatchCollection(polygons)
        return p

class DecoderIndicatome():
    """Set of decoder indicatrices"""

    def __init__(self, model, x: Union[np.ndarray, torch.Tensor]):
        """Initialise decoder indicatome

        Args:
            model (Autoencoder): Trained Autoencoder object.
            x (Union[np.ndarray, torch.Tensor]): Input data to `model`.
        """
        self.model = model
        if isinstance(x, np.ndarray):
            x = torch.FloatTensor(x)
        self.x = x
        self.act = None
        self.patches = None
        self.coords = None
        self.stepsize = None

    def compute_indicatrices(self, batch_size: int = 256, n_steps: int = 20, n_polygon: int = 100):
        """Compute decoder indicatrices

        Args:
            batch_size (int, optional): Batch size for forward pass of `self.x` through `self.model`. Defaults to 256.
            n_steps (int, optional): Number of steps along each axis of latent representation to generate grid of indicatrices. Defaults to 20.
            n_polygon (int, optional): Number of points in each polygon approximating the unit sphere in pullback metric. Defaults to 100.
        """
        ## Get latent activations
        act = None
        loader = DataLoader(self.x, batch_size=batch_size, shuffle=False)
        for batch in loader:
            if isinstance(batch, list):
                batch = batch[0]
            if self.model.variational:
                batch_act,_,_ = self.model.encode(batch)
            else:
                batch_act = self.model.encode(batch)
            act = batch_act if act is None else torch.vstack((act, batch_act))
        act = act.detach().cpu()
        self.act = act

        ## Get coordinates of grid points
        xmin, xmax = torch.min(self.act[:,0]).item(), torch.max(self.act[:,0]).item()
        ymin, ymax = torch.min(self.act[:,1]).item(), torch.max(self.act[:,1]).item()
        nsteps_x = n_steps
        nsteps_y = math.ceil((ymax-ymin)/(xmax-xmin)*nsteps_x)
        stepsize_x = (xmax-xmin)/(nsteps_x)
        stepsize_y = (ymax-ymin)/(nsteps_y)
        stepsize = min(stepsize_x, stepsize_y)
        self.stepsize = stepsize

        xs = torch.linspace(xmin, xmax, steps=nsteps_x)
        ys = torch.linspace(ymin, ymax, steps=nsteps_y)

        mg = torch.meshgrid([xs, ys], indexing='ij')
        coords = torch.vstack([torch.flatten(mg[0]), torch.flatten(mg[1])]).T

        hull = Delaunay(self.act)
        coords = coords[hull.find_simplex(coords)>=0]
        self.coords = coords

        ## Get vector patches at grid points
        phi = torch.linspace(0., 2*np.pi, n_polygon)
        raw_vecs = torch.stack((torch.sin(phi), torch.cos(phi)))
        metric = metric_tensor(jacobian(self.model.decoder, coords)) # pullback metric tensor

        ## Normalise vectors in pullback metric
        norm_vecs = torch.sqrt(torch.einsum('mn,imn->in', raw_vecs, torch.einsum('ijk,kl->ijl', metric, raw_vecs)))
        norm_vecs = norm_vecs.unsqueeze(2).expand(*norm_vecs.shape, raw_vecs.shape[0])

        ## Reshape raw vectors
        raw_vecs = raw_vecs.unsqueeze(2).expand(*raw_vecs.shape, coords.shape[0])
        raw_vecs = torch.transpose(raw_vecs, dim0=0, dim1=2)

        ## Normalise vector patches
        self.patches = torch.where(norm_vecs!=0, raw_vecs/norm_vecs, torch.zeros_like(raw_vecs))
    
    def get_embedding(self) -> np.ndarray:
        """Get latent space embedding

        Returns:
            np.ndarray: Embedding coordinates.
        """
        return self.act

    def get_polygons(self, scale_factor: float = 1e-2) -> PatchCollection:
        """Get polygonal approximations of indicatrices
        Args:
            scale_factor (float, optional): Scaling factor for polygons. Defaults to 1e-2.
        
        Returns:
            matplotlib.collections.PatchCollection: Collection of polygons.
        """
        if self.patches is None:
            raise ValueError('Compute indicatrices first')
        
        ## Scale patches for plotting
        vec_norms = torch.linalg.norm(self.patches.reshape(-1,2), dim=1)
        min_vec_norm = torch.min(vec_norms[torch.nonzero(vec_norms)])
        normed_patches = self.patches/min_vec_norm*self.stepsize*scale_factor

        ## Anchor patches to grid
        anchored_patches = self.coords.unsqueeze(1).expand(*normed_patches.shape) + normed_patches

        ## Create polygons
        polygons = [Polygon(tuple(vec.tolist()), closed=True) for vec in anchored_patches]
        p = PatchCollection(polygons)
        p.set_color([0 / 255, 0 / 255, 0 / 255, 0.3])

        return p