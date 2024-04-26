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

from typing import Optional, Union, List, Tuple

import numpy as np

import torch

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon

from .geometry import EncoderIndicatome, DecoderIndicatome

palette = [
            '#000000', '#1CE6FF', '#FF34FF', '#FF4A46', '#008941', '#006FA6', '#A30059',
            '#7A4900', '#0000A6', '#63FFAC', '#B79762', '#004D43', '#8FB0FF', '#997D87',
            '#5A0007', '#809693', '#1B4400', '#4FC601', '#3B5DFF', '#4A3B53', '#FF2F80',
            '#61615A', '#BA0900', '#6B7900', '#00C2A0', '#FFAA92', '#FF90C9', '#B903AA', '#D16100',
            '#DDEFFF', '#000035', '#7B4F4B', '#A1C299', '#300018', '#0AA6D8', '#013349', '#00846F',
            '#372101', '#FFB500', '#C2FFED', '#A079BF', '#CC0744', '#C0B9B2', '#C2FF99', '#001E09',
            '#00489C', '#6F0062', '#0CBD66', '#EEC3FF', '#456D75', '#B77B68', '#7A87A1', '#788D66',
            '#885578', '#FAD09F', '#FF8A9A', '#D157A0', '#BEC459', '#456648', '#0086ED', '#886F4C',
            '#34362D', '#B4A8BD', '#00A6AA', '#452C2C', '#636375', '#A3C8C9', '#FF913F', '#938A81',
            '#575329', '#00FECF', '#B05B6F', '#8CD0FF', '#3B9700', '#04F757', '#C8A1A1', '#1E6E00',
            '#7900D7', '#A77500', '#6367A9', '#A05837', '#6B002C', '#772600', '#D790FF', '#9B9700',
            '#549E79', '#FFF69F', '#201625', '#72418F', '#BC23FF', '#99ADC0', '#3A2465', '#922329',
            '#5B4534', '#FDE8DC', '#404E55', '#0089A3', '#CB7E98', '#A4E804', '#324E72', '#6A3A4C'
        ]

def plot_embedding(
        embedding: np.ndarray,
        labels: Optional[np.ndarray] = None,
        unassigned: Optional[Union[str, List[str]]] = None,
        values: Optional[np.ndarray] = None,
        s: float = 0.05,
        palette: List = palette,
        figsize: Tuple[int, int] = (5,5)
):
    """Plot 2-d embedding

    Args:
        embedding (np.ndarray): Coordinates.
        labels (np.ndarray, optional): Pointwise labels (alternative to `values`). Defaults to None.
        unassigned (str or List[str], optional): Value(s) in `labels` given to unlabelled points (plotted grey, in background). Defaults to None.
        values (np.ndarray, optional): Pointwise values (alternative to `labels`). Defaults to None.
        s (float, optional): Point size. Defaults to 0.05.
        palette (List, optional): Custom hex-code colour palette for `labels`.
        figsize (Tuple, optional): Size of figure to display. Defaults to (3,3).
    """
    fig, ax = plt.subplots(figsize=figsize)

    if labels is not None:
        ## Plot overlay of population labels
        if unassigned is not None:
            ## Plot unassigned points
            if not isinstance(unassigned, list):
                unassigned = [unassigned]
            for upop in np.array(unassigned):
                idcs = np.where(labels == upop)[0]
                if len(idcs)>0:
                    ax.scatter(embedding[idcs,0], embedding[idcs,1], s=s, c='#bfbfbf')
                    embedding = np.delete(embedding, idcs, axis=0)
                    labels = np.delete(labels, idcs)
        idx_pop = 0
        for pop in np.unique(labels):
            ## Plot assigned points
            idcs = np.where(labels == pop)[0]
            ax.scatter(embedding[idcs,0], embedding[idcs,1], label=pop, s=s, c=palette[idx_pop])
            idx_pop += 1
        l = plt.legend(bbox_to_anchor=(1.05, 1.05), loc='upper left', markerscale=30)
        ax.set_facecolor('white')
    elif values is not None:
        ## Plot values of continuous quantity
        ax.scatter(embedding[:,0], embedding[:,1], c=values, s=s, cmap='viridis')
        sm = mpl.cm.ScalarMappable(cmap='viridis')
        sm.set_array(values)
        cax = plt.axes([0.92, 0.1, 0.02, 0.8])
        _ = plt.colorbar(sm, cax=cax)
    fig.patch.set_facecolor('white')
    plt.show()

def plot_indicatrices(
        indicatrices: Union[EncoderIndicatome,DecoderIndicatome],
        scale_factor: Optional[float] = None,
        s: float = 0.05,
        figsize: Tuple[int, int] = (5,5),
        **kwargs
    ):
    """Plot encoder/decoder indicatrices

    Plots results of `encoder_indicatrices` or `decoder_indicatrices`.

    Args:
        indicatrices (Union[EncoderIndicatome,DecoderIndicatome]): Set of indicatrices.
        scale_factor (float, optional): Scaling factor for the polygons. Defaults to 1e-2 for decoder and radius^(-1) for encoder.
        s (float, optional): Point size. Defaults to 0.05.
        figsize (Tuple[int, int], optional): Figure size. Defaults to (5,5).
        **kwargs: Keywords arguments to `matplotlib.pyplot.scatter`.
    """
    if scale_factor is None:
        if isinstance(indicatrices, EncoderIndicatome):
            scale_factor = 1/indicatrices.r
        elif isinstance(indicatrices, DecoderIndicatome):
            scale_factor = 1e-2

    emb = indicatrices.get_embedding()
    pol = indicatrices.get_polygons(scale_factor=scale_factor)

    _, ax = plt.subplots(figsize=figsize)
    ax.scatter(emb[:,0], emb[:,1], s=s, **kwargs)
    pol.set_color([0 / 255, 0 / 255, 0 / 255, 0.3])
    ax.add_collection(pol)

    plt.show()