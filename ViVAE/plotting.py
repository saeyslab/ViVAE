"""
FlowSOM tree plotting adopted from:

https://github.com/saeyslab/FlowSOM_Python

A. Couckuyt, B. Rombaut, Y. Saeys, and S. Van Gassen, “Efficient
cytometry analysis with FlowSOM in Python boosts interoperability with
other single-cell tools,” Bioinformatics, vol. 40, no. 4, p. btae179,
Apr. 2024, doi: 10.1093/bioinformatics/btae179.

S. Van Gassen et al., “FlowSOM: Using self-organizing maps for
visualization and interpretation of cytometry data,” Cytometry Part A,
vol. 87, no. 7, pp. 636–645, 2015, doi: 10.1002/cyto.a.22625.

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

from typing import Optional, Union, List, Tuple

import numpy as np

import torch

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon
import matplotlib.collections as mc
from matplotlib.patches import Circle, Wedge
import pandas as pd
import copy
import flowsom as fs

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
    equal_axis_scales: bool = True,
    fsom: Optional[fs.main.FlowSOM] = None,
    fsom_show_nodes: bool = True,
    fsom_view: Optional[str] = 'labels',
    fsom_markers: Optional[Union[str, List]] = None,
    fsom_node_scale: float = 0.003,
    fsom_edge_scale: float = 0.8,
    fsom_text_size: float = 6.,
    fsom_plot_unassigned: bool = False,
    dr_model = None,
    palette: List = palette,
    figsize: Tuple[int, int] = (6,6)
):
    """Plot 2-d embedding

    Plot a 2-dimensional embedding. Manual labels of points (`labels`) or some continuous
    point-wise value (`values`), eg. expression of a marker per biological cell, can be
    displayed using colour scale.

    If a trained FlowSOM model is available, the FlowSOM minimum spanning tree (MST) with
    can be plotted over the embedding to check for concordance between dimensionality
    reduction and FlowSOM clustering.
    
    Nodes of the tree (corresponding to clusters) can show proportions of manually
    assigned populations belonging to each cluster (set `fsom_view` to 'labels'), relative
    expression levels of one or more specified markers (specify `fsom_markers` and set
    `fsom_view` to 'markers') or cluster/metacluster numbers (set `fsom_view` to 'clusters'
    or 'metaclusters').

    Alternatively, FlowSOM tree nodes can be omitted (if `fsom_show_nodes` is False), to
    only show the FlowSOM tree topology as a hypothetical skeleton of the dataset.

    The FlowSOM tree plotting is based on code from the Python FlowSOM implementation:
    https://github.com/saeyslab/FlowSOM_Python
    A. Couckuyt, B. Rombaut, Y. Saeys, and S. Van Gassen,
        “Efficient cytometry analysis with FlowSOM in Python boosts interoperability with
        other single-cell tools,” Bioinformatics, vol. 40, no. 4, p. btae179, Apr. 2024,
        doi: 10.1093/bioinformatics/btae179.
    S. Van Gassen et al.,
        “FlowSOM: Using self-organizing maps for visualization and interpretation of
        cytometry data,” Cytometry Part A, vol. 87, no. 7, pp. 636–645, 2015,
        doi: 10.1002/cyto.a.22625.

    Args:
        embedding (np.ndarray): Coordinates.
        labels (np.ndarray, optional): Pointwise labels (alternative to `values`). Defaults to None.
        unassigned (str or List[str], optional): Value(s) in `labels` given to unlabelled points (plotted grey, in background). Defaults to None.
        values (np.ndarray, optional): Pointwise values (alternative to `labels`). Defaults to None.
        s (float, optional): Point size. Defaults to 0.05.
        equal_axis_scales (bool, optional): Whether x-axis and y-axis should use the same scale. Defaults to True.
        fsom (flowsom.main.FlowSOM, optional): FlowSOM model trained on input high-dimensional data. Defaults to None.
        fsom_view (str, optional): One of 'labels', 'markers', 'clusters', 'metaclusters' or None for FlowSOM tree view mode. Defaults to 'labels'.
        fsom_markers (str or List, optional): One or more markers (or channels or feature indices) in the FlowSOM model to plot cluster-wise intensities for, instead of population proportions. Defaults to None.
        fsom_show_nodes (bool, optional): Whether to show nodes of FlowSOM tree. Defaults to True (unless no labels or markers are specified).
        fsom_node_scale (float, optional): Scaling factor for size of FlowSOM tree nodes. Defaults to 0.003.
        fsom_edge_scale (float, optional): Scaling factor for size of FlowSOM tree edges. Defaults to 0.8
        fsom_text_size (float, optional): Text size if `fsom_view` is 'clusters' or 'metaclusters'. Defaults to 6.
        fsom_plot_unassigned (bool, optional): Whether FlowSOM tree nodes should show proportions of unassigned cells. Defaults to False.
        dr_model (optional): Dimension-reduction model with a `.transform` method that generated `embedding`. Needed if `fsom` is specified. Defaults to None.
        palette (List, optional): Custom hex-code colour palette for `labels`.
        figsize (Tuple, optional): Size of figure to display. Defaults to (6,6).
    """
    fig, ax = plt.subplots(figsize=figsize)

    draw_mst = fsom is not None and dr_model is not None
    if draw_mst:
        alpha = .35
    else:
        alpha = 1.

    if labels is not None:
        ## Plot overlay of population labels
        l = copy.deepcopy(labels)
        if unassigned is not None:
            ## Plot unassigned points
            if not isinstance(unassigned, list):
                unassigned = [unassigned]
            for upop in np.array(unassigned):
                idcs = np.where(labels == upop)[0]
                if len(idcs)>0:
                    ax.scatter(embedding[idcs,0], embedding[idcs,1], s=s, c='#bfbfbf', alpha=alpha)
                    embedding = np.delete(embedding, idcs, axis=0)
                    l = np.delete(l, idcs)
        idx_pop = 0
        for pop in np.unique(l):
            ## Plot assigned points
            idcs = np.where(l == pop)[0]
            ax.scatter(embedding[idcs,0], embedding[idcs,1], label=pop, s=s, c=palette[idx_pop], alpha=alpha)
            idx_pop += 1
        if not (draw_mst and fsom_view=='markers' and fsom_markers is not None):
            l = plt.legend(bbox_to_anchor=(1.05, 1.05), loc='upper left', markerscale=30)

            for lh in l.legend_handles: 
                lh.set_alpha(1.)
                lh._sizes = [50] 

        ax.set_facecolor('white')
    elif values is not None:
        ## Plot values of continuous quantity
        ax.scatter(embedding[:,0], embedding[:,1], c=values, s=s, cmap='viridis', alpha=alpha)
        sm = mpl.cm.ScalarMappable(cmap='viridis')
        sm.set_array(values)
        cax = plt.axes([0.92, 0.1, 0.02, 0.8])
        _ = plt.colorbar(sm, cax=cax)
    else:
        ax.scatter(embedding[:,0], embedding[:,1], c='#2d22c9', s=s, alpha=alpha)

    ## The following code has been adapted based on FlowSOM_Python
    ## https://github.com/saeyslab/FlowSOM_Python
    if draw_mst:
        ## Determine FlowSOM tree node sizes
        cluster_sizes = fsom.get_cluster_data().obs['percentages']
        cluster_empty = cluster_sizes==0.
        ranges = np.ptp(embedding, axis=0)
        maxsize = np.min(ranges)*fsom_node_scale
        node_sizes = np.sqrt(np.multiply((np.divide(cluster_sizes, np.max(cluster_sizes))), maxsize))
        node_sizes[cluster_empty] = min([0.05, node_sizes.max()])
        ## Get embedding of cluster centroids
        centroids = fsom.get_cluster_data().obsm['codes']
        layout    = dr_model.transform(centroids)
        # Add FlowSOM tree edges
        edge_list = fsom.get_cluster_data().uns['graph'].get_edgelist()
        segment_plot = [
            (layout[nodeID[0], 0],
             layout[nodeID[0], 1],
             layout[nodeID[1], 0],
             layout[nodeID[1], 1])
             for nodeID in edge_list
        ]
        edges = np.asarray(segment_plot, dtype=np.float32)
        e = [[(row[0], row[1]), (row[2], row[3])] for row in edges]
        mst = mc.LineCollection(e)
        mst.set_edgecolor('black')
        mst.set_linewidth(fsom_edge_scale)
        mst.set_zorder(0)
        ax.add_collection(mst)
        if fsom_show_nodes:
            # Add FlowSOM tree nodes
            nodes = [Circle((row[0], row[1]), node_sizes.iloc[i]) for i, row in enumerate(layout)]

            if fsom_markers is not None and not isinstance(fsom_markers, list):
                fsom_markers = list(fsom_markers)

            if fsom_view=='markers' and fsom_markers is not None and len(fsom_markers)==1:
                cmap = mpl.colormaps['viridis']
                n = mc.PatchCollection(nodes, cmap=cmap)
            else:
                n = mc.PatchCollection(nodes)
            n.set_facecolor(['#C7C7C7' if tf else '#FFFFFF' for tf in cluster_empty])
            n.set_edgecolor('black')
            n.set_linewidth(fsom_edge_scale/1.5)
            n.set_zorder(2)
            ax.add_collection(n)
            if fsom_view=='labels' and labels is not None:
                ## Set up colour palette for nodes
                pops = np.unique(labels)
                color_dict = dict(zip(pops, palette))
                if unassigned is not None:
                    for u in unassigned:
                        color_dict[u] = '#bfbfbf'
                ## Plot pie per node for labelled cell populations
                for cl in range(fsom.get_cell_data().uns['n_nodes']):
                    node_cell_types = labels[fsom.get_cell_data().obs['clustering'] == cl]
                    if not fsom_plot_unassigned and unassigned is not None:
                        node_cell_types = node_cell_types[[x not in unassigned for x in node_cell_types]]
                    if len(node_cell_types) != 0:
                        table = pd.crosstab(node_cell_types, columns='count')
                        table['part'] = np.multiply(np.divide(table['count'], sum(table['count'])), 360)
                        angles = np.asarray(np.cumsum(table['part']))
                        if 0 not in angles:
                            angles = np.insert(angles, 0, 0)
                        row = layout[cl, :]
                        patches = fs.pl._plot_helper_functions.add_wedges(
                            tuple(row), heights=np.repeat(node_sizes.iloc[cl], len(angles)), angles=angles
                        )
                        p = mc.PatchCollection(patches)
                        p.set_facecolor([color_dict.get(key) for key in table.index.values])
                        p.set_edgecolor('black')
                        p.set_linewidth(fsom_edge_scale/4.)
                        p.set_zorder(3)
                        ax.add_collection(p)
            elif fsom_view=='markers' and fsom_markers is not None:
                
                var_names = fsom.get_cell_data().var_names
                marker_table = fsom.get_cell_data().var['marker']
                
                markers = []
                for i, m in enumerate(fsom_markers):
                    if m not in var_names:
                        idx = marker_table[marker_table==m].index
                        if len(idx)!=1:
                            raise ValueError(f'Marker {m} cannot be matched to a FlowSOM channel')
                        markers.append(idx[0])
                    else:
                        markers.append(m)
                
                markers = np.asarray(markers)

                pretty_markers = fsom.get_cell_data()[:, markers].var['pretty_colnames']
                max_x, max_y = np.max(layout, axis=0)

                if len(markers)==1:
                    marker = markers[0]
                    ref_markers_bool = fsom.get_cell_data().var['cols_used']
                    ref_markers = fsom.get_cell_data().var_names[ref_markers_bool]
                    mfis = fsom.get_cluster_data().X
                    #ref_markers = list(fs.tl.get_channels(fsom, ref_markers).keys())
                    ref_markers = list(ref_markers)
                    indices_markers = (np.asarray(fsom.get_cell_data().var_names)[:, None] == ref_markers).argmax(axis=0)
                    lim = (mfis[:, indices_markers].min(), mfis[:, indices_markers].max())
                    marker_index = np.where(fsom.get_cell_data().var_names == marker)[0][0]

                    variable = mfis[:, marker_index]
                    
                    n.set_array(variable)
                    n.set_clim(lim)
                    n.set_edgecolor('black')
                    n.set_linewidth(fsom_edge_scale/4.)
                    n.set_zorder(2)
                    ax.add_collection(n)

                    ax, fig = fs.pl._plot_helper_functions.add_legend(
                        fig=fig, ax=ax, data=variable, title=fsom_markers[0], location='upper left', bbox_to_anchor=(0.5, 0.5), cmap=cmap, categorical=False
                    )

                else:
                    fig, ax = fs.pl._plot_helper_functions.plot_star_legend(
                        fig, ax, pretty_markers,
                        coords=(max_x, max_y),
                        max_star_height=np.max(node_sizes)*3.,
                        star_height=1.,
                    )
                    
                    data = fsom.get_cluster_data()[:, markers].X
                    heights = fs.pl._plot_helper_functions.scale_star_heights(data, node_sizes)
                    s = mc.PatchCollection(fs.pl._plot_helper_functions.add_stars(layout, heights))
                    s.set_array(range(data.shape[1]))
                    s.set_edgecolor('black')
                    s.set_linewidth(fsom_edge_scale/4.)
                    s.set_zorder(3)
                    ax.add_collection(s)
                    plt.axis('off')
            elif fsom_view in ['metaclusters', 'clusters']:
                if fsom_view=='clusters':
                    numbers = np.arange(1, fsom.get_cell_data().uns['n_nodes']+1)
                elif fsom_view=='metaclusters':
                    numbers = np.asarray(fsom.get_cluster_data().obs['metaclustering'], dtype=int)
                ax = fs.pl._plot_helper_functions.add_text(ax, layout, numbers, text_size=fsom_text_size, text_color='black', ha=['center'])

                
    if equal_axis_scales:
        ax.axis('equal')
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