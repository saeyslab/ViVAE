<img src="./logo.png" alt="ViVAE" width="250"/>

ViVAE (*vee-vay*) is a toolkit for single-cell data denoising and dimensionality reduction.

It is published together with **[ViScore](https://github.com/saeyslab/ViScore)**, a framework for fair and scalable evaluation of dimensionality reduction.
Check out the associated [paper](https://www.biorxiv.org/content/10.1101/2023.11.23.568428v3): *Interpretable models for scRNA-seq data embedding with multi-scale structure preservation*, where we describe and validate our methods in-depth.

<img src="./overview.png" alt="overview" width="900"/>

## Why use ViVAE

* ViVAE achieves state-of-the-art multi-scale structure preservation.
    * This is especially, but not exclusively, suitable for data with trajectories, outlier populations or suspected batch effects.
* Our embedding model implements encoder indicatrices: a tool to measure local distortions of latent space.
* We integrate ViVAE with [FlowSOM](https://github.com/saeyslab/FlowSOM_Python) for visualisation.
* The ViVAE model is parametric, enabling transfer learning and embedding of new points.
* ViVAE can take advantage of modern GPU architectures, especially for training on large datasets.

## Setting up

With most datasets, ViVAE can be run on a consumer laptop with or without GPU acceleration.

To try out ViVAE without installing, check out the tutorial in `tutorials/example_scrnaseq.ipynb` with directions for use in [Google Colab](https://colab.research.google.com).

<details>
<summary><b>Python installation</b></summary>
<br>

ViVAE is a Python package based on PyTorch.
We recommend creating a new Anaconda environment for it.

On Linux or macOS, use the command line for installation.
On Windows, use Anaconda Prompt.

```bash
conda create --name ViVAE python=3.11.9
conda activate ViVAE
pip install --upgrade git+https://github.com/saeyslab/ViVAE.git
```

For FlowSOM integration, also run

```bash
pip install git+https://github.com/saeyslab/FlowSOM_Python
```

Then you can import *ViVAE* in Python.
Check out our tutorials (described below) or the relevant documentation (see the following code snippet) to get started.

```python
## In Python:
import vivae as vv

help(vv.ViVAE) # documentation of the ViVAE model
```

<hr>
</details>

<details>
<summary><b>R installation</b></summary>
<br>

We are working on an R implementation of ViVAE that is independent of PyTorch.
In the meantime, to install and run ViVAE in R using [reticulate](https://rstudio.github.io/reticulate/), use our R vignette (`tutorials/example_r.Rmd`) (an RMarkdown file you can open in RStudio).

<hr>
</details>

<details>
<summary><b>Using GPU acceleration</b></summary>
<br>

While ViVAE runs well on CPU, the model can take advantage of GPU acceleration via CUDA or MPS.
By default, ViVAE attempts to use CUDA if available, but not MPS.
To enable or disable the CUDA or MPS backend for ViVAE, modify the environment variables `VIVAE_CUDA` and `VIVAE_MPS`, respectively, before importing ViVAE:

```python
## In Python:
import os
os.environ['VIVAE_CUDA'] = '1' # enable
os.environ['VIVAE_MPS'] = '0' # disable
import vivae as vv
```

Set both to `'0'` to use CPU.
In our tests, using MPS for acceleration only seems to make sense with larger training batch sizes.

<hr>
</details>

<details>
<summary><b>Reproducibility</b></summary>
<br>

To facilitate the reproducibility of results within a fixed workflow, ViVAE enables [PyTorch determinism](https://pytorch.org/docs/stable/notes/randomness.html) by default.
This is not a guarantee of full reproducibility across different hardware and software set-ups, but it increases the chance of getting the same result across runs.

Please beware, ViVAE is likely to run slower when deterministic.
You can enable the use of non-deterministic operations by setting the `'VIVAE_DETERMINISTIC'` environment variable to `'0'` before importing ViVAE:

```python
## In Python:
import os
os.environ['VIVAE_DETERMINISTIC'] = '0' # disable
import vivae as vv
```

To maximise reproducibility of results in the tutorials and case studies provided in this repo, please use the CPU backend (see the section above) with determinism enabled.

We do not attempt reproducibility between an MPS-accelerated model and a CPU-/CUDA-accelerated one.
This is because, at least for the time being, we need to use a custom data loader class internally to make ViVAE work with MPS.

See also the *Stability* section below, which speaks to the reduction of randomness the ViVAE model.

</details>

## Tutorials

Our tutorials will help you start using ViVAE quickly, be it with scRNA-seq or cytometry data.
The tutorials include data pre-processing, discuss the most important hyperparameters of ViVAE and touch on evaluation of embeddings using [ViScore](https://github.com/saeyslab/ViScore).

<details>
<summary><b>Using ViVAE with scRNA-seq data</b></summary>
<br>

ViVAE was primarily designed for, and tested with, single-cell transcriptomic datasets.

To get you started, we provide an example workflow for analysis of bone marrow single-cell transcriptomic data with ViVAE.
We evaluate the separation of distincts immune cell lineages and general structure preservation by ViVAE, t-SNE and UMAP.

Additionally, we compute embedding errors by population and demonstrate the use of neighbourhood composition plots for explaining sources of error.

Advantages and potential pitfalls of smooth embeddings are described and discussed.

The tutorial is provided as a Jupyter notebook (`tutorials/example_scrnaseq.ipynb`).

<hr>
</details>

<details>
<summary><b>Using ViVAE with cytometry data</b></summary>
<br>

ViVAE, while intended mainly for scRNA-seq data, is straightforward to use with flow and mass cytometry data as well.

Its structure-preserving properties are especially advantageous if global structures are of interest.
Additionally, ViVAE integrates with FlowSOM to provide a graph-based view of cytometry datasets.

We provide a Jupyter notebook tutorial (`tutorials/example_cytometry.ipynb`) that covers importing and pre-processing of data, denoising, dimensionality reduction and some evaluation of the resulting embedding.

Our R installation vignette (`tutorials/example_r.Rmd`) shows how to use ViVAE denoising and dimensionality reduction from R.

<hr>
</details>

We also showcase some experimental modifications of the model that will mostly be interesting for developers of dimensionality reduction algorithms.

## Case studies

The associated manuscript presents case studies on various single-cell datasets.
These case studies are replicated using Jupyter notebooks in the `case_studies` directory.

<details>
<summary><b>Breast immune cells transcriptome study (Reed)</b></summary>
<br>

`case_study_reed.ipynb` provides code to reproduce the [*Reed*](https://cellxgene.cziscience.com/collections/48259aa8-f168-4bf5-b797-af8e88da6637) dataset case study from our paper.
This dataset comes from the Human Breast Cell Atlas.
The authors provide labels for various leukocyte populations.

We compare ViVAE with *t*-SNE and UMAP and describe embedding errors per cell population using the Extended Neighbourhood-Proportion-Error (xNPE) and neighbourhood composition plots.

<hr>
</details>

<details>
<summary><b>Developing zebrafish embryos transcriptome study (Farrell)</b></summary>
<br>

`case_study_farrell.ipynb` provides code to reproduce the [*Farrell*](https://singlecell.broadinstitute.org/single_cell/study/SCP162/single-cell-reconstruction-of-developmental-trajectories-during-zebrafish-embryogenesis) dataset case study from our paper.
This dataset contains cells from multiple stages of zebrafish embryo development.
The authors provide labels of distinct cell lineages.

We compare *t*-SNE, UMAP, a vanilla VAE, default ViVAE and ViVAE-EncoderOnly (a decoder-less model that implements parametric stochastic MDS with GPU acceleration).
The analysis in our paper focuses on the differences between neighbour-embedding algorithms (which tend to form separate clusters) and multi-dimensional scaling algorithms (which produce more continuous represerntations).
We use encoder indicatrices to describe different manners of latent space distortion by the three VAE-based models.

<hr>
</details>

<details>
<summary><b>Mouse bone marrow CyTOF dataset study (Samusik)</b></summary>
<br>

`case_study_samusik.ipynb` provides code to reproduce the [*Samusik*](https://pubmed.ncbi.nlm.nih.gov/27183440/) dataset case study from our paper.
This is a popular reference dataset for showcasing dimensionality reduction and clustering tools.
The authors provide labels for various immune cell populations.

We use ViVAE to create a nice embedding of the data, then use FlowSOM for clustering (independent of the dimension reduction) and show a plot of the embedding with the FlowSOM minimum spanning tree (MST) superimposed.

To explore more options for evaluating cytometry data embeddings and integrating FlowSOM for informative visualisation, we refer you to the cytometry analysis tutorial in `tutorials/example_cytometry.ipynb`.

</details>

## Stability

The ViVAE model is based on a variational autoencoder (VAE).
This is a type of neural network with a comparatively high number of parameters, which are trained based on input data.

It is known that the final parameters of a trained VAE can differ based on the initialisation of the network.
As a consequence, we might see small differences between embeddings across runs if the model converges to a slightly different local optimum.
There are multiple ways to address this and end up with a more stable model, beyond basic hyperparameter tuning (suitable learning rate, batch size, number of epochs).

First, we can use determinism (as described above in *Setting up*) to reduce the amount of stochastic operations used in model training.

Second, we propose the use of PCA initialisation of model, which has been used before with *t*-SNE, *UMAP* and other embedding algorithms.
Here, we propose the use of *imitation loss*, which ensures that the VAE network first learns a parametric mapping of the data that imitates a deterministic PCA embedding, and only then optimises the ViVAE loss function.

See `tutorials/imitation.ipynb` to learn how to stabilise your ViVAE model.

## Using cosine distances

Many domain experts differ in how they approach to problem of quantifying dissimilarities between high-dimensional data points.
Our work relies heavily on the notion of neighbourhood ranks based on Euclidean distances.
However, if you want to use cosine distances in the training your ViVAE model, we make that possible.

See `tutorials/cosine.ipynb` to learn how to do so.

## Evaluation framework

In our paper, we compare ViVAE and other DR methods in terms of local and global structure preservation in terms of neighbourhoods, rather than distances.
To this end, we created [ViScore](https://github.com/saeyslab/ViScore).
The ViScore repository contains our documented [benchmarking set-up](https://github.com/saeyslab/ViScore/blob/main/benchmarking), which can be extended to other datasets and DR methods.
This set-up includes full documentation to guide the user through the process of benchmarking or hyperparameter tuning on an HPC cluster from start to finish.

<img src="https://github.com/saeyslab/ViScore/blob/main/benchmarking/schematic.png?raw=true" />