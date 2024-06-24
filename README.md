# ViVAE

ViVAE is a toolkit for single-cell data denoising and dimensionality reduction.

It is published together with **[ViScore](https://github.com/saeyslab/ViScore)**, a collection of tools for evaluation of dimensionality reduction.
Our [pre-print](https://www.biorxiv.org/content/10.1101/2023.11.23.568428v2) uses an [older version](https://github.com/saeyslab/ViVAE_old) of ViVAE.
We are heavily re-working the pre-print right now!

## Why use ViVAE

* ViVAE strikes a favourable balance between local and global structure preservation.
    * This is especially good for data with trajectories, outlier populations or suspected batch effects.
* ViVAE implements encoder indicatrices: a tool to measure local distortions of latent space.
* ViVAE integrates with [FlowSOM](https://github.com/saeyslab/FlowSOM_Python) for visualisation.
* ViVAE is a parametric model, enabling transfer learning and embedding of new points.

## Installation

ViVAE is a Python package based on PyTorch.
We recommend creating a new Anaconda environment for it.

On Linux or macOS, use the command line for installation.
On Windows, use Anaconda Prompt.

```
conda create --name ViVAE python=3.11.7 \
    numpy==1.26.3 numba==0.59.0 pandas==2.2.0 matplotlib==3.8.2 scipy==1.12.0 pynndescent==0.5.11 scikit-learn==1.4.0 scanpy==1.9.8 pytorch==2.1.2
conda activate ViVAE
pip install git+https://github.com/saeyslab/FlowSOM_Python.git@80529c6b7a1747e8e71042102ac8762c3bfbaa1b
pip install --upgrade git+https://github.com/saeyslab/ViVAE.git
```

### GPU verification

GPU acceleration is recommended if available.
To verify whether PyTorch can use CUDA, activate your ViVAE environment and type:

```
python -c "import torch; print(torch.cuda.is_available())"
```

Alternatively, to verify whether PyTorch can use Metal (on AMD/Apple Silicon Macs):

```
python -c "import torch; print(torch.backends.mps.is_available())"
```

This will print either `True` or `False`.

## Tutorials

We provide tutorials on using ViVAE with cytometry data ([here](https://github.com/saeyslab/ViVAE/blob/main/example_cytometry.ipynb)) and with scRNA-seq data ([here](https://colab.research.google.com/drive/1eNpgH_TzbCSu-_4ZPmK7tk6It4BYK5sh?usp=sharing)).

In these tutorials we cover

* import of input files
* standard pre-processing
* dimensionality reduction and its hyperparameters
* integration of ViVAE with FlowSOM
* encoder indicatrices for distortion detection
* evaluation of structure preservation with [ViScore](https://github.com/saeyslab/ViScore)
* evaluation of distortions of annotated populations with ViScore
* saving and loading trained ViVAE models

## Benchmarking

We benchmark ViVAE in terms of local and global structure preservation, using [ViScore](https://github.com/saeyslab/ViScore).
The [ViScore](https://github.com/saeyslab/ViScore) repository contains our documented [benchmarking set-up](https://github.com/saeyslab/ViScore/blob/main/benchmarking), which can be extended to other datasets and DR methods.

## Pre-print

The pre-print of our publication is available [here](https://www.biorxiv.org/content/10.1101/2023.11.23.568428v2) on bioRxiv.

It describes underlying methodology of ViVAE and ViScore, reviews past work in dimensionality reduction and evaluation of it and links to publicly available datasets on which performance of ViVAE was evaluated.
**We are heavily revising this pre-print.**

The pre-print currently uses the old version of *ViVAE*, available [here](https://github.com/saeyslab/ViVAE_old).
