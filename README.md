# ViVAE

*[David Novak](https://github.com/davnovak), Cyril de Bodt, Pierre Lambert, John A. Lee, Sofie Van Gassen, Yvan Saeys*

ViVAE is a toolkit for single-cell data denoising and dimensionality reduction.

**It is published together with [ViScore](https://github.com/saeyslab/ViScore), a collection of tools for evaluation of dimensionality reduction.**

Our [pre-print](https://www.biorxiv.org/content/10.1101/2023.11.23.568428v2) uses an [older version](https://github.com/saeyslab/ViVAE_old) of ViVAE.
We are heavily re-working the pre-print right now!

## Installation

ViVAE is a Python package based on PyTorch.
We recommend creating a new Anaconda environment for ViVAE.

On Linux or macOS, use the command line for installation.
On Windows, use Anaconda Prompt.

```
conda create --name ViVAE python=3.11.7 \
    numpy==1.26.3 numba==0.59.0 pandas==2.2.0 matplotlib==3.8.2 scipy==1.12.0 pynndescent==0.5.11 scikit-learn==1.4.0 scanpy==1.9.8 pytorch==2.1.2
conda activate ViVAE
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

## Example

While ViVAE is primarily tailored toward scRNA-seq data, its use extends to cytometry, particularly for trajectory inference and outlier population detection.
We provide a Jupyter notebook (`example_cytometry.ipynb`) showing an application to a mass cytometry (CyTOF) dataset.

(Look [here](https://docs.jupyter.org/en/latest/install/notebook-classic.html) if you are interested in how to use Jupyter notebooks.)

## Pre-print

The pre-print of our publication is available [here](https://www.biorxiv.org/content/10.1101/2023.11.23.568428v2) on bioRxiv.

It describes underlying methodology of ViVAE and ViScore, reviews past work in dimensionality reduction and evaluation of it and links to publicly available datasets on which performance of ViVAE was evaluated.

The pre-print uses the old version of *ViVAE*, available [here](https://github.com/saeyslab/ViVAE_old).