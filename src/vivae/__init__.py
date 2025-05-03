"""ViVAE: Single-cell dimensionality reduction toolkit"""

import logging
import os
from importlib.metadata import version

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import torch  # noqa

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DETERMINISTIC = os.environ.get("VIVAE_DETERMINISTIC", default="1") != "0"
TRY_CUDA = os.environ.get("VIVAE_CUDA", default="1") != "0"
TRY_MPS = os.environ.get("VIVAE_MPS", default="0") != "0"

if TRY_CUDA and torch.cuda.is_available():
    DEVICE = torch.tensor([1.0], device=torch.device("cuda")).device
    DEVICE_NAME = "cuda"
elif TRY_MPS and torch.backends.mps.is_available():
    DEVICE = torch.tensor([1.0], device=torch.device("mps")).device
    DEVICE_NAME = "mps"
else:
    DEVICE = torch.tensor([1.0], device=torch.device("cpu")).device
    DEVICE_NAME = "cpu"
torch.set_default_device(DEVICE)
torch.set_default_dtype(torch.float32)

torch.use_deterministic_algorithms(DETERMINISTIC)

from .model import ViVAE
from .diagnostics import encoder_indicatrices
from .knn import make_knn, smooth
from .plotting import plot_embedding, plot_indicatrices

logger.info(
    'initialised with "%s" backend and determinism %s',
    DEVICE_NAME,
    "enabled" if DETERMINISTIC else "disabled",
)

__all__ = [
    "DEVICE",
    "DEVICE_NAME",
    "DETERMINISTIC",
    "ViVAE",
    "encoder_indicatrices",
    "make_knn",
    "smooth",
    "plot_embedding",
    "plot_indicatrices"
]
__version__ = version("vivae")