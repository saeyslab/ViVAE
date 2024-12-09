import logging

import os
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

import torch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DETERMINISTIC = True # setting to False may improve performance but worsen reproducibility
TRY_CUDA      = True # try to use CUDA if available
TRY_MPS       = False # try to use MPS if available (disabled by defaul)

if TRY_CUDA and torch.cuda.is_available():
    DEVICE      = torch.tensor([1.],device=torch.device('cuda')).device
    DEVICE_NAME = 'cuda'
elif TRY_MPS and torch.backends.mps.is_available():
    DEVICE      = torch.tensor([1.],device=torch.device('mps')).device
    DEVICE_NAME = 'mps'
else:
    DEVICE      = torch.tensor([1.],device=torch.device('cpu')).device
    DEVICE_NAME = 'cpu'
torch.set_default_device(DEVICE)
torch.set_default_dtype(torch.float32)

torch.use_deterministic_algorithms(DETERMINISTIC)

logger.info('initialised with "%s" backend and determinism %s', DEVICE_NAME, 'enabled' if DETERMINISTIC else 'disabled')

from .model import ViVAE
from .plotting import plot_embedding, plot_indicatrices
from .knn import make_knn, smooth
