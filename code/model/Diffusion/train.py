import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torchaudio
from models import UNet, GaussianDiffusionTrainer, GaussianDiffusionSampler
import sys
sys.path.append("../")
from config import PADDING_LENGTH, MEL_PADDING_LENGTH, MEL_MIN, MEL_MAX


def standardize(x: torch.Tensor) -> torch.Tensor:
    """
        Standardization to fit the input of DDPM.
        Estimated from Opencpop dataset.
    """
    return (x-MEL_MIN) / (MEL_MAX-MEL_MIN) * 2 - 1


def scale_back(x: torch.Tensor) -> torch.Tensor:
    """
        Scale back the output of DDPM.
        Estimated from Opencpop dataset.
    """
    return (x + 1) / 2 * (MEL_MAX-MEL_MIN) + MEL_MIN


class SVCDataset(Dataset):
    pass


if __name__ == '__main__':
    pass
