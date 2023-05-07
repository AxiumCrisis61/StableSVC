from copy import deepcopy
from collections import OrderedDict
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torchaudio
from ddpm import GaussianDiffusionTrainer, GaussianDiffusionSampler
from conversion_model import DiffusionConverter
from argparse import ArgumentParser
import sys
sys.path.append("../")
from config import MEL_PADDING_LENGTH, MEL_MIN, MEL_MAX


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


class EMA(nn.Module):
    """
        Wrapper for building Pytorch model with Exponential Moving Average.
        Borrowed and modified from: https://www.zijianhu.com/post/pytorch/ema/
    """

    def __init__(self, model: nn.Module, decay=0.999):
        super().__init__()
        self.decay = decay

        self.model = model
        self.shadow = deepcopy(self.model)

        for param in self.shadow.parameters():
            param.detach_()

    @torch.no_grad()
    def update(self):
        if not self.training:
            print("EMA update should only be called during training", file=sys.stderr, flush=True)
            return

        model_params = OrderedDict(self.model.named_parameters())
        shadow_params = OrderedDict(self.shadow.named_parameters())

        # check if both model contains the same set of keys
        assert model_params.keys() == shadow_params.keys()

        for name, param in model_params.items():
            # see https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage
            # shadow_variable -= (1 - decay) * (shadow_variable - variable)
            shadow_params[name].sub_((1. - self.decay) * (shadow_params[name] - param))

        model_buffers = OrderedDict(self.model.named_buffers())
        shadow_buffers = OrderedDict(self.shadow.named_buffers())

        # check if both model contains the same set of keys
        assert model_buffers.keys() == shadow_buffers.keys()

        for name, buffer in model_buffers.items():
            # buffers are copied
            shadow_buffers[name].copy_(buffer)

    def forward(self, **inputs):
        if self.training:
            return self.model(**inputs)
        else:
            return self.shadow(**inputs)


class SVCDataset(Dataset):
    pass


def train_simple_ddpm(args):
    pass


if __name__ == '__main__':
    args = ArgumentParser(description='arguments for training DDPM')

