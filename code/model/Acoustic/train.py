from copy import deepcopy
from collections import OrderedDict
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
import torchaudio
from ddpm import GaussianDiffusionTrainer, GaussianDiffusionSampler
from conversion_model import DiffusionConverter
from argparse import ArgumentParser
import warnings
import sys
sys.path.append("../")
from config import MEL_PADDING_LENGTH, MEL_MIN, MEL_MAX, INPUT_WAVS_DIR, INPUT_MELS_DIR, INPUT_WHISPER_DIR, \
    INPUT_F0_DIR, INPUT_LOUDNESS_DIR, CHECKPOINT_PATH_ACOUSTIC


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
    def __init__(self, arguments):
        self.wav_path = arguments.input_wav
        self.mel_path = arguments.input_mel
        self.f0_path = arguments.input_f0
        self.loudness_path = arguments.input_loudness

        self.f0 = torch.load(self.f0_path)
        assert len(self.f0.shape) == 2, 'f0 input needs to be in shape of (batch, mel_padding_length)'
        self.loudness = torch.load(self.loudness_path)
        assert len(self.f0.shape) == 2, 'loudness input needs to be in shape of (batch, mel_padding_length)'

    def __getitem__(self, index):
        pass


def train_simple_ddpm(arguments):
    pass


if __name__ == '__main__':
    warnings.simplefilter(action='ignore', category=FutureWarning)

    arg_parser = ArgumentParser(description='arguments for training DDPM')

    arg_parser.add_argument('--framework', type=str, choices=('simple_diffusion', ), default='simple_diffusion',
                            help='choice of conversion framework')
    arg_parser.add_argument('--use-ema', type=bool, default=True,
                            help='whether to use Exponential Moving Average to the model')

    arg_parser.add_argument('--resume', type=bool, default=True,
                            help='whether to resume training from the latest checkpoint')
    arg_parser.add_argument('--validation', type=bool, default=True,
                            help='whether to perform validation during training')
    arg_parser.add_argument('--val-interval', type=int, default=200,
                            help='validation interval (steps)')

    arg_parser.add_argument('--epochs', type=int, default=100,
                            help='number of training epochs')
    arg_parser.add_argument('--batch-size', type=int, default=8,
                            help='batch size for mini-batch optimization')

    arg_parser.add_argument('--input-wav', type=str, default=INPUT_WAVS_DIR,
                            help='input directory of wav files')
    arg_parser.add_argument('--input-mel', type=str, default=INPUT_MELS_DIR,
                            help='input directory of wav files')
    arg_parser.add_argument('--input-f0', type=str, default=INPUT_F0_DIR,
                            help='input directory of wav files')
    arg_parser.add_argument('--input-loudness', type=str, default=INPUT_LOUDNESS_DIR,
                            help='input directory of wav files')
    arg_parser.add_argument('--ckpt-dir', type=str, default=CHECKPOINT_PATH_ACOUSTIC,
                            help='checkpoint path for the acoustic model')

    arg_parser.add_argument('--beta1', type=float, deault=0.9,
                            help='beta_1 for AdamW optimizer')
    arg_parser.add_argument('--beta2', type=float, deault=0.999,
                            help='beta_2 for ')
    arg_parser.add_argument('--weight-decay', type=float, deault=1e-2,
                            help='weight decay coefficient for AdamW optimizer')

    args = arg_parser.parse_args()

    if args.framework == 'simple_diffusion':
        train_simple_ddpm(args)
    else:
        raise ValueError("Unsupported conversion framework")
