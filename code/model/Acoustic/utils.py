import os
import json
from copy import deepcopy
from collections import OrderedDict
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import Dataset
import sys

sys.path.append("../../")
from config import data_path, dataset2wavpath, RE_SAMPLE_RATE, MEL_PADDING_LENGTH, \
    MEL_MIN, MEL_MAX, F0_MAX, F0_MIN, LOUDNESS_MAX, LOUDNESS_MIN


def save_checkpoint(filepath, obj):
    print("Saving checkpoint to {}".format(filepath))
    torch.save(obj, filepath)
    print("Complete.")


def load_checkpoint(filepath, device):
    assert os.path.isfile(filepath)
    print("Loading '{}'".format(filepath))
    checkpoint_dict = torch.load(filepath, map_location=device)
    print("Complete.")
    return checkpoint_dict


def get_standardizer():
    """
        Standardizer for Mel-spectrograms, f0 and loudness

    Returns:
        mel_standardizer, f0_standardizer, loudness_standardizer
    """
    return Standardizer(MEL_MIN, MEL_MAX), Standardizer(F0_MIN, F0_MAX), Standardizer(LOUDNESS_MIN, LOUDNESS_MAX)


class Standardizer(object):
    """
        Standardizer to the range of [-1, 1] to fit the input of DDPM.
        Estimated from Opencpop dataset.
    """

    def __init__(self, min_value, max_value):
        self.min = min_value
        self.max = max_value

    def __call__(self, x: torch.Tensor) -> torch.Tensor: return self.scale(x)

    def scale(self, x: torch.Tensor) -> torch.Tensor:
        """
            Scale the input to the range of [-1, 1] to fit the input of DDPM.
            Estimated from Opencpop dataset.
        """
        return (x - self.min) / (self.max - self.min) * 2 - 1

    def scale_back(self, x: torch.Tensor) -> torch.Tensor:
        """
            Scale back the output of DDPM.
            Estimated from Opencpop dataset.
        """
        return (x + 1) / 2 * (self.max - self.min) + self.min


class SVCDataset(Dataset):
    """
        PyTorch Dataset for SVC task
    """

    def __init__(self, dataset, dataset_type):
        """
            Initialize PyTorch Dataset for SVC task

        Args:
            dataset: name of the dataset, current support: ('Opencpop', 'M4Singer')
            dataset_type: type of the dataset, ('train', 'test')
        """
        super().__init__()

        self.dataset = dataset
        self.dataset_type = dataset_type

        # load transcription file
        self.data_dir = os.path.join(data_path, dataset)
        with open(os.path.join(self.data_dir, "{}.json".format(dataset_type)), "r") as f:
            self.transcription = json.load(f)

        # feature paths
        self.wav_path = dataset2wavpath[dataset]
        self.mel_path = os.path.join(self.data_dir, 'Mel', dataset_type)
        self.whisper_path = os.path.join(self.data_dir, 'Whisper', dataset_type)
        self.f0_path = os.path.join(self.data_dir, 'F0', '{}.pth'.format(dataset_type))
        self.loudness_path = os.path.join(self.data_dir, 'Loudness', '{}.pth'.format(dataset_type))

        # load and save small features
        self.f0 = torch.load(self.f0_path)
        assert len(self.f0[0].shape) == 1, 'f0 input needs to be one dimensional'
        self.loudness = torch.load(self.loudness_path)
        assert len(self.loudness[0].shape) == 1, 'loudness input needs to be one dimensional'

        # standardizer
        self.mel_standardizer = Standardizer(MEL_MIN, MEL_MAX)
        self.f0_standardizer = Standardizer(F0_MIN, F0_MAX)
        self.loudness_standardizer = Standardizer(LOUDNESS_MIN, LOUDNESS_MAX)

    def __getitem__(self, index):
        uid = self.transcription[index]["Uid"]

        # load
        mel = self.mel_standardizer(torch.Tensor(np.load(os.path.join(self.mel_path, "{}.npy".format(uid)))))
        whisper = torch.Tensor(np.load(os.path.join(self.whisper_path, "{}.npy".format(uid))))
        f0 = self.f0_standardizer(torch.Tensor(self.f0[index]))
        loudness = self.loudness_standardizer(torch.Tensor(self.loudness[index]))

        # temporally pad or trim the acoustic features
        # Mels extracted by codes from Hifi-GAN will have the temporal dimension less than f0 and loudness
        # extracted by torchaudio than 1 ?
        # Different lengths of f0 and loudness ?
        length = mel.shape[-1]
        if length <= MEL_PADDING_LENGTH:
            mel = F.pad(mel, (0, MEL_PADDING_LENGTH - length), 'constant', 0)
        else:
            mel = mel[:, :MEL_PADDING_LENGTH]

        length = f0.shape[-1]
        if length <= MEL_PADDING_LENGTH:
            f0 = F.pad(f0, (0, MEL_PADDING_LENGTH - length), 'constant', 0)
        else:
            f0 = f0[:MEL_PADDING_LENGTH]

        length = loudness.shape[-1]
        if length <= MEL_PADDING_LENGTH:
            loudness = F.pad(loudness, (0, MEL_PADDING_LENGTH - length), 'constant', 0)
        else:
            loudness = loudness[:MEL_PADDING_LENGTH]

        return mel, whisper, f0, loudness

    def __len__(self):
        return len(self.transcription)


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


# for testing functionality
if __name__ == '__main__':
    print(torch.Tensor(torch.Tensor(torch.zeros((5, 5)))))
    print(3 == 2 > 0)