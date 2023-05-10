from config import CHECKPOINT_PATH_ACOUSTIC, FRAMEWORK, VOCODER_CONFIG_PATH
import json
from argparse import ArgumentParser
from preprocess.extract_whisper import whisper_encoder
from preprocess.extract_acoustics import extract_acoustic_features
from model.Acoustic.ddpm import GaussianDiffusionSampler
from model.Hifi_GAN.models import Generator
import matplotlib.pyplot as plt


def inference(audio, output_mel=False):
    """
        Inference of a single audio input.

    Args:
        audio: source audio
        output_mel: whether to output converted mel-spectrogram or audio (if yes, Hifi-GAN vocoder will be called)

    Returns:
        Converted audio / mel-spectrogram
    """
    if output_mel:
        with open(VOCODER_CONFIG_PATH, "r") as f:
            h = json.load(f)
        vocoder = Generator(h)


    pass


def inference_from_path(path, output_mel=False):
    """

    Args:
        path: path containing source audios
        output_mel: whether to output converted mel-spectrograms or audios (if yes, Hifi-GAN vocoder will be called)

    Returns:
        Converted audios / mel-spectrograms
    """
    pass


if __name__ == '__main__':
    pass
