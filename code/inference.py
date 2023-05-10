import os
import json
from copy import deepcopy
from argparse import ArgumentParser

import numpy as np

from preprocess.extract_acoustics import extract_acoustic_features
from model.Acoustic.ddpm import GaussianDiffusionSampler
from model.Acoustic.conversion_model import DiffusionConverter
from model.Hifi_GAN.models import Generator
from model.Acoustic.utils import load_checkpoint, get_standardizer, EMA
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchaudio
import pandas as pd
import whisper
import diffsptk
from config import CKPT_ACOUSTIC, CKPT_VOCODER, VOCODER_CONFIG_PATH, INFERENCE_DATA_PATH, OUTPUT_DIR, \
    DIFFUSION_STEPS, NOISE_SCHEDULE, MEL_FREQ_BINS, MEL_PADDING_LENGTH, RE_SAMPLE_RATE, WHISPER_MODEL_SIZE, \
    WHISPER_PADDING_LENGTH, WHISPER_MAPPED_RATE, STFT_HOP_SIZE


MAX_WAV_VALUE = 32768.0


def get_vocoder(vocoder_config_path, device):
    with open(vocoder_config_path, "r") as f:
        h = json.load(f)
    vocoder = Generator(h)
    vocoder.load_state_dict(load_checkpoint(CKPT_VOCODER, device)['generator'])
    vocoder.eval()
    vocoder.remove_weight_norm()

    return vocoder


def whisper_encoder(waveform_list):
    """
        Modified from preprocess.extract_whisper/whisper_encoder to adapt to waveform input
    """
    # load Wisper Encoder
    model = whisper.load_model(WHISPER_MODEL_SIZE)
    if torch.cuda.is_available():
        print("Using GPU...\n")
        model = model.cuda()
    else:
        print("Using CPU...\n")
    model = model.eval()

    batch = len(waveform_list)
    batch_mel = torch.zeros((batch, 80, WHISPER_PADDING_LENGTH*100), dtype=torch.float, device=model.device)

    for i, audio in enumerate(waveform_list):
        audio = whisper.pad_or_trim(audio, length=WHISPER_PADDING_LENGTH*16000)
        batch_mel[i] = whisper.log_mel_spectrogram(audio).to(model.device)

    with torch.no_grad():
        features = model.embed_audio(batch_mel)
        features = torch.transpose(features, 1, 2)
        features = F.avg_pool1d(features, kernel_size=WHISPER_MAPPED_RATE, stride=WHISPER_MAPPED_RATE)

    del batch_mel
    for i in range(5):
        torch.cuda.empty_cache()

    return features.cpu().detach().numpy()


class InferenceDataset(Dataset):
    def __init__(self, input_dir):
        class AcousticArguments(object):
            def __init__(self):
                self.mel = False
                self.f0 = True
                self.loudness = True
                self.hifi_gan = True

        acoustic_arguments = AcousticArguments()
        wav_path_list = [i for i in os.listdir(input_dir) if i[-3:] == 'wav']
        self.wav_name_list = wav_path_list
        self.num_samples = len(wav_path_list)
        _, self.f0_standardizer, self.loudness_standardizer = get_standardizer()

        # load waveform
        waveform_list = []
        for wav_path in wav_path_list:
            waveform, sample_rate = torchaudio.load(wav_path)
            waveform = torchaudio.functional.resample(waveform, orig_freq=sample_rate, new_freq=RE_SAMPLE_RATE)
            waveform_list.append(waveform)

        # get whisper embedding
        self.whisper_features = whisper_encoder(waveform_list)

        # get acoustic features: f0 and loudness
        acoustic_features = []
        crepe = diffsptk.Pitch(STFT_HOP_SIZE, RE_SAMPLE_RATE, out_format='pitch', model='tiny')
        for waveform in waveform_list:
            acoustic_features.append(extract_acoustic_features(waveform,  crepe, acoustic_arguments))
        del crepe
        f0_list, loudness_list = zip(acoustic_features)
        del acoustic_features

        # pad or trim
        self.f0_list = []
        for f0 in f0_list:
            length = f0.shape[-1]
            if length <= MEL_PADDING_LENGTH:
                f0 = F.pad(f0, (0, MEL_PADDING_LENGTH - length), 'constant', 0)
            else:
                f0 = f0[:MEL_PADDING_LENGTH]
            self.f0_list.append(f0)
        self.loudness_list = []
        for loudness in loudness_list:
            length = loudness.shape[-1]
            if length <= MEL_PADDING_LENGTH:
                loudness = F.pad(loudness, (0, MEL_PADDING_LENGTH - length), 'constant', 0)
            else:
                loudness = loudness[:MEL_PADDING_LENGTH]
            self.loudness_list.append(loudness)
        if self.whisper_features.shape[2] > MEL_PADDING_LENGTH:
            trim_len_whisper = (self.whisper_features.shape[2] - MEL_PADDING_LENGTH) // 2
            self.whisper_features = self.whisper_features[:, :, trim_len_whisper:-trim_len_whisper]
        else:
            pad_len_whisper = (MEL_PADDING_LENGTH - self.whisper_features.shape[2]) // 2
            self.whisper_features = F.pad(self.whisper_features, (pad_len_whisper, pad_len_whisper), 'replicate')

    def __getitem__(self, index):
        return self.whisper_features[index], \
               self.f0_standardizer(torch.Tensor(self.f0_list[index])), \
               self.loudness_standardizer(torch.Tensor(self.loudness_list[index]))

    def __len__(self):
        return self.num_samples


def inference(input_dir, output_type='all', output_dir=OUTPUT_DIR, plot_interval=0, evaluation=True, arguments=None):
    """
    Args:
        input_dir: path containing source audios (stored as wav files)
        output_type: whether to output converted mel-spectrograms or audios (if yes, Hifi-GAN vocoder will be called)
        output_dir: directory to store converted output
        plot_interval: intervals to plot the noise mel-spectrograms during reverse diffusion process
        evaluation: whether to evaluate the results
        arguments: arguments for the model

    Returns:
        Converted audios / mel-spectrograms
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mel_standardizer, _, _ = get_standardizer()

    # create output directory
    if output_type == 'audio':
        output_dir_audio = os.path.join(output_dir, 'audio')
        os.makedirs(output_dir_audio, exist_ok=True)
        output_dir_mel = None
    elif output_type == 'mel':
        output_dir_audio = None
        output_dir_mel = os.path.join(output_dir, 'mel')
        os.makedirs(output_dir_mel, exist_ok=True)
    elif output_type == 'all':
        output_dir_audio = os.path.join(output_dir, 'audio')
        os.makedirs(output_dir_audio, exist_ok=True)
        output_dir_mel = os.path.join(output_dir, 'mel')
        os.makedirs(output_dir_mel, exist_ok=True)
    else:
        raise ValueError("Unsupported output type, choose from ('audio', 'mel', 'all')")

    # data loader
    inference_dataset = InferenceDataset(input_dir)
    data_loader = DataLoader(inference_dataset, batch_size=arguments.batch_size)

    # models
    if arguments.framework == 'simple_diffusion':
        # load backbone
        backbone = DiffusionConverter(cross_attention=False)
        if arguments.use_ema:
            arguments.use_ema = 'ema'
            backbone = EMA(backbone)
        else:
            arguments.use_ema = 'model'
        backbone.load_state_dict(load_checkpoint(os.path.join(CKPT_ACOUSTIC, arguments.framework, arguments.epoch),
                                                  device)[arguments.use_ema])
        if arguments.use_ema == 'ema':
            shadow = deepcopy(backbone.shadow)
            del backbone
            backbone = shadow

        # load converter
        converter = GaussianDiffusionSampler(backbone, T=DIFFUSION_STEPS, noise_schedule=NOISE_SCHEDULE,
                                             plot_interval=plot_interval)
    else:
        raise ValueError('Other types of SVC framework not supported')
    backbone.to(device)
    backbone.eval()
    converter.to(device)
    converter.eval()

    # inference
    num_samples = inference_dataset.num_samples
    converted_mels = torch.zeros((num_samples, 80, MEL_PADDING_LENGTH), dtype=torch.float, device=device)
    start = end = 0
    with torch.no_grad():
        for whisper, f0, loudness in data_loader:
            # create noise
            end += num
            num = whisper.shape[0]
            noise = torch.randn((num, MEL_FREQ_BINS, MEL_PADDING_LENGTH)).to(device)
            whisper = whisper.to(device)
            f0 = f0.to(device)
            loudness = loudness.to(device)

            # conversion
            x = converter(noise, whisper=whisper, f0=f0, loudness=loudness)

            # scale the converted Mel-spectrograms back from [-1, 1] and store
            converted_mels[start:end] = mel_standardizer.scale_back(x.cpu().numpy())

            del whisper, f0, loudness, x
            for i in range(5):
                torch.cuda.empty_cache()

    # save the converted Mel-spectrograms
    np.save(os.path.join(output_dir_mel, 'mels.npy'), converted_mels)

    # converting to waveform
    del backbone, converter
    for i in range(5):
        torch.cuda.empty_cache()
    if output_dir_mel is not None:
        vocoder = get_vocoder(VOCODER_CONFIG_PATH, device)
        vocoder.to(device)

        with torch.no_grad():
            converted_audios = vocoder(torch.Tensor(converted_mels).to(device))

        converted_audios = converted_audios.squeeze()
        converted_audios = converted_audios * MAX_WAV_VALUE
        converted_audios = converted_audios.cpu()

        for index, wav_name in enumerate(inference_dataset.wav_name_list):
            torchaudio.save(os.path.join(output_dir_audio, '{}_converted.wav'.format(wav_name[:-4])),
                            converted_audios[index], sample_rate=RE_SAMPLE_RATE)


if __name__ == '__main__':
    # inference settings
    arg_parser_settings = ArgumentParser(description='Settings for inference')
    arg_parser_settings.add_argument('--input-dir', type=str, default=INFERENCE_DATA_PATH,
                                     help='input wav files directory')
    arg_parser_settings.add_argument('--output-type', type=str, choices=('audio', 'mel', 'all'), default='all',
                                     help='contents to output')
    arg_parser_settings.add_argument('--output-dir', type=str, default=OUTPUT_DIR,
                                     help='output directory')
    arg_parser_settings.add_argument('--plot-interval', type=int, default=10,
                                     help='intervals to plot the noise mel-spectrograms during reverse diffusion process')
    arg_parser_settings.add_argument('--evaluation', type=bool, default=True,
                                     help='whether to evaluate the results')

    # models configuration
    arg_parser_model = ArgumentParser(description='Arguments for inference model')
    arg_parser_model.add_argument('--batch-size', type=int, default=8, help='inference batch size')
    arg_parser_model.add_argument('--epoch', type=str, choices=('latest', 'best'), default='best')
    arg_parser_model.add_argument('--use-ema', type=bool, default=True)
    arg_parser_model.add_argument('--framework', type=str, choices=('simple_diffusion', ))

    settings = arg_parser_settings.parse_args()
    arguments = arg_parser_model.parse_args()

    # inference
    inference(settings.input_dir,
              settings.output_type,
              settings.output_dir,
              settings.plot_interval,
              settings.evaluation,
              arguments)
