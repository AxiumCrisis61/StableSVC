import os
import sys
import json
import pickle
import numpy as np
import torch
import torchaudio
import diffsptk
from tqdm import tqdm
from argparse import ArgumentParser, ArgumentTypeError
import librosa
from librosa.util import normalize
from librosa.filters import mel as librosa_mel_fn
import warnings
warnings.simplefilter(action='ignore', category=UserWarning)

sys.path.append("../")
from config import data_path, dataset2wavpath, RE_SAMPLE_RATE, MEL_FREQ_BINS, STFT_N, STFT_WINDOW_SIZE, STFT_HOP_SIZE, \
    F_MIN, F_MAX

MAX_WAV_VALUE = 32768.0

""" function transform_mel_spectrogram is borrowed from Hifi-GAN repository """


def dynamic_range_compression(x, C=1, clip_val=1e-5):
    return np.log(np.clip(x, a_min=clip_val, a_max=None) * C)


def dynamic_range_decompression(x, C=1):
    return np.exp(x) / C


def dynamic_range_compression_torch(x, C=1, clip_val=1e-5):
    return torch.log(torch.clamp(x, min=clip_val) * C)


def dynamic_range_decompression_torch(x, C=1):
    return torch.exp(x) / C


def spectral_normalize_torch(magnitudes):
    output = dynamic_range_compression_torch(magnitudes)
    return output


def spectral_de_normalize_torch(magnitudes):
    output = dynamic_range_decompression_torch(magnitudes)
    return output


mel_basis = {}
hann_window = {}


def transform_mel_spectrogram(y, n_fft, num_mels, sampling_rate, hop_size, win_size, fmin, fmax, center=False):
    if torch.min(y) < -1.:
        print('min value is ', torch.min(y))
    if torch.max(y) > 1.:
        print('max value is ', torch.max(y))

    global mel_basis, hann_window
    if fmax not in mel_basis:
        mel = librosa_mel_fn(sr=sampling_rate, n_fft=n_fft, n_mels=num_mels, fmin=fmin, fmax=fmax)
        mel_basis[str(fmax) + '_' + str(y.device)] = torch.from_numpy(mel).float().to(y.device)
        hann_window[str(y.device)] = torch.hann_window(win_size).to(y.device)

    y = torch.nn.functional.pad(y.unsqueeze(1), (int((n_fft - hop_size) / 2), int((n_fft - hop_size) / 2)),
                                mode='reflect')
    y = y.squeeze(1)

    spec = torch.stft(y, n_fft, hop_length=hop_size, win_length=win_size, window=hann_window[str(y.device)],
                      center=center, pad_mode='reflect', normalized=False, onesided=True, return_complex=False)

    spec = torch.sqrt(spec.pow(2).sum(-1) + 1e-9)

    spec = torch.matmul(mel_basis[str(fmax) + '_' + str(y.device)], spec)
    spec = spectral_normalize_torch(spec)

    return spec


def extract_acoustic_features(wave_file, pitch_extractor):
    # waveform: (1, seq)
    waveform, sample_rate = torchaudio.load(wave_file)
    waveform = torchaudio.functional.resample(waveform, orig_freq=sample_rate, new_freq=RE_SAMPLE_RATE)

    # transform to Mel-spectrogram
    mel_spectrogram = transform_mel_spectrogram(waveform[0], STFT_N, MEL_FREQ_BINS,
                                                RE_SAMPLE_RATE, STFT_HOP_SIZE, STFT_WINDOW_SIZE, F_MIN, F_MAX,
                                                center=False)
    # mel_spectrogram = torchaudio.transforms.MelSpectrogram(n_fft=STFT_N,
    #                                                        win_length=STFT_WINDOW_SIZE,
    #                                                        hop_length=STFT_HOP_SIZE,
    #                                                        n_mels=MEL_FREQ_BINS,
    #                                                        normalized=True)(waveform[0])

    # # extract loudness
    # power_spectrogram = torchaudio.transforms.Spectrogram(n_fft=STFT_N,
    #                                                       win_length=STFT_WINDOW_SIZE,
    #                                                       hop_length=STFT_HOP_SIZE,
    #                                                       power=2)(waveform[0])
    # weighted_spectrogram = librosa.perceptual_weighting(power_spectrogram,
    #                                                     librosa.cqt_frequencies(power_spectrogram.shape[0],
    #                                                                             fmin=librosa.note_to_hz('A1')))
    # loudness = np.log(np.mean(np.exp(weighted_spectrogram[0]), axis=0) + 1e-5)
    # # extract pitch
    # pitch = pitch_extractor(waveform[0])

    # return mel_spectrogram, pitch, loudness
    return mel_spectrogram, None, None


def extract_acoustic_features_of_datasets(dataset, dataset_type):
    """
    Extract acoustic features of the audio dataset.
    Modified from 'extract_mcep_features' from 'extract_mcep.py'

    Args:
        dataset: name of the dataset
        dataset_type: train / test

    Returns:
        acoustic features: (MelSpectrogram, f0, loudness, AP)
    """
    print("-" * 20)
    print("Dataset: {}, {}".format(dataset, dataset_type))
    crepe = diffsptk.Pitch(STFT_HOP_SIZE, RE_SAMPLE_RATE, out_format='pitch', model='tiny')

    # handle directories
    data_dir = os.path.join(data_path, dataset)
    os.makedirs(data_dir, exist_ok=True)
    wave_dir = dataset2wavpath[dataset]

    # load dataset transcriptions
    with open(os.path.join(data_dir, "{}.json".format(dataset_type)), "r") as f:
        datasets = json.load(f)

    # extract acoustic features, i.e., Mel-spectrogram, f0(pitch) and loudness
    print("\nExtracting acoustic features...")
    mel_spectrograms = []
    f0_features = []
    loudness_features = []
    for utt in tqdm(datasets):
        uid = utt["Uid"]
        wave_file = os.path.join(wave_dir, "{}.wav".format(uid))

        if dataset == "M4Singer":
            wave_file = os.path.join(wave_dir, utt["Path"])

        mel, f0, loudness = extract_acoustic_features(wave_file, crepe)

        mel_spectrograms.append(mel)
        f0_features.append(f0)
        loudness_features.append(loudness)

    dict_features = {
        'Mel': mel_spectrograms,
        # 'F0': f0_features,
        # 'Loudness': loudness_features
    }

    # save
    for feature_name, feature in dict_features.items():
        output_dir = os.path.join(data_dir, feature_name)
        os.makedirs(output_dir, exist_ok=True)
        torch.save(feature, os.path.join(output_dir, "{}.pth".format(dataset_type)))


if __name__ == '__main__':
    parser = ArgumentParser(description="Acoustic Mapping")
    parser.add_argument("--dataset", type=str, choices=('Opencpop', 'M4Singer'))
    parser.add_argument("--dataset-type", type=str, choices=('train', 'test'))
    args = parser.parse_args()

    extract_acoustic_features_of_datasets(args.dataset, args.dataset_type)
    # extract_acoustic_features_of_datasets("Opencpop", "test")
    # extract_acoustic_features_of_datasets("Opencpop", "train")
    # extract_acoustic_features_of_datasets("M4Singer", "test")
