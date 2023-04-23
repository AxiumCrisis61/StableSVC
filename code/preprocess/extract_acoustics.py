import os
import sys
import json
import pickle
import numpy as np
import torch
import torchaudio
import diffsptk
import librosa
from tqdm import tqdm

sys.path.append("../")
from config import data_path, dataset2wavpath, RE_SAMPLE_RATE, MEL_FREQ_BINS, STFT_N, STFT_WINDOW_SIZE, STFT_HOP_SIZE


def extract_acoustic_features(wave_file, pitch_extractor):
    # waveform: (1, seq)
    waveform, sample_rate = torchaudio.load(wave_file)
    waveform = torchaudio.functional.resample(waveform, orig_freq=sample_rate, new_freq=RE_SAMPLE_RATE)

    # transform to Mel-spectrogram
    mel_spectrogram = torchaudio.transforms.MelSpectrogram(n_fft=STFT_N,
                                                           win_length=STFT_WINDOW_SIZE,
                                                           hop_length=STFT_HOP_SIZE,
                                                           n_mels=MEL_FREQ_BINS,
                                                           normalized=True)(waveform[0])
    # extract loudness
    power_spectrogram = torchaudio.transforms.Spectrogram(n_fft=STFT_N,
                                                          win_length=STFT_WINDOW_SIZE,
                                                          hop_length=STFT_HOP_SIZE,
                                                          power=2)(waveform[0])
    weighted_spectrogram = librosa.perceptual_weighting(power_spectrogram,
                                                        librosa.cqt_frequencies(power_spectrogram.shape[0],
                                                                                fmin=librosa.note_to_hz('A1')))
    loudness = np.log(np.mean(np.exp(weighted_spectrogram[0]), axis=0) + 1e-5)
    # extract pitch
    pitch = pitch_extractor(waveform[0])

    return mel_spectrogram, pitch, loudness


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
    crepe = diffsptk.Pitch(STFT_HOP_SIZE, RE_SAMPLE_RATE, out_format='pitch')

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
        'F0': f0_features,
        'Loudness': loudness_features
    }

    # save
    for feature_name, feature in dict_features.items():
        output_dir = os.path.join(data_dir, feature_name)
        os.makedirs(output_dir, exist_ok=True)
        torch.save(feature, os.path.join(output_dir, "{}.pth".format(dataset_type)))


if __name__ == '__main__':
    extract_acoustic_features_of_datasets("Opencpop", "test")
    extract_acoustic_features_of_datasets("Opencpop", "train")
    # extract_acoustic_features_of_datasets("M4Singer", "test")
