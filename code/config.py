import os
from argparse import ArgumentTypeError


# Please configure the path of your downloaded datasets
dataset2path = {
    "Opencpop": "/content/drive/MyDrive/MDS_6002_SVC/StableSVC/code/data/Opencpop",
    "M4Singer": "/content/drive/MyDrive/MDS_6002_SVC/StableSVC/code/data/M4Singer",
}       # for running on Colab


# Please configure the root path to save your data and model
root_path = "/content/drive/MyDrive/MDS_6002_SVC/StableSVC/code"        # for running on Colab
data_path = os.path.join(root_path, "preprocess")
model_path = os.path.join(root_path, "model")


# Wav files path
dataset2wavpath = {
    "Opencpop": os.path.join(dataset2path["Opencpop"], "segments/wavs"),
    "M4Singer": dataset2path["M4Singer"],
}


# Training features paths
INPUT_WAVS_DIR = "/content/drive/MyDrive/MDS_6002_SVC/StableSVC/code/data/Opencpop/segments/wavs"
INPUT_MELS_DIR = "/content/drive/MyDrive/MDS_6002_SVC/StableSVC/code/preprocess/Opencpop/Mel"
INPUT_F0_DIR = "/content/drive/MyDrive/MDS_6002_SVC/StableSVC/code/preprocess/Opencpop/F0"
INPUT_LOUDNESS_DIR = "/content/drive/MyDrive/MDS_6002_SVC/StableSVC/code/preprocess/Opencpop/Loudness"
INPUT_WHISPER_DIR = "/content/drive/MyDrive/MDS_6002_SVC/StableSVC/code/preprocess/Opencpop/Whisper"


# We select 5 utterances randomly for every singer
NUMS_OF_SINGER = 5


# Acoustic features hyperparameters
MEL_PADDING_LENGTH = 496         # roughly 8s of audio under the above settings
WHISPER_PADDING_LENGTH = 30      # padding length of the Whisper input audios, not changeable
RE_SAMPLE_RATE = 16000
MEL_FREQ_BINS = 80               # frequency bins for mel-spectrograms
STFT_N = 1024                    # size of FFT in STFT
STFT_WINDOW_SIZE = 1024          # window size of STFT
STFT_HOP_SIZE = 256              # hop size of STFT
F_MAX = 8000                     # maximal frequency for Mel filter banks (inherited from Hifi-GAN configuration v1)
F_MIN = 0
# For standardizing acoustic features in order to fit the input of DDPM
# (obtained from Opencpop training dataset as estimation)
MEL_MAX = 1.3106
MEL_MIN = -9.2947
F0_MAX = 508.5507
F0_MIN = 0
LOUDNESS_MAX = 33.35714916562806
LOUDNESS_MIN = -11.512925464970229


# Whisper hyperparameters
WHISPER_SEQ = 1500
WHISPER_DIM = 512             # 512 for 'base', 1024 for 'medium'
WHISPER_MAPPED_RATE = 3       # temporal-average-pooling rate for whisper feature maps
WHISPER_MODEL_SIZE = 'base'


# Hifi-GAN training settings (for default)
INPUT_TRAINING_FILE = "/content/drive/MyDrive/MDS_6002_SVC/StableSVC/code/data/Opencpop/segments/train.txt"
INPUT_VALIDATION_FILE = "/content/drive/MyDrive/MDS_6002_SVC/StableSVC/code/data/Opencpop/segments/test.txt"
CHECKPOINT_PATH_HIFI = "/content/drive/MyDrive/MDS_6002_SVC/StableSVC/code/model/Hifi_GAN/ckpt"
PRETRAIN_PATH = "/content/drive/MyDrive/MDS_6002_SVC/StableSVC/code/model/Hifi_GAN/ckpt/UNIVERSAL_V1"


# Hifi-GAN Vocoder settings


# VAE-PatchGAN hyperparameters
VAE_DIMENSIONS = {}


# Acoustic model settings
CHECKPOINT_PATH_ACOUSTIC = "/content/drive/MyDrive/MDS_6002_SVC/StableSVC/code/model/Acoustic/ckpt"
# Framework
USE_CROSS_ATTN = False
USE_EMA = False
# UNet
CHANNELS_INPUT = 4                              # Mel + Whisper + F0 + Loudness
CHANNELS_OUTPUT = 1                             # Mel
CHANNELS_BASE = 40                              # base channel for UNet (the output channel for the first block)
CHANNELS_MULT_FACTORS = (2, 4, 8, 8)           # from official DDPM, (320, 640, 1280, 1280) channels for 'AUDIT'
BASIC_BLOCK = 'convnext'                        # basic block of the denoising UNet: ('resnet', 'convnext')
POSITION_EMBED_DIM = 128                        # dimension of raw time embedding, same as 'DiffSVC'
# DDPM
DIFFUSION_STEPS = 100
LINEAR_BETA_1 = 0.9                             # same as 'AUDIT'
LINEAR_BETA_T = 0.9999
MEAN_PARAMETERIZATION = 'eps'
# Cross-attention
ATTN_HEADS_WHISPER = 4
ATTN_HEADS_F0 = 1
ATTN_HEADS_LOUDNESS = 1


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise ArgumentTypeError("Boolean value expected.")
