import os
from argparse import ArgumentParser, ArgumentTypeError


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
# For standardizing mel-spectrograms in order to fit the input of DDPM (obtained from Opencpop dataset)
MEL_MAX = 1.3106
MEL_MIN = -9.2947


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
# UNet
CHANNELS_INPUT = 4                              # Mel + Whisper + F0 + Loudness
CHANNELS_OUTPUT = 1                             # Mel
CHANNELS_BASE = 80                              # base channel for UNet (the output channel for the first block)
CHANNELS_MULT_FACTORS = (2, 4, 8, 16)           # from official DDPM, (320, 640, 1280, 1280) channels for 'AUDIT'
BASIC_BLOCK = 'convnext'                        # basic block of the denoising UNet: ('resnet', 'convnext')
POSITION_EMBED_DIM = 128                        # dimension of raw time embedding, same as 'DiffSVC'
# DDPM
DIFFUSION_STEPS = 100
LINEAR_BETA_1 = 0.9                             # same as 'AUDIT'
LINEAR_BETA_T = 0.9999
MEAN_PARAMETERIZATION = 'eps'
# Cross-attention
ATTN_DIM_WHISPER = 320
ATTN_HEADS_WHISPER = 4
ATTN_DIM_F0 = 80
ATTN_HEADS_F0 = 1
ATTN_DIM_LOUDNESS = 80
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


parser = ArgumentParser(description="Acoustic Mapping")

# ======================== Dataset ========================

parser.add_argument("--dataset", type=str, default="Opencpop")
parser.add_argument("--converse", type=str2bool, default=False)
parser.add_argument("--whisper_dim", type=int, default=WHISPER_DIM)
# parser.add_argument("--output_dim", type=int, default=MCEP_DIM)
parser.add_argument(
    "--save", type=str, default="ckpts/debug", help="folder to save the final model"
)

# ======================== Accoutic Models ========================
parser.add_argument("--model", type=str, default="Transformer")

parser.add_argument("--transformer_input_length", type=int, default=800)
parser.add_argument("--transformer_dropout", type=float, default=0.1)
parser.add_argument("--transformer_d_model", type=int, default=768)
parser.add_argument("--transformer_nhead", type=int, default=8)
parser.add_argument("--transformer_nlayers", type=int, default=6)

# ======================== Training ========================

parser.add_argument("--lr", type=float, default=1e-4, help="initial learning rate")
parser.add_argument("--epochs", type=int, default=500, help="upper epoch limit")
parser.add_argument(
    "--batch_size", type=int, default=32, metavar="N", help="batch size"
)
parser.add_argument(
    "--start_epoch", type=int, default=0, help="No. of the epoch to start training"
)
parser.add_argument("--resume", type=str, default="", help="path to load trained model")
parser.add_argument(
    "--evaluate", type=str2bool, default=False, help="only use for evaluating"
)
parser.add_argument("--debug", type=str2bool, default=False)

# ======================== Devices ========================

parser.add_argument("--seed", type=int, default=9, help="random seed")
parser.add_argument("--device", default="cpu")
