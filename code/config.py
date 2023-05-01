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

# We select 5 utterances randomly for every singer
NUMS_OF_SINGER = 5

# Hifi_GAN Vocoder settings




# Acoustic features hyperparameters
PADDING_LENGTH = 30           # padding length of the audios, not changeable due to the fixed input length of Whisper being 30s
RE_SAMPLE_RATE = 16000
MEL_FREQ_BINS = 80           # frequency bins for mel-spectrograms
STFT_N = 1024                # size of FFT in STFT
STFT_WINDOW_SIZE = 1024      # window size of STFT
STFT_HOP_SIZE = 256          # hop size of STFT
MEL_PAD_LENGTH = 496         # 8s of audio under the above settings

# Whisper hyperparameters
WHISPER_SEQ = 1500
WHISPER_DIM = 512
WHISPER_MAPPED = False      # whether to map Whisper features to the length of MCEP
WHISPER_MODEL_SIZE = 'base'


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
