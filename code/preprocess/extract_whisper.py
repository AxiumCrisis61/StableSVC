import whisper
import torch
import os
import json
import numpy as np
import pickle
from tqdm import tqdm
import sys
from argparse import ArgumentParser, ArgumentTypeError

sys.path.append("../")
from config import data_path, dataset2wavpath, WHISPER_SEQ, WHISPER_DIM, WHISPER_MAPPED, PADDING_LENGTH


def whisper_encoder(audio_paths):
    batch = len(audio_paths)
    batch_mel = torch.zeros((batch, 80, PADDING_LENGTH*100), dtype=torch.float, device=model.device)

    for i, audio_path in enumerate(audio_paths):
        # load audio and pad/trim it to fit 30 seconds (determined by PADDING_LENGTH, seemingly not changeable)
        # (16000*PADDING_LENGTH, ): (480000,)
        audio = whisper.load_audio(str(audio_path))
        audio = whisper.pad_or_trim(audio, length=PADDING_LENGTH*16000)

        # (80, 100*PADDING_LENGTH): (80, 3000)
        batch_mel[i] = whisper.log_mel_spectrogram(audio).to(model.device)

    with torch.no_grad():
        # (batch, WHISPER_SEQ, WHISPER_DIM): (batch, 1500, 1024)
        features = model.embed_audio(batch_mel)

    del batch_mel
    for i in range(5):
        torch.cuda.empty_cache()

    return features.cpu().detach().numpy()


def get_mapped_whisper_features(dataset, dataset_type, raw_whisper_features):
    MCEP_dir = os.path.join(data_path, dataset, "MCEP")
    with open(os.path.join(MCEP_dir, "{}.pkl".format(dataset_type)), "rb") as f:
        mceps = pickle.load(f)
    print("MCEPs: {}, mceps[0] = {}".format(len(mceps), mceps[0].shape))

    whisper_features = []
    for index, mcep in enumerate(tqdm(mceps)):
        sz = len(mcep)

        # (1500, 1024)
        raw_feats = raw_whisper_features[index]

        feats = np.zeros((sz, WHISPER_DIM), dtype=float)
        for i in range(sz):
            feats[i] = raw_feats[int(i / 2)]
        whisper_features.append(feats)

    return whisper_features


def extract_whisper_features(dataset, dataset_type, arguments):
    batch_size = arguments.batch_size
    print("-" * 20)
    print("Dataset: {}, {}".format(dataset, dataset_type))

    # create output directory
    data_dir = os.path.join(data_path, dataset)
    os.makedirs(data_dir, exist_ok=True)
    output_dir = os.path.join(data_dir, "Whisper", dataset_type)
    os.makedirs(output_dir, exist_ok=True)

    # load directory for .wav file
    wave_dir = dataset2wavpath[dataset]
    with open(os.path.join(data_dir, "{}.json".format(dataset_type)), "r") as f:
        datasets = json.load(f)

    # Extract raw features: (sz, 1500, 1024)
    print("\nExtracting raw whisper features...")
    audio_paths = [
        os.path.join(wave_dir, "{}.wav".format(utt["Uid"])) for utt in datasets
    ]
    if dataset == "M4Singer":
        audio_paths = [os.path.join(wave_dir, utt["Path"]) for utt in datasets]

    end = args.start_point
    while end <= len(audio_paths):
        # update progress
        start = end
        end = start + batch_size
        print("{}/{}...".format(min(len(audio_paths), end), len(audio_paths)))

        # extract Whisper features
        whisper_features = whisper_encoder(audio_paths[start:end])

        # save each sample's Whisper embedding respectively
        for index in range(min(batch_size, len(audio_paths)-start)):
            torch.save(whisper_features[index], os.path.join(output_dir, "{}.pth".format(start+index)))

    # Mapping to MCEP's lengths [WARN: Not maintained.]
    if WHISPER_MAPPED:
        print("\nTransform to mapped features...")
        whisper_features = get_mapped_whisper_features(dataset, dataset_type, whisper_features)
        torch.save(whisper_features, os.path.join(output_dir, "{}.pth".format(dataset_type)))


if __name__ == "__main__":
    parser = ArgumentParser(description="Acoustic Mapping")
    parser.add_argument("--batch-size", type=int, default=80)
    parser.add_argument("--dataset", type=str, choices=('Opencpop', 'M4Singer'))
    parser.add_argument("--dataset-type", type=str, choices=('train', 'test'))
    parser.add_argument("--start-point", type=int, default=0)
    args = parser.parse_args()

    print("Loading Model...")

    model = whisper.load_model("medium")
    if torch.cuda.is_available():
        print("Using GPU...\n")
        model = model.cuda()
    else:
        print("Using CPU...\n")

    model = model.eval()

    extract_whisper_features(args.dataset, args.dataset_type, args)
    # extract_whisper_features("Opencpop", "test", args)
    # extract_whisper_features("Opencpop", "train", args)
    # extract_whisper_features("M4Singer", "test", args)
