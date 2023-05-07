import whisper
import numpy as np
import torch
import torch.nn.functional as F
import os
import json
from argparse import ArgumentParser
import sys
sys.path.append("../")
from config import data_path, dataset2wavpath, WHISPER_SEQ, WHISPER_DIM, WHISPER_MODEL_SIZE, WHISPER_PADDING_LENGTH, \
    WHISPER_MAPPED_RATE


def whisper_encoder(audio_paths, arguments):
    batch = len(audio_paths)
    batch_mel = torch.zeros((batch, 80, WHISPER_PADDING_LENGTH*100), dtype=torch.float, device=model.device)

    for i, audio_path in enumerate(audio_paths):
        # load audio and pad/trim it to fit 30 seconds (determined by PADDING_LENGTH, seemingly not changeable)
        # (16000*PADDING_LENGTH, ): (480000,)
        audio = whisper.load_audio(str(audio_path))
        audio = whisper.pad_or_trim(audio, length=WHISPER_PADDING_LENGTH*16000)

        # (80, 100*PADDING_LENGTH): (80, 3000)
        batch_mel[i] = whisper.log_mel_spectrogram(audio).to(model.device)

    with torch.no_grad():
        # (batch, WHISPER_SEQ, WHISPER_DIM): (batch, 1500, 512)
        features = model.embed_audio(batch_mel)
        # transpose the feature maps to align with mel-spectrograms
        # (batch, WHISPER_DIM, WHISPER_SEQ): (batch, 512, 1500)
        features = torch.transpose(features, 1, 2)
        # (batch, WHISPER_DIM, WHISPER_SEQ/ave_rate): (batch, 512, 500)
        features = F.avg_pool1d(features, kernel_size=arguments.ave_rate, stride=arguments.ave_rate)

    del batch_mel
    for i in range(5):
        torch.cuda.empty_cache()

    return features.cpu().detach().numpy()


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

    # create saving list
    if not arguments.save_separate:
        whisper_feature_list = np.zeros((len(datasets), WHISPER_DIM, WHISPER_SEQ/arguments.ave_rate), dtype=float)

    # Extract raw features: (batch, WHISPER_DIM, WHISPER_SEQ/ave_rate)
    print("\nExtracting raw whisper features...")
    audio_paths = [os.path.join(wave_dir, "{}.wav".format(utt["Uid"])) for utt in datasets]
    if dataset == "M4Singer":
        audio_paths = [os.path.join(wave_dir, utt["Path"]) for utt in datasets]

    end = args.start_point
    while end <= len(audio_paths):
        # update progress
        start = end
        end = start + batch_size
        print("{}/{}...".format(min(len(audio_paths), end), len(audio_paths)))

        # extract Whisper features
        whisper_features = whisper_encoder(audio_paths[start:end], arguments)

        # save each sample's Whisper embedding respectively
        if arguments.save_separate:
            for index in range(min(batch_size, len(audio_paths)-start)):
                torch.save(whisper_features[index], os.path.join(output_dir, "{}.pth".format(datasets[start+index]['Uid'])))
        else:
            whisper_feature_list[start:end] = whisper_features


if __name__ == "__main__":
    parser = ArgumentParser(description="Acoustic Mapping")
    parser.add_argument("--batch-size", type=int, default=80)
    parser.add_argument("--dataset", type=str, choices=('Opencpop', 'M4Singer'))
    parser.add_argument("--dataset-type", type=str, choices=('train', 'test'))
    parser.add_argument("--start-point", type=int, default=0)
    parser.add_argument("--ave-rate", type=int, default=WHISPER_MAPPED_RATE,
                        help='kernel size of temporal average pooling to the Whisper feature maps')
    parser.add_argument("--save-separate", type=bool, default=True,
                        help='whether to save each feature map of the audio as separate file')
    args = parser.parse_args()

    print("Loading Model...")

    model = whisper.load_model(WHISPER_MODEL_SIZE)
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
