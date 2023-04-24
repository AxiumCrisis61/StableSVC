import numpy as np
import torch
import os
from tqdm import tqdm
import sys

sys.path.append("../")
from config import data_path, WHISPER_SEQ, WHISPER_DIM


def dump_whisper_features(whisper_features, dataset, dataset_type):
    # create output directory
    data_dir = os.path.join(data_path, dataset)
    os.makedirs(data_dir, exist_ok=True)
    output_dir = os.path.join(data_dir, "Whisper", dataset_type)
    os.makedirs(output_dir, exist_ok=True)

    # save each sample's Whisper feature
    print('Dumping given Whisper features...')
    for i in tqdm(range(whisper_features.shape[0])):
        torch.save(whisper_features[i], os.path.join(output_dir, "{}.pth".format(i)))


def load_whisper_features(dataset, dataset_type):
    data_dir = os.path.join(data_path, dataset)
    input_dir = os.path.join(data_dir, "Whisper", dataset_type)

    num = len(tuple(os.walk(input_dir)))
    whisper_features = np.zeros((num, WHISPER_SEQ, WHISPER_DIM), dtype=float)

    print("Loading Whisper features from: ", input_dir)
    for index in range(num):
        whisper_features[index] = torch.load(os.path.join(input_dir, "{}.pth".format(index)))

    return whisper_features
