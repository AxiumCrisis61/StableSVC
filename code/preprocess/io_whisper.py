import numpy as np
import torch
import os
from tqdm import tqdm
import sys

sys.path.append("../")
from config import data_path


def load_whisper_features(dataset, dataset_type):
    data_dir = os.path.join(data_path, dataset)
    input_dir = os.path.join(data_dir, "Whisper", dataset_type)
    print("Loading Whisper features from: ", input_dir)
    whisper_features = None

    for root, dirs, files in os.walk(input_dir):
        num = len(tuple(files))
        whisper_feature_shape = torch.load(os.path.join(input_dir, files[0])).shape
        whisper_features = np.zeros((num, whisper_feature_shape[0], whisper_feature_shape[1]), dtype=float)
        for index, file in enumerate(files):
            print(index)
            whisper_features[index] = torch.load(os.path.join(input_dir, file))

    if whisper_features is None:
        raise ValueError('Cannot find Whisper features!')
    else:
        return whisper_features
