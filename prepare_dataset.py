import cv2
import audioread
import logging
import gc
import os
import sys
sys.path.append('../input/pytorch-image-models/pytorch-image-models-master')
import random
import time
import warnings

import librosa
import numpy as np
import pandas as pd
import soundfile as sf
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as torchdata

from contextlib import contextmanager
from joblib import Parallel, delayed
from pathlib import Path
from typing import Optional
from sklearn.model_selection import StratifiedKFold, GroupKFold

from albumentations.core.transforms_interface import ImageOnlyTransform
from torchlibrosa.stft import LogmelFilterBank, Spectrogram
from torchlibrosa.augmentation import SpecAugmentation
from tqdm import tqdm

import albumentations as A
import albumentations.pytorch.transforms as T

import matplotlib.pyplot as plt


SR = 32000
USE_SEC = 30 # 90 # 60

def Audio_to_Array(path):
    y, sr = sf.read(path, always_2d=True)
    y = np.mean(y, 1) # there is (X, 2) array
    if len(y) > SR:
        y = y[SR:-SR]

    if len(y) > SR * USE_SEC:
        y = y[:SR * USE_SEC]
    return y

def save_(path):
    save_path = "dataset/train_np/" + "/".join(path.split('/')[-2:])
    np.save(save_path, Audio_to_Array(path))

AUDIO_PATH = 'dataset/birdclef-2022/train_audio'

train = pd.read_csv('dataset/birdclef-2022/train_metadata.csv')
train["file_path"] = AUDIO_PATH + '/' + train['filename']
paths = train["file_path"].values

NUM_WORKERS = 16
CLASSES = sorted(os.listdir(AUDIO_PATH))

for dir_ in tqdm(CLASSES):
    _ = os.makedirs('dataset/train_np/' + dir_, exist_ok=True)
_ = Parallel(n_jobs=NUM_WORKERS)(delayed(save_)(AUDIO_PATH) for AUDIO_PATH in tqdm(paths))




