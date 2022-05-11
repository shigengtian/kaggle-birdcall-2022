import sys
sys.path.append('../input/pytorch-image-models/pytorch-image-models-master')
import random
import time
import warnings

import numpy as np
import pandas as pd

from contextlib import contextmanager
from joblib import Parallel, delayed
from pathlib import Path
from typing import Optional
from sklearn.model_selection import StratifiedKFold, GroupKFold

import ast
import glob

all_path = glob.glob('dataset/train_np/*/*.npy')



train = pd.read_csv('dataset/birdclef-2022/train_metadata.csv')


train['new_target'] = train['primary_label'] + ' ' + train['secondary_labels'].map(lambda x: ' '.join(ast.literal_eval(x)))
train['len_new_target'] = train['new_target'].map(lambda x: len(x.split()))


path_df = pd.DataFrame(all_path, columns=['file_path'])
path_df['filename'] = path_df['file_path'].map(lambda x: x.split('/')[-2]+'/'+x.split('/')[-1][:-4])


print(path_df.head())

train = pd.merge(train, path_df, on='filename')
print(train.shape)
train.head()


Fold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for n, (trn_index, val_index) in enumerate(Fold.split(train, train['primary_label'])):
    train.loc[val_index, 'kfold'] = int(n)
train['kfold'] = train['kfold'].astype(int)

train.to_csv('train_folds.csv', index=False)