
import sys
sys.path.append('../input/pytorch-image-models/pytorch-image-models-master')
import colorednoise as cn
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

from tqdm import tqdm

import albumentations as A
import albumentations.pytorch.transforms as T

import matplotlib.pyplot as plt


import transformers
from torch.cuda.amp import autocast, GradScaler

import os
import random
import time
from model import TimmSED
from utils import AverageMeter, MetricMeter, loss_fn, mixup, mixup_criterion, cutmix, cutmix_criterion 
from datasets import WaveformDataset

class CFG:
    ######################
    # Globals #
    ######################
    EXP_ID = 'EX001'
    seed = 42
    epochs = 40
    cutmix_and_mixup_epochs = 18
    folds = [0] # [0, 1, 2, 3, 4]
    N_FOLDS = 5
    LR = 1e-3
    ETA_MIN = 1e-6
    WEIGHT_DECAY = 1e-6
    train_bs = 16 # 32
    valid_bs = 32 # 64
    base_model_name = "tf_efficientnet_b0_ns"
    EARLY_STOPPING = True
    DEBUG = False # True
    EVALUATION = 'AUC'
    apex = True

    pooling = "max"
    pretrained = True
    num_classes = 152
    in_channels = 3
    target_columns = 'afrsil1 akekee akepa1 akiapo akikik amewig aniani apapan arcter \
                      barpet bcnher belkin1 bkbplo bknsti bkwpet blkfra blknod bongul \
                      brant brnboo brnnod brnowl brtcur bubsan buffle bulpet burpar buwtea \
                      cacgoo1 calqua cangoo canvas caster1 categr chbsan chemun chukar cintea \
                      comgal1 commyn compea comsan comwax coopet crehon dunlin elepai ercfra eurwig \
                      fragul gadwal gamqua glwgul gnwtea golphe grbher3 grefri gresca gryfra gwfgoo \
                      hawama hawcoo hawcre hawgoo hawhaw hawpet1 hoomer houfin houspa hudgod iiwi incter1 \
                      jabwar japqua kalphe kauama laugul layalb lcspet leasan leater1 lessca lesyel lobdow lotjae \
                      madpet magpet1 mallar3 masboo mauala maupar merlin mitpar moudov norcar norhar2 normoc norpin \
                      norsho nutman oahama omao osprey pagplo palila parjae pecsan peflov perfal pibgre pomjae puaioh \
                      reccar redava redjun redpha1 refboo rempar rettro ribgul rinduc rinphe rocpig rorpar rudtur ruff \
                      saffin sander semplo sheowl shtsan skylar snogoo sooshe sooter1 sopsku1 sora spodov sposan \
                      towsol wantat1 warwhe1 wesmea wessan wetshe whfibi whiter whttro wiltur yebcar yefcan zebdov'.split()

    img_size = 224 # 128
    main_metric = "epoch_f1_at_03"

    period = 5
    n_mels = 224 # 128
    fmin = 20
    fmax = 16000
    n_fft = 2048
    hop_length = 512
    sample_rate = 32000
    duration = period
    sr = sample_rate
    melspectrogram_parameters = {
        "n_mels": 224, # 128,
        "fmin": 20,
        "fmax": 16000
    }
    
    
# class AudioParams:
#     """
#     Parameters used for the audio data
#     """
#     sr = CFG.sample_rate
#     duration = CFG.period
#     # Melspectrogram
#     n_mels = CFG.n_mels
#     fmin = CFG.fmin
#     fmax = CFG.fmax


def set_seed(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
def train_fn(model, data_loader, device, optimizer, scheduler):
    model.train()
    scaler = GradScaler(enabled=CFG.apex)
    losses = AverageMeter()
    scores = MetricMeter()
    tk0 = tqdm(data_loader, total=len(data_loader))
    
    for data in tk0:
        optimizer.zero_grad()
        inputs = data['image'].to(device)
        targets = data['targets'].to(device)
        with autocast(enabled=CFG.apex):
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        scheduler.step()
        losses.update(loss.item(), inputs.size(0))
        scores.update(targets, outputs)
        tk0.set_postfix(loss=losses.avg)
    return scores.avg, losses.avg



def train_mixup_cutmix_fn(model, data_loader, device, optimizer, scheduler):
    model.train()
    scaler = GradScaler(enabled=CFG.apex)
    losses = AverageMeter()
    scores = MetricMeter()
    tk0 = tqdm(data_loader, total=len(data_loader))

    for data in tk0:
        optimizer.zero_grad()
        inputs = data['image'].to(device)
        targets = data['targets'].to(device)

        if np.random.rand()<0.5:
            inputs, new_targets = mixup(inputs, targets, 0.4)
            with autocast(enabled=CFG.apex):
                outputs = model(inputs)
                loss = mixup_criterion(outputs, new_targets) 
        else:
            inputs, new_targets = cutmix(inputs, targets, 0.4)
            with autocast(enabled=CFG.apex):
                outputs = model(inputs)
                loss = cutmix_criterion(outputs, new_targets)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        scheduler.step()
        losses.update(loss.item(), inputs.size(0))
        scores.update(new_targets[0], outputs)
        tk0.set_postfix(loss=losses.avg)
    return scores.avg, losses.avg


def valid_fn(model, data_loader, device):
    model.eval()
    losses = AverageMeter()
    scores = MetricMeter()
    tk0 = tqdm(data_loader, total=len(data_loader))
    valid_preds = []
    with torch.no_grad():
        for data in tk0:
            inputs = data['image'].to(device)
            targets = data['targets'].to(device)
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)
            losses.update(loss.item(), inputs.size(0))
            scores.update(targets, outputs)
            tk0.set_postfix(loss=losses.avg)
    return scores.avg, losses.avg


def inference_fn(model, data_loader, device):
    model.eval()
    tk0 = tqdm(data_loader, total=len(data_loader))
    final_output = []
    final_target = []
    with torch.no_grad():
        for b_idx, data in enumerate(tk0):
            inputs = data['image'].to(device)
            targets = data['targets'].to(device).detach().cpu().numpy().tolist()
            output = model(inputs)
            output = output["clipwise_output"].cpu().detach().cpu().numpy().tolist()
            final_output.extend(output)
            final_target.extend(targets)
    return final_output, final_target



if __name__ == '__main__':

    OUTPUT_DIR = f'./'
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    
    set_seed(CFG.seed)
    cfg = CFG()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train = pd.read_csv('train_folds.csv')

    mean = (0.485, 0.456, 0.406) # RGB
    std = (0.229, 0.224, 0.225) # RGB

    albu_transforms = {
        'train' : A.Compose([
                A.HorizontalFlip(p=0.5),
                A.OneOf([
                    A.Cutout(max_h_size=5, max_w_size=16),
                    A.CoarseDropout(max_holes=4),
                ], p=0.5),
                A.Normalize(mean, std),
        ]),
        'valid' : A.Compose([
                A.Normalize(mean, std),
        ]),
    }
    
    for fold in range(5):
        if fold not in CFG.folds:
            continue
        print("=" * 100)
        print(f"Fold {fold} Training")
        print("=" * 100)

        trn_df = train[train.kfold != fold].reset_index(drop=True)
        val_df = train[train.kfold == fold].reset_index(drop=True)

        train_dataset = WaveformDataset(trn_df, cfg, albu_transforms, mode='train')
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset, batch_size=CFG.train_bs, num_workers=0, pin_memory=True, shuffle=True
        )
        
        valid_dataset = WaveformDataset(val_df, cfg, albu_transforms, mode='valid')
        valid_dataloader = torch.utils.data.DataLoader(
            valid_dataset, batch_size=CFG.valid_bs, num_workers=0, pin_memory=True, shuffle=False
        )

        model = TimmSED(
            base_model_name=CFG.base_model_name,
            cfg=cfg,
            pretrained=CFG.pretrained,
            num_classes=CFG.num_classes,
            in_channels=CFG.in_channels)

        optimizer = torch.optim.AdamW(model.parameters(), lr=CFG.LR, weight_decay=CFG.WEIGHT_DECAY)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, eta_min=CFG.ETA_MIN, T_max=500)
 
        model = model.to(device)

        min_loss = 999
        best_score = -np.inf

        for epoch in range(CFG.epochs):
            print("Starting {} epoch...".format(epoch+1))

            start_time = time.time()

            if epoch < CFG.cutmix_and_mixup_epochs:
                train_avg, train_loss = train_mixup_cutmix_fn(model, train_dataloader, device, optimizer, scheduler)
            else: 
                train_avg, train_loss = train_fn(model, train_dataloader, device, optimizer, scheduler)

            valid_avg, valid_loss = valid_fn(model, valid_dataloader, device)

            elapsed = time.time() - start_time

            print(f'Epoch {epoch+1} - avg_train_loss: {train_loss:.5f}  avg_val_loss: {valid_loss:.5f}  time: {elapsed:.0f}s')
            print(f"Epoch {epoch+1} - train_f1_at_03:{train_avg['f1_at_03']:0.5f}  valid_f1_at_03:{valid_avg['f1_at_03']:0.5f}")
            print(f"Epoch {epoch+1} - train_f1_at_05:{train_avg['f1_at_05']:0.5f}  valid_f1_at_05:{valid_avg['f1_at_05']:0.5f}")

            if valid_avg['f1_at_03'] > best_score:
                print(f">>>>>>>> Model Improved From {best_score} ----> {valid_avg['f1_at_03']}")
                print(f"other scores here... {valid_avg['f1_at_03']}, {valid_avg['f1_at_05']}")
                torch.save(model.state_dict(), f'fold-{fold}.bin')
                best_score = valid_avg['f1_at_03']



    #     model_paths = [f'fold-{i}.bin' for i in CFG.folds]

    #     calc_cv(model_paths)
