import numpy as np
import pandas as pd
from sklearn import metrics
import torch
import torch.nn as nn
import torch.nn.functional as F

# ====================================================
# Training helper functions
# ====================================================
class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        

class MetricMeter(object):
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.y_true = []
        self.y_pred = []
    
    def update(self, y_true, y_pred):
        self.y_true.extend(y_true.cpu().detach().numpy().tolist())
        self.y_pred.extend(y_pred["clipwise_output"].cpu().detach().numpy().tolist())

    @property
    def avg(self):
        self.f1_03 = metrics.f1_score(np.array(self.y_true), np.array(self.y_pred) > 0.3, average="micro")
        self.f1_05 = metrics.f1_score(np.array(self.y_true), np.array(self.y_pred) > 0.5, average="micro")
        
        return {
            "f1_at_03" : self.f1_03,
            "f1_at_05" : self.f1_05,
        }
    
    
# https://www.kaggle.com/c/rfcx-species-audio-detection/discussion/213075
class BCEFocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, preds, targets):
        bce_loss = nn.BCEWithLogitsLoss(reduction='none')(preds, targets)
        probas = torch.sigmoid(preds)
        loss = targets * self.alpha * (1. - probas)**self.gamma * bce_loss +  (1. - targets) * probas**self.gamma * bce_loss
        loss = loss.mean()
        return loss


class BCEFocal2WayLoss(nn.Module):
    def __init__(self, weights=[1, 1], class_weights=None):
        super().__init__()

        self.focal = BCEFocalLoss()

        self.weights = weights

    def forward(self, input, target):
        input_ = input["logit"]
        target = target.float()

        framewise_output = input["framewise_logit"]
        clipwise_output_with_max, _ = framewise_output.max(dim=1)

        loss = self.focal(input_, target)
        aux_loss = self.focal(clipwise_output_with_max, target)

        return self.weights[0] * loss + self.weights[1] * aux_loss


def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


def cutmix(data, targets, alpha):
    indices = torch.randperm(data.size(0))
    shuffled_data = data[indices]
    shuffled_targets = targets[indices]

    lam = np.random.beta(alpha, alpha)
    bbx1, bby1, bbx2, bby2 = rand_bbox(data.size(), lam)
    data[:, :, bbx1:bbx2, bby1:bby2] = data[indices, :, bbx1:bbx2, bby1:bby2]
    # adjust lambda to exactly match pixel ratio
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (data.size()[-1] * data.size()[-2]))

    new_targets = [targets, shuffled_targets, lam]
    return data, new_targets

def mixup(data, targets, alpha):
    indices = torch.randperm(data.size(0))
    shuffled_data = data[indices]
    shuffled_targets = targets[indices]

    lam = np.random.beta(alpha, alpha)
    new_data = data * lam + shuffled_data * (1 - lam)
    new_targets = [targets, shuffled_targets, lam]
    return new_data, new_targets


def cutmix_criterion(preds, new_targets):
    targets1, targets2, lam = new_targets[0], new_targets[1], new_targets[2]
    criterion = BCEFocal2WayLoss()
    return lam * criterion(preds, targets1) + (1 - lam) * criterion(preds, targets2)

def mixup_criterion(preds, new_targets):
    targets1, targets2, lam = new_targets[0], new_targets[1], new_targets[2]
    criterion = BCEFocal2WayLoss()
    return lam * criterion(preds, targets1) + (1 - lam) * criterion(preds, targets2)


def loss_fn(logits, targets):
    loss_fct = BCEFocal2WayLoss()
    loss = loss_fct(logits, targets)
    return loss


def calc_loss(y_true, y_pred):
    return metrics.roc_auc_score(np.array(y_true), np.array(y_pred))

# def calc_cv(model_paths):
#     df = pd.read_csv('train_folds.csv')
#     y_true = []
#     y_pred = []
#     for fold, model_path in enumerate(model_paths):
#         model = TimmSED(
#             base_model_name=CFG.base_model_name,
#             pretrained=CFG.pretrained,
#             num_classes=CFG.num_classes,
#             in_channels=CFG.in_channels)

#         model.to(device)
#         model.load_state_dict(torch.load(model_path))
#         model.eval()

#         val_df = df[df.kfold == fold].reset_index(drop=True)
#         dataset = WaveformDataset(df=val_df, mode='valid')
#         dataloader = torch.utils.data.DataLoader(
#             dataset, batch_size=CFG.valid_bs, num_workers=0, pin_memory=True, shuffle=False
#         )

#         final_output, final_target = inference_fn(model, dataloader, device)
#         y_pred.extend(final_output)
#         y_true.extend(final_target)
#         torch.cuda.empty_cache()

#         f1_03 = metrics.f1_score(np.array(y_true), np.array(y_pred) > 0.3, average="micro")
#         print(f'micro f1_0.3 {f1_03}')

#     f1_03 = metrics.f1_score(np.array(y_true), np.array(y_pred) > 0.3, average="micro")
#     f1_05 = metrics.f1_score(np.array(y_true), np.array(y_pred) > 0.5, average="micro")

#     print(f'overall micro f1_0.3 {f1_03}')
#     print(f'overall micro f1_0.5 {f1_05}')
#     return
