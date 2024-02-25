import time
import torch.nn.functional as F
import torch
import numpy as np


class Timer(object):
    """A simple timer."""
    def __init__(self):
        self.total_time = 0.
        self.calls = 0
        self.start_time = 0.
        self.diff = 0.
        self.average_time = 0.

    def tic(self):
        self.start_time = time.time()

    def toc(self, average=True):
        self.diff = time.time() - self.start_time
        self.total_time += self.diff
        self.calls += 1
        self.average_time = self.total_time / self.calls
        if average:
            return self.average_time
        else:
            return self.diff

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


# def accumulate_distribution(dist_list, pred, gt, cls_fore, cls_back):
#     h, w = 64, 64
#     b = pred.size()[0]

#     gt = F.interpolate(gt, size=[h, w], mode='nearest').squeeze(dim=1).view(b, h*w).view(b*h*w)
#     pred = F.interpolate(pred, size=[h, w], mode='nearest').squeeze(dim=1).view(b, h*w).view(b*h*w)
#     mask_fore, mask_back = (gt==cls_fore), (gt==cls_back)
#     pred_fore = torch.abs(cls_fore - pred[mask_fore]).tolist()
#     pred_back = torch.abs(cls_back - pred[mask_back]).tolist()
#     dist_list[cls_fore].extend(pred_fore)
#     dist_list[cls_back].extend(pred_back)

#     return dist_list


def accumulate_distribution(list_length, dist_list, pred, gt, cls_fore, cls_back):
    h, w = 64, 64
    b = pred.size()[0]

    gt = F.interpolate(gt, size=[h, w], mode='nearest').squeeze(dim=1).view(b, h*w).view(b*h*w)
    pred = F.interpolate(pred, size=[h, w], mode='nearest').squeeze(dim=1).view(b, h*w).view(b*h*w)
    mask_fore, mask_back = (gt==cls_fore), (gt==cls_back)
    pred_fore = torch.abs(cls_fore - pred[mask_fore]).tolist()
    pred_back = torch.abs(cls_back - pred[mask_back]).tolist()

    if pred_fore != []:
        dist_list[cls_fore].append(pred_fore)
    if pred_back != []:
        dist_list[cls_back].append(pred_back)
    if len(dist_list[cls_fore]) > list_length:
        dist_list[cls_fore].pop(0)
    if len(dist_list[cls_back]) > list_length:
        dist_list[cls_back].pop(0)

    return dist_list


def calc_threshold(dist_list):
    total_sum = 0
    total_num = 1e-7
    for inner_list in dist_list:
        total_num += len(inner_list)
        total_sum += sum(inner_list)

    thr = total_sum / total_num

    return thr


def cutmix(imgs, gts, masks):
    batch_size, _, img_h, img_w = imgs.size()
    indices = torch.linspace(batch_size-1, 0, batch_size).long()
    shuffled_imgs = imgs[indices]
    shuffled_gts = gts[indices]
    shuffled_masks = masks[indices]

    lam = np.random.uniform(0, 1)
    cut_h, cut_w = int(img_h * lam), int(img_w * lam)

    x1, y1 = np.random.randint(img_h-cut_h), np.random.randint(img_w-cut_w)
    x2, y2 = x1 + cut_h, y1 + cut_w
    mask_cutmix = torch.zeros((batch_size, 1, img_h, img_w)).cuda()

    imgs = imgs*mask_cutmix + shuffled_imgs*(1 - mask_cutmix)
    gts = gts*mask_cutmix + shuffled_gts*(1 - mask_cutmix)
    masks = masks*mask_cutmix + shuffled_masks*(1 - mask_cutmix)

    return imgs.detach(), gts.detach(), masks.detach()