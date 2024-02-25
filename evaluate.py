import os
import json
import torch
import torch.nn as nn
import numpy as np
import os.path as osp
import torch.nn.functional as F
from PIL import Image
from dataloader import CreateTestDataLoader
from torch.autograd import Variable
from options.test_options import TestOptions
from utils.metric import Evaluator
from utils.util import accumulate_distribution
import cv2
torch.set_num_threads(4)


def evaluate(num_classes, val_loader, model):
    model.eval()
    val_pred_0 = 1e-5
    val_pred_1 = 1e-5

    metric = Evaluator(num_class=2)
    with torch.no_grad():
        for i, (val_img, _, val_gt, _) in enumerate(val_loader):
            val_img = val_img.cuda().detach()                 # to gpu
            val_gt = val_gt.cuda().detach().unsqueeze(dim=1)  # to gpu
            _, _, h, w = val_img.size()
            # val_feat = model.get_features(val_img)           # get features
            # val_pred = model.get_predicts(val_feat, [h, w])  # make classification
            val_pred = model(val_img)           # get features
            val_pred = (val_pred>=0.5).long()
            val_pred = val_pred.cpu().detach().numpy()
            val_gt = val_gt.cpu().detach().numpy()
            metric.add_batch(val_gt, val_pred)

            print (np.sum(val_pred==0), np.sum(val_pred==1))

            val_pred_0 += np.sum(val_pred==0 + 0)
            val_pred_1 += np.sum(val_pred==1 + 0)

    val_acc = metric.Pixel_Accuracy()
    val_IoU = metric.Intersection_over_Union()

    model.train()

    return val_acc, val_IoU, val_pred_0/val_pred_1



