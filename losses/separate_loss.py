import os
import torch
import torch.nn as nn
import torch.nn.functional as F


class SeparateLoss(nn.Module):
    def __init__(self, num_classes=2):
        super(SeparateLoss, self).__init__()
        self.num_classes = num_classes

    def forward(self, feat, label, size=(64,64)):
        feat = F.interpolate(feat.float(), size, mode='nearest')
        label = F.interpolate(label.float(), size, mode='nearest')  
        b, c, h, w = feat.size()
        N = b*h*w

        feat = feat.permute(0,2,3,1).view(N, c)
        label = label.squeeze(dim=1).view(N, 1).expand(N, N)
        feat = F.normalize(feat, p=2)   # N x C
        simi = torch.mm(feat, feat.permute(1,0))
        mask = (~torch.eq(label, label.permute(1,0))).float()

        loss_map = mask*simi

        loss = loss_map.mean()
        # loss_map = loss_map.max(dim=1)[0]
        # loss = loss_map.mean(0)

        return loss


# class SeparateLoss(nn.Module):
#     def __init__(self, num_classes=2):
#         super(SeparateLoss, self).__init__()
#         self.num_classes = num_classes

#     def forward(self, feat, label):
#         size = feat.size()[-2:]
#         label = F.interpolate(label.float(), size, mode='nearest')  

#         mask_back, mask_build = (label==0).float(), (label==1).float()
#         feat_back  = (mask_back*feat).sum(dim=-1).sum(dim=-1) / (mask_back.sum(dim=-1).sum(dim=-1) + 1e-5)
#         feat_build = (mask_build*feat).sum(dim=-1).sum(dim=-1) / (mask_build.sum(dim=-1).sum(dim=-1) + 1e-5)
#         feat_back  = F.normalize(feat_back, p=2)   # B x C
#         feat_build = F.normalize(feat_build, p=2)  # B x C
#         mask_back  = 1 - (mask_back.sum(dim=-1).sum(dim=-1)==0).float()
#         mask_build = 1 - (mask_build.sum(dim=-1).sum(dim=-1)==0).float()
#         mask = torch.mm(mask_back, mask_build.permute(1,0))
#         feat = torch.mm(feat_back, feat_build.permute(1,0))
#         dis  = (mask*(feat/2 + 0.5)).mean()

#         return dis


