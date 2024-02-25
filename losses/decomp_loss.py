import random
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from .jaccard_loss import JaccardLoss


class DecomposeLoss(nn.Module):
    def __init__(self, ampli=1, cls_fore=1, cls_back=0):
        super(DecomposeLoss, self).__init__()
        self.ampli = ampli
        self.cls_fore = cls_fore
        self.cls_back = cls_back

    # def get_seperated_predicts(self, feat, ):

    def forward(self, model, feat_fore, feat_back, out_fus, label, cls_wgt, h, w):
        rand = (np.random.beta(2,1) + 1) / 2
        w_high, w_low = max(rand, 1-rand), min(rand, 1-rand)
        # print (w_high, w_low)
        feat_fore = w_high*feat_fore + w_low*feat_back
        feat_back = w_low*feat_fore + w_high*feat_back
        out_fore = model.get_predicts(feat_fore, [h,w])            
        out_back = model.get_predicts(feat_back, [h,w])    

        # Decomposition Loss     
        cls_wgt = {0:0.5, 1:0.5}  
        wgt_fore = (label==self.cls_fore).float()*w_high*cls_wgt[self.cls_fore] + (label==self.cls_back).float()*w_low*cls_wgt[self.cls_back]
        wgt_back = (label==self.cls_back).float()*w_high*cls_wgt[self.cls_back] + (label==self.cls_fore).float()*w_low*cls_wgt[self.cls_fore]
        loss_dcp = (F.binary_cross_entropy(out_fore, label, reduction='none')*wgt_fore).mean() \
                 + (F.binary_cross_entropy(out_back, label, reduction='none')*wgt_back).mean()
        loss_dcp *= 2

        # Calibration loss
        wgt_fore = torch.abs(self.cls_fore-out_back) * ((label==self.cls_fore).float())
        wgt_back = torch.abs(self.cls_back-out_fore) * ((label==self.cls_back).float())
        loss_cbr = (F.binary_cross_entropy(out_fus, label, reduction='none') * (wgt_fore + wgt_back)).mean()

        return loss_cbr*self.ampli, loss_dcp*self.ampli



# class DecomposeLoss(nn.Module):
#     def __init__(self, ampli=1, cls_fore=1, cls_back=0):
#         super(DecomposeLoss, self).__init__()
#         self.ampli = ampli
#         self.cls_fore = cls_fore
#         self.cls_back = cls_back

#     def forward(self, model, feat_fore, feat_back, out_fus, label, cls_wgt, h, w):
#         rand = (np.random.beta(2,1) + 1) / 2
#         w_high, w_low = max(rand, 1-rand), min(rand, 1-rand)
#         # print (w_high, w_low)
#         feat_fore = w_high*feat_fore + w_low*feat_back
#         feat_back = w_low*feat_fore + w_high*feat_back
#         out_fore = model.get_predicts(feat_fore)            
#         out_back = model.get_predicts(feat_back)    
#         out_fore = F.interpolate(out_fore, size=(h, w), mode='bilinear')
#         out_back = F.interpolate(out_back, size=(h, w), mode='bilinear')

#         # Decomposition Loss       
#         wgt_fore = (label==self.cls_fore).float()*w_high + (label==self.cls_back).float()*w_low
#         wgt_back = (label==self.cls_back).float()*w_high + (label==self.cls_fore).float()*w_low
#         loss_dcp = (F.binary_cross_entropy(out_fore, label, reduction='none')*wgt_fore).mean()*cls_wgt[self.cls_fore] \
#                  + (F.binary_cross_entropy(out_back, label, reduction='none')*wgt_back).mean()*cls_wgt[self.cls_back]
#         loss_dcp *= 2

#         # Calibration loss
#         wgt_fore = torch.abs(self.cls_fore-out_back) * ((label==self.cls_fore).float())
#         wgt_back = torch.abs(self.cls_back-out_fore) * ((label==self.cls_back).float())
#         loss_cbr = (F.binary_cross_entropy(out_fus, label, reduction='none') * (wgt_fore + wgt_back)).mean()

#         return loss_cbr*self.ampli, loss_dcp*self.ampli


