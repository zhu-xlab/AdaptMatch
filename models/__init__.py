import os 
import torch
from models.deeplab_v3plus import Deeplab_V3plus
from models.fcn8s import VGG16_FCN8s
from models.hrnet import HRNet
from models.swin_transformer import Swin_Transformer
from models.segformer import SegFormer
from models.buildformer import BuildFormer
from models.efficientunet import efficientunet_b0, efficientunet_b1, efficientunet_b2, efficientunet_b3

def CreateModel(model, num_classes):
    if model == 'DeepLab_V3plus':
        seg_model = Deeplab_V3plus(num_classes=num_classes)
        
    if model == 'HRNet':
        seg_model = HRNet(num_classes=num_classes)

    if model == 'Swin_Transformer':
        seg_model = Swin_Transformer(num_classes=num_classes)

    if model == 'SegFormer':
        seg_model = SegFormer(num_classes=num_classes, phi='b2')

    if model == 'BuildFormer':
        seg_model = BuildFormer(num_classes=num_classes)

    if model == 'EfficientUNet':
        seg_model = efficientunet_b1(out_channels=num_classes)

    return seg_model
