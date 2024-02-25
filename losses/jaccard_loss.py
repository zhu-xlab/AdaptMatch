import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def make_one_hot(input, num_classes):
    """Convert class index tensor to one hot encoding tensor.
    Args:
         input: A tensor of shape [N, 1, *]
         num_classes: An int of number of class
    Returns:
        A tensor of shape [N, num_classes, *]
    """
    shape = np.array(input.shape)
    shape[1] = num_classes
    shape = tuple(shape)
    result = torch.zeros(shape)
    result = result.scatter_(1, input.cpu(), 1)

    return result.cuda()


class BinaryJaccardLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(BinaryJaccardLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):        
        #flatten label and prediction tensors
        inputs = inputs.reshape(-1)
        targets = targets.reshape(-1)
        
        #intersection is equivalent to True Positive count
        #union is the mutually inclusive area of all labels & predictions 
        intersection = (inputs * targets).sum()
        total = (inputs + targets).sum()
        union = total - intersection 
        
        IoU = (intersection + smooth)/(union + smooth)
                
        return 1 - IoU


class JaccardLoss(nn.Module):
    """Dice loss of binary class
    Args:
        smooth: A float number to smooth loss, and avoid NaN error, default: 1
        p: Denominator value: sum{x^p} + sum{y^p}, default: 2
        predict: A tensor of shape [N, *]
        target: A tensor of shape same with predict
        reduction: Reduction method to apply, return mean over batch if 'mean',
            return sum if 'sum', return a tensor of shape [N,] if 'none'
    Returns:
        Loss tensor according to arg reduction
    Raise:
        Exception if unexpected reduction
    """
    def __init__(self, num_classes=2):
        super(JaccardLoss, self).__init__()
        self.num_classes = num_classes
        self.jaccard = BinaryJaccardLoss()

    def forward(self, predict, target, weight=[0.5, 0.5]):
        # predict = F.softmax(predict, dim=1)
        # target = make_one_hot(target.unsqueeze(dim=1), self.num_classes)
        target = (target>0.5).float()

        total_loss = 0
        for i in range(self.num_classes):
            jaccard_loss = self.jaccard(predict[:, i], target[:, i])
            jaccard_loss *= weight[i]
            total_loss += jaccard_loss

        return total_loss 