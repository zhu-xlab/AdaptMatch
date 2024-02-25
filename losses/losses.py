import os
import torch
import torch.nn as nn
import torch.nn.functional as F



def con_loss(out, out_bar):
    loss = F.l1_loss(F.normalize(out, dim=1).detach(), F.normalize(out_bar, dim=1))
    # loss = F.l1_loss(F.normalize(out, dim=1), F.normalize(out_bar, dim=1))

    return loss

def mme_loss(out):
    prob  = F.softmax(out, dim=1)   
    loss = 0.1 * torch.mean(torch.sum(prob * (torch.log(prob + 1e-5)), 1))

    return loss

def ent_loss(out):
    prob  = F.softmax(out, dim=1)   
    loss = -0.1 * torch.mean(torch.sum(prob * (torch.log(prob + 1e-5)), 1))

    return loss


