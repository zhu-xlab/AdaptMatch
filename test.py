import os
import json
import torch
import torch.nn as nn
import numpy as np
import os.path as osp
import torch.nn.functional as F
from PIL import Image
from dataloader import CreateTestDataLoader
from models import CreateModel
from torch.autograd import Variable
from options.test_options import TestOptions
from utils.metric import Evaluator
from utils import *
import cv2
import openpyxl
torch.set_num_threads(2)


def test(num_classes, test_loader, model):
    model.eval()

    metric = Evaluator(num_class=2)
    with torch.no_grad():
        for j, (test_img, _, test_lbl, _) in enumerate(test_loader):
            test_img = test_img.cuda()      # to gpu
            _, _, h, w = test_img.size()

            # test_out = model(test_img)   
            # test_out = (torch.max(test_out, dim=1)[1]!=0).long()
            test_out = model(test_img)
            test_out = (test_out > 0.5).long()
            test_out = test_out.cpu().detach().numpy()
            test_lbl = test_lbl.unsqueeze(dim=1).cpu().detach().numpy()
            metric.add_batch(test_lbl, test_out)

    acc = metric.Pixel_Accuracy()
    Precison, Recall = metric.Precison_Recall()
    acc, Precison, Recall = np.around(acc*100, 2), np.around(Precison*100, 2), np.around(Recall*100, 2)
    IoU = np.around(metric.Intersection_over_Union()*100, 2)
    F1 = np.around(metric.F1_Score()*100, 2)

    return acc, Precison, Recall, IoU, F1


if __name__ == '__main__':
    opt = TestOptions()
    args = opt.initialize()
    data_dir = os.path.join(args.data_dir, args.dataset)
    opt.print_options(args)
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

    model = CreateModel(args.model, args.num_classes).cuda()

    # resume checkpoint
    save_dir = os.path.join(args.save_dir, args.model, args.method)
    checkpoint_file = '{}_{}%.pth'.format(args.dataset.replace('INRIA/', 'INRIA_'), args.percent)
    checkpoint_file = os.path.join(save_dir, checkpoint_file)

    resume = torch.load(checkpoint_file)
    model.load_state_dict(resume['state_dict'], strict=False)
    start_iter = resume['iter']
    best_acc = resume['best_acc']
    best_IoU = resume['best_IoU']
    print ('loaded checkpoint from: {}, iter: {}, best_IoU: {} '\
            .format(checkpoint_file, start_iter, best_IoU))

    test_loader = CreateTestDataLoader(data_dir, 'lists/test.txt')
    test_acc, test_Precison, test_Recall, test_IoU, test_F1 = \
            test(args.num_classes, test_loader, model)

    print ('OA/Recall/Precision/IoU/F1: %.2f/%.2f/%.2f/%.2f/%.2f' % \
                (test_acc, test_Recall[1], test_Precison[1], test_IoU[1], test_F1[1]) )
    # IoU_F1 = []
    # for i in range(len(test_IoU)):
    #     IoU_F1.append(test_IoU[i])
    #     IoU_F1.append(test_F1[i])
    # IoU_F1.append(np.mean(test_IoU))
    # IoU_F1.append(np.mean(test_F1))
    # print (IoU_F1)

    # save to excel
    xlsx_file = "{}_{}.xlsx".format(args.model, args.method)
    try:
        workbook = openpyxl.load_workbook(xlsx_file)
    except FileNotFoundError:
        workbook = openpyxl.Workbook()
        workbook.save(xlsx_file)

    worksheet = workbook.active
    new_data = [args.percent, args.dataset, test_Recall[1], test_Precison[1], test_IoU[1], test_F1[1]]
    worksheet.append(new_data)
    workbook.save(xlsx_file)