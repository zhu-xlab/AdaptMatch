import os
import torch
import random
import argparse
import numpy as np
import torch.nn.functional as F
from options.train_options import TrainOptions
from dataloader import CreateDataLoader
from models import CreateModel
from evaluate import evaluate
from losses import *
from utils.util import Timer, AverageMeter, accumulate_distribution, calc_threshold, cutmix


torch.set_num_threads(4)



def main():
    _t = {'epoch_time': Timer()}
    _t['epoch_time'].tic()

    # get args parameters
    cls_fore, cls_back = 1, 0
    opt = TrainOptions()
    args = opt.initialize()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

    args.percent = str(args.percent)
    args.data_dir = args.data_dir.replace('dataset', args.dataset)
    args.train_lbl_list = args.train_lbl_list.replace('percent', args.percent)
    args.train_unl_list = args.train_unl_list.replace('percent', args.percent)
    opt.print_options(args)

    # create model
    torch.cuda.manual_seed(args.seed)
    model = CreateModel(args.model, args.num_classes).cuda()
    optimizer = torch.optim.Adam(model.optim_parameters(args), lr=args.learning_rate)

    # resume checkpoint
    args.save_dir = os.path.join(args.save_dir, args.model)
    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)
    save_dir = os.path.join(args.save_dir, args.method)
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    checkpoint_file = '{}_{}%.pth'.format(args.dataset.replace('/', '_'), args.percent)
    checkpoint_file = os.path.join(save_dir, checkpoint_file)
    log_file = checkpoint_file.replace('pth', 'txt')

    start_iter = 0
    if os.path.exists(checkpoint_file) and args.restore:
        resume = torch.load(checkpoint_file)
        model.load_state_dict(resume['state_dict'])
        start_iter = resume['iter']
        best_acc = resume['best_acc']
        best_IoU = resume['best_IoU']
        print ('loading checkpoint from: {}, iter: {}, best_IoU: {} '\
                .format(checkpoint_file, start_iter, best_IoU))
    else:
        best_acc = 0
        best_IoU = 0
        start_iter = 0

    # build loader
    print ('Loading dataset ...')
    lbl_loader, unl_loader, val_loader = CreateDataLoader(args)
    lbl_iter, unl_iter = iter(lbl_loader), iter(unl_loader)

    # initialize class-wise weight
    dist_l = {cls_fore: [], cls_back: []}
    dist_u = {cls_fore: [], cls_back: []}
    cls_thr_l = {cls_fore: 0, cls_back: 0}
    cls_thr_u = {cls_fore: 0, cls_back: 0}
    # print ('class_weight: ', cls_wgt)

    jaccard_loss = JaccardLoss(args.num_classes)
    losses = AverageMeter()
    losses_s = AverageMeter()
    losses_u = AverageMeter()
    losses_jaccard = AverageMeter()

    #-------------------------------------------------------------------#
    # Training, K-means, and Evaluation
    print ('Starting training ...')
    for i in range(start_iter, args.num_iters):
        # adjust model status
        model.adjust_learning_rate(args, optimizer, i)    

        # i training
        optimizer.zero_grad()                                                 
        model.train()
        if i % len(lbl_loader) == 0:
            lbl_iter = iter(lbl_loader)
        if i % len(unl_loader) == 0:
            unl_iter = iter(unl_loader)
        img_l, img_l_bar, gt_l, _ = lbl_iter.next()                      # new batch target
        img_u, img_u_bar, gt_u, _ = unl_iter.next()                      # new batch target
        img_l, img_l_bar, gt_l = img_l.cuda().detach(), img_l_bar.cuda().detach(), gt_l.unsqueeze(dim=1).cuda() 
        img_u, img_u_bar, gt_u = img_u.cuda().detach(), img_u_bar.cuda().detach(), gt_u.unsqueeze(dim=1).cuda() 
        b, _, h, w = img_l.size()

        # feature and prediction extraction
        pred_l = model(img_l)   
        with torch.no_grad():
            pred_u = model(img_u)   
        if 'Fixmatch' in args.method or 'Adaptmatch' in args.method:
            if 'DeepLab_V3plus' in args.model:
                feat_u_bar, feat_u_low_bar = model.get_features(img_u_bar)   
                pred_u_bar = model.get_predicts(F.dropout(feat_u_bar, p=0.5), F.dropout(feat_u_low_bar, p=0.5), [h,w])
            else:            
                feat_u_bar = model.get_features(img_u_bar)   
                pred_u_bar = model.get_predicts(F.dropout(feat_u_bar, p=0.5), [h,w])

        # sup loss
        loss_s = F.binary_cross_entropy(pred_l, gt_l)
        loss_jaccard = jaccard_loss(pred_l, gt_l)

        # unsuperivised loss
        if args.method == 'Sup':
            pseudo_gt = (pred_u > 0.5).float().detach()

        if 'Fixmatch' in args.method:
            thr = 0.95
            thr_fore = thr
            thr_back = 1 - thr
            mask = torch.logical_or(pred_u>thr_fore, pred_u<thr_back)
            pseudo_gt = (pred_u > 0.5).float().detach()
            loss_u = (F.binary_cross_entropy(pred_u_bar, pseudo_gt, reduction='none')*mask).mean() 

        if 'Adaptmatch' in args.method:
            thr_fore = np.abs(cls_fore - cls_thr_l[cls_fore])
            thr_back = np.abs(cls_back - cls_thr_l[cls_back])
            mask = torch.logical_or(pred_u>thr_fore, pred_u<thr_back)
            pseudo_gt = (pred_u > 0.5).float().detach()
            loss_u = (F.binary_cross_entropy(pred_u_bar, pseudo_gt, reduction='none')*mask).mean() 

        # overall loss
        loss = loss_s + loss_jaccard
        if 'Fixmatch' in args.method or 'Adaptmatch' in args.method:
            loss += loss_u  

        loss.backward()
        optimizer.step()

        # record loss
        losses.update(loss.item(), b)  
        losses_s.update(loss.item(), b)  
        losses_jaccard.update(loss_jaccard.item(), b)  
        if 'Fixmatch' in args.method or 'Adaptmatch' in args.method:
            losses_u.update(loss_u.item(), b)  

        # accumulate weight   
        list_length = 100
        with torch.no_grad():
            dist_l = accumulate_distribution(list_length, dist_l, pred_l.detach(), gt_l, cls_fore, cls_back)
            dist_u = accumulate_distribution(list_length*5, dist_l, pred_u.detach(), pseudo_gt, cls_fore, cls_back)

        # calculate thr: labeled & unlabeled
        if i > list_length:
            cls_thr_l[cls_fore] = calc_threshold(dist_l[cls_fore]+dist_u[cls_fore])
            cls_thr_l[cls_back] = calc_threshold(dist_l[cls_back]+dist_u[cls_fore])
                
        # print info
        if (i+1) % args.print_freq == 0:
            _t['epoch_time'].toc(average=False)
            if args.method=='Sup':
                print('[Iter: %d-%d][loss_s %.4f][loss_jaccard %.4f][lr %.4f][%.2fs]' % \
                        (i+1, args.num_iters, losses_s.avg, losses_jaccard.avg, 
                            optimizer.param_groups[0]['lr']*1e4, _t['epoch_time'].diff) )
            else:
                print('[Iter: %d-%d][loss_s %.4f][loss_jaccard %.4f][loss_u %.4f][lr %.4f][%.2fs]' % \
                        (i+1, args.num_iters, losses_s.avg, losses_jaccard.avg, losses_u.avg,
                            optimizer.param_groups[0]['lr']*1e4, _t['epoch_time'].diff) )

       # evaluation and save
        # reset the weight
        if (i+1) % args.eval_freq == 0:
            # evaluate 
            val_acc, val_IoU, val_ratio = evaluate(args.num_classes, val_loader, model)

            val_IoU = val_IoU[cls_fore]
            print ('Val--  IoU: {} cls_thr_l: {}'\
                    .format(val_IoU, cls_thr_l))

            # record class thresholds    
            with open(log_file, 'a') as file:
                line = str(cls_thr_l[cls_fore]) + ' ' + str(cls_thr_l[cls_back]) + '\n' 
                file.write(line)

            # save checkpoint
            if best_IoU < val_IoU:
                best_acc = val_acc
                best_IoU = val_IoU
                state = {
                    'iter': i,
                    'best_acc': best_acc,
                    'best_IoU': best_IoU,
                    'state_dict': model.state_dict(),
                }
                torch.save(state, checkpoint_file)
                print ('taking snapshot ...')
                print ()

            record_dir = './records'
            if not os.path.exists(record_dir):
                os.mkdir(record_dir)
            record_file = os.path.join(record_dir, args.method + '_' \
                        + args.dataset.replace('INRIA/', 'INRIA_') \
                        + '_' + args.percent + '.txt')
            with open(record_file, 'a') as f:
                f.write(str(val_IoU) + ' ' + str(val_ratio) + '\n')

            # reset loss
            losses.reset()
            losses_s.reset()
            losses_u.reset()
            losses_jaccard.reset()
            
            _t['epoch_time'].tic()

if __name__ == '__main__':
    main()

