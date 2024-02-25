import numpy as np
from torch.utils import data
from dataloader.dataset_SemiRS import SemiRS_DataSet, RS_Binary_DataSet


def CreateDataLoader(args):
    train_lbl_dataset = RS_Binary_DataSet(args.data_dir, 
                                          args.train_lbl_list,
                                          max_size=args.max_img_size,  
                                          is_training=True)    
    train_lbl_loader = data.DataLoader(train_lbl_dataset, 
                                       batch_size=args.batch_size,
                                       shuffle=True, 
                                       drop_last=False,
                                       num_workers=args.num_workers, 
                                       pin_memory=False)    
    train_unl_dataset = RS_Binary_DataSet(args.data_dir, 
                                          args.train_unl_list, 
                                          max_size=args.max_img_size,  
                                          is_training=True)    
    train_unl_loader = data.DataLoader(train_unl_dataset, 
                                       batch_size=args.batch_size,
                                       shuffle=True, 
                                       drop_last=False,
                                       num_workers=args.num_workers, 
                                       pin_memory=False)    
    val_dataset = RS_Binary_DataSet(args.data_dir, 
                                    args.val_list, 
                                    is_training=False)    
    val_loader = data.DataLoader(val_dataset, 
                                 batch_size=args.batch_size*3,
                                 shuffle=True, 
                                 drop_last=False,
                                 num_workers=args.num_workers, 
                                 pin_memory=False)    

    return train_lbl_loader, train_unl_loader, val_loader


def CreateSupDataLoader(args):
    train_lbl_dataset = RS_Binary_DataSet(args.data_dir, 
                                          args.train_lbl_list.replace('_labeled', ''),
                                          max_size=args.max_img_size,  
                                          is_training=True)    
    train_lbl_loader = data.DataLoader(train_lbl_dataset, 
                                       batch_size=args.batch_size,
                                       shuffle=True, 
                                       drop_last=False,
                                       num_workers=args.num_workers, 
                                       pin_memory=False)    
    val_dataset = RS_Binary_DataSet(args.data_dir, 
                                    args.val_list, 
                                    is_training=False)    
    val_loader = data.DataLoader(val_dataset, 
                                 batch_size=args.batch_size*3,
                                 shuffle=True, 
                                 drop_last=False,
                                 num_workers=args.num_workers, 
                                 pin_memory=False)    

    return train_lbl_loader, val_loader


def CreateTestDataLoader(dir_test, list_test):
    test_dataset = RS_Binary_DataSet(dir_test, list_test, is_training=False)
    test_dataloader = data.DataLoader(test_dataset,
                                      batch_size=1, 
                                      shuffle=False, 
                                      drop_last=False,
                                      pin_memory=False)

    return test_dataloader


