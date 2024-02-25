import os
import cv2
import torch
import random
import os.path as osp
import numpy as np
from PIL import Image
from torch.utils import data
from torchvision import transforms
from .weak_augment import horizontal_flip, vertical_flip, rotate, resize_crop, color_swap, hsv_shift
from .rand_augment import RandAugmentMC


class SemiRS_DataSet(data.Dataset):
    def __init__(self, root, lbl_list_path, unl_list_path, is_strong=False, is_training=True, ignore_label=-1):
        self.root = root  # folder for GTA5 which contains subfolder images, labels
        self.lbl_list_path = osp.join(root, lbl_list_path)   # list of image names
        self.unl_list_path = osp.join(root, unl_list_path)   # list of image names
        self.ignore_label = ignore_label
        self.is_training = is_training
        self.is_strong = is_strong
        self.strong = RandAugmentMC(n=2, m=10, no_swap=True)
        self.paths_l = []
        self.paths_u = []
        
        f_l = open(self.lbl_list_path)            
        f_u = open(self.unl_list_path)            
        self.paths_l = f_l.readlines()    
        self.paths_u = f_u.readlines()    
        self.paths_l = [path.strip('\n') for path in self.paths_l]         
        self.paths_u = [path.strip('\n') for path in self.paths_u]         
        f_l.close()  
        f_u.close()  
        print (len(self.paths_l), len(self.paths_u))

        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])


    def __len__(self):
        return max(len(self.paths_l), len(self.paths_u))

    def get_image_label(self, paths, index):
        image_path, label_path = paths[index].split(' ')
        image_path = osp.join(self.root, image_path)
        label_path = osp.join(self.root, label_path.strip('\n'))
        image, label = Image.open(image_path), Image.open(label_path).convert('L')

        if self.is_training:
            # spatial transform
            image, label = horizontal_flip(image, label)
            image, label = vertical_flip(image, label)
            image, label = rotate(image, label)
            image, label = resize_crop(image, label)

        image_bar = self.strong(image)
        image_bar = self.normalize(self.to_tensor(image_bar))

        image = self.normalize(self.to_tensor(image))
        label = np.asarray(label, np.float32) / 255.0

        return image, image_bar, label

    def __getitem__(self, index):
        image_l, image_bar_l, label_l = self.get_image_label(self.paths_l, int(index/len(self.paths_l)))
        image_u, image_bar_u, label_u = self.get_image_label(self.paths_u, int(index/len(self.paths_u)))

        return image_l, image_bar_l, label_l, image_u, image_bar_u, label_u


class RS_Binary_DataSet(data.Dataset):
    def __init__(self, root, list_path, max_size=None, is_training=True, ignore_label=-1):
        self.root = root  # folder for GTA5 which contains subfolder images, labels
        self.list_path = osp.join(root, list_path)   # list of image names
        self.ignore_label = ignore_label
        self.max_size = [max_size, max_size]
        self.is_training = is_training
        self.strong = RandAugmentMC(n=2, m=10)
        self.paths = []
        
        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

        f = open(self.list_path)            
        self.paths = f.readlines()    
        self.paths = [path.strip('\n') for path in self.paths]         
        print (len(self.paths))
        f.close()  

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        image_path, label_path = self.paths[index].split(' ')
        image_path = osp.join(self.root, image_path)
        label_path = osp.join(self.root, label_path.strip('\n'))
        image, label = Image.open(image_path), Image.open(label_path).convert('L')
        max_size = image.size  if self.max_size == [None, None]  else self.max_size

        if self.is_training:
            # spatial transform
            image, label = horizontal_flip(image, label)
            image, label = vertical_flip(image, label)
            image, label = rotate(image, label)
            image, label = resize_crop(image, label, max_size)

        image_bar = self.strong(image)
        image_bar = self.normalize(self.to_tensor(image_bar))

        image = self.normalize(self.to_tensor(image))
        label = np.asarray(label, np.float32) / 255.0

        return image, image_bar, label, [image_path, label_path]
