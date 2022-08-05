import enum
from typing import Sequence
from torch.utils import data
from torch.utils.data.dataset import Dataset
import os
from PIL import Image
import random
import numpy as np
def string_to_tuple(string):
    string = string.replace('(',"")
    string = string.replace(')',"")
    string = string.split(',')
    string = [int(value) for value in string]
    # print(tuple(string))
    return tuple(string)

def string_to_array(string,obj):
    string = string.replace(',)',")")
    string = string.replace('(',"")
    string = string.replace(')',"")
    string = string.replace('[ ',"")
    string = string.replace('] ',"")
    string = string.replace('  '," ")
    string = string.replace('[',"")
    string = string.replace(']',"") 
    string = string.split(',')
    
    try:
        string = [int(value) for value in string]
    except:
        print(string)
        print(obj)
        exit()
    # print(tuple(string))
    return string

class SingleViewDataSet(Dataset):

    def find_classes(self, dir):
        classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}

        return classes, class_to_idx

    def __init__(self, root, data_type, transform=None,classes='all'):
        print("creating " + str(data_type) + " dataset")
        self.x = []
        self.y = []
        self.root = root

        if classes == 'all':
            self.classes, self.class_to_idx = self.find_classes(os.path.join(root,data_type))
        else: 
            classes.sort()
            self.classes = classes
            self.class_to_idx = {classes[i]: i for i in range(len(classes))}

        self.transform = transform

        # root / <train/test> / <label> / <view>.png
        for label in os.listdir(os.path.join(root,data_type)): # Label
            if label in self.classes:
                if os.path.isdir(os.path.join(root,data_type,label)):
                    # print("creating dataset for " + label)
                    files = os.listdir(os.path.join(root,data_type,label))
                    files.sort()
                    for file in files:
                        self.x.append(os.path.join(root,data_type,label,file))
                        self.y.append(self.class_to_idx[label])
                 

    # Override to give PyTorch access to any image on the dataset
    def __getitem__(self, index):
        original_input = self.x[index]

        im = Image.open(original_input)
        im = im.convert('RGB')
        if self.transform is not None:
            im = self.transform(im)

        return im, self.y[index]

    # Override to give PyTorch size of dataset
    def __len__(self):
        return len(self.x)

class SingleViewDataSetMeta(Dataset):

    def find_classes(self, dir):
        classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}

        return classes, class_to_idx

    def __init__(self, root, data_type, transform=None):
        print("creating " + str(data_type) + " dataset")
        self.x = []
        self.y = []
        self.obj_inds = []
        self.views = []
        self.paths = []
        self.root = root

        self.classes, self.class_to_idx = self.find_classes(os.path.join(root,data_type))

        self.transform = transform
        self.h_step_angle = 18
        self.v_step_angle = 45

        # root / <train/test> / <label> / <view>.png
        for label in os.listdir(os.path.join(root,data_type)): # Label
            if os.path.isdir(os.path.join(root,data_type,label)):
                # print("creating dataset for " + label)
                files = os.listdir(os.path.join(root,data_type,label))
                files.sort()
                for file in files:
                    self.x.append(os.path.join(root,data_type,label,file))
                    self.y.append(self.class_to_idx[label])
                    obj_ind = file[0:-4].split('_')[-4]
                    self.obj_inds.append(obj_ind)
                    h_angle = int(file[:-4].split('_')[-2])
                    v_angle = int(file[:-4].split('_')[-1])
                    hor_split_n = h_angle/self.h_step_angle
                    ver_split_n = v_angle/self.v_step_angle
                    self.views.append(int(hor_split_n*8 + ver_split_n))

                 

    # Override to give PyTorch access to any image on the dataset
    def __getitem__(self, index):
        original_input = self.x[index]

        im = Image.open(original_input)
        im = im.convert('RGB')
        if self.transform is not None:
            im = self.transform(im)

        return im, self.y[index],self.obj_inds[index],self.views[index], original_input

    # Override to give PyTorch size of dataset
    def __len__(self):
        return len(self.x)
