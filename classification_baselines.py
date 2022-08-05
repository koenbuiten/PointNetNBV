import torch
from torch.utils.data import DataLoader

import torchvision.transforms as transforms

import argparse
import numpy as np
import os

from models.resnet import *
from models.mvcnn import *
from dataloaders.multiViewDataSet import MultiViewDataSet, DescriptorBasedMultiViewDataSet, MultiBestViewDataset
from dataloaders.singleViewDataset import SingleViewDataSet
from utils.util import printProgressBar
import pandas as pd

def load_data(data_path,set_view_num=40,sampling_method='random',view_num=1,sampling_data=''):
    print('==> Loading data')
    transform = transforms.Compose([
        transforms.CenterCrop(500),
        transforms.Resize(224),
        transforms.ToTensor(),
    ])
    # Load dataset
  
    if sampling_method == 'random':
        dset_val = MultiViewDataSet(data_path, 'test',view_num=view_num,
        set_view_num=set_view_num, transform=transform,local=True, random_vps=True)
    elif sampling_method == 'normal':
        dset_val = SingleViewDataSet(data_path,'test',transform)
    elif sampling_method == 'best':
        descriptor_data = pd.read_csv(sampling_data)
        dset_val = MultiBestViewDataset(data_path,'test',descriptor_data,set_view_num,view_num,transform)
    else:
        descriptor_data = pd.read_csv(sampling_data)
        dset_val = DescriptorBasedMultiViewDataSet(data_path, 'test',sampling_method,
        descriptor_data,set_view_num=set_view_num,view_num=view_num,selection_method=sampling_method, transform=transform)    

    val_loader = DataLoader(dset_val,batch_size=1, num_workers=0)

    return val_loader

def load_model(model_path, depth,num_classes,device): 
    
    if depth == 18:
        model = resnet18(pretrained=True, num_classes=num_classes,pooling='majority')
    else:
        model = resnet34(pretrained=True, num_classes=num_classes,pooling='majority')

    print('\n==> Loading checkpoint on {}..'.format(device))
    assert os.path.isfile(model_path), 'Error: no checkpoint file found!'

    checkpoint = torch.load(model_path)
    best_acc = checkpoint['best_acc']
    print("INFO: Best acc: {}".format(best_acc))
    model.load_state_dict(checkpoint['state_dict'])
    model.to(device)
    
    return model

def get_local_views(start_view):

    full_map = np.asarray(range(40))
    full_map = np.reshape(full_map,(5,8))
    start_view_loc = np.where(full_map == start_view)
    shift = 3-start_view_loc[1]
    full_map = np.roll(full_map,shift,axis=1)
    start_view_loc = np.where(full_map == start_view)
    local_map = full_map[:,start_view_loc[1][0]-1:start_view_loc[1][0]+2]

    local_map = local_map.flatten()
    
    np.delete(local_map,np.where(local_map == 9))
    return local_map

# Validation and Testing
def eval(model,data_loader,sampling_method='normal',verbose=False,device='cpu',view_num=5):
    # Eval
    total = 0.0
    correct = 0.0
    predictions = []
    ground_truth = []

    for i, (inputs, target) in enumerate(data_loader):
        with torch.no_grad():
            # Swap batch and in channels
            inputs = np.stack(inputs, axis=1)

            inputs = torch.from_numpy(inputs)
            inputs = inputs.to(device)
            outputs = model(inputs)
            
            _, predicted = torch.max(outputs,1)
            predictions.append(predicted.cpu().numpy())
            ground_truth.append(target.numpy())
            correct += int(np.sum(predicted.cpu().numpy() == target.numpy()))
    
            total += 1
        if verbose and correct != 0:
            # print(len(data_loader))
            printProgressBar(i,len(data_loader),prefix='{:.2f}%'.format(correct/total*100))

    return predictions, ground_truth
    # , avg_loss

def classification_test(data_path,model_path,set_view_num,model_depth,view_num,sampling_data_path,sampling_method,verbose=False,out_path=None):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    classes = [d for d in os.listdir(os.path.join(data_path, 'test')) if os.path.isdir(os.path.join(data_path, 'test',d))]
    classes.sort()
    model = load_model(model_path,model_depth,len(classes),device=device)
    model.eval()
    results = []
    if sampling_method == 'random':
        # Do 5 runs to make up for random high accuracy
        num_runs = 10
        if verbose:
            print("\n==> {} evaluation runs with random viewpoint selection".format(num_runs))
        for run in range(num_runs):
            data_loader = load_data(data_path,set_view_num,sampling_method,view_num,sampling_data_path)
            pred, gt = eval(model,data_loader,sampling_method,verbose=verbose,device=device,view_num=view_num)
            results.append(pd.DataFrame({'gt':gt,'pred':pred}))
        # results = pd.concat(results,axis=1,keys=['run {}'.format(i) for i in range(5)])
        results = pd.concat(results,ignore_index=True)

    elif sampling_method == 'best':
        data_loader = load_data(data_path,set_view_num,sampling_method,view_num,sampling_data_path)
        pred, gt = eval(model,data_loader,sampling_method,verbose=verbose,device=device)
        results=pd.DataFrame({'gt':gt,'pred':pred})
    else:
        data_loader = load_data(data_path,set_view_num,sampling_method,view_num,sampling_data_path)
        pred, gt = eval(model,data_loader,sampling_method,verbose=verbose,device=device)
        results=pd.DataFrame({'gt':gt,'pred':pred})
    
    if out_path:
        if not os.path.isdir(out_path):
            os.mkdir(out_path)
        save_path = os.path.join(out_path,'Modelnet{}_{}_Resnet{}'.format(len(classes),set_view_num,model_depth))
        if not os.path.isdir(save_path):
            os.mkdir(os.path.join(save_path))
        results.to_csv(os.path.join(save_path,'predictions_{}_{}_views.csv'
            .format(sampling_method,view_num)))
    return results

    
