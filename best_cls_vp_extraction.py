import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.autograd import Variable

import torchvision.transforms as transforms

import argparse
import numpy as np
import os

from models.resnet import *
from dataloaders.singleViewDataset import SingleViewDataSetMeta
import pandas as pd
from utils.util import printProgressBar

parser = argparse.ArgumentParser(description='Best vp extraction')
parser.add_argument('--data', metavar='DIR', help='path to dataset')
parser.add_argument('--depth', choices=[18, 34, 50, 101, 152], type=int, metavar='N', default=18, help='resnet depth (default: resnet18)')
parser.add_argument('--set_type',default='test')
parser.add_argument('--model_suffix', type=str, default='default', help='Suffix of model name')
parser.add_argument('-b', '--batch_size', default=4, type=int,
                    metavar='N', help='mini-batch size (default: 4)')
parser.add_argument('--view_num', default=40, type=int,
                    metavar='N', help='Number of views for each object')
parser.add_argument('-r', '--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--num_classes',type=str,choices=['10','40','MSO'],default=40,help="Number of classes, ModelNet10, ModelNet40 and MSO (Medium Sized Object)")
parser.add_argument('--pooling',choices=[None,'maxpool','majority'],default=None,type=str,help='Multi view pooling method')
parser.add_argument('--pretrained', dest='pretrained', action='store_true', help='use pre-trained model')
parser.add_argument('--verbose', dest='verbose', action='store_true', help='Print progressbar')
args = parser.parse_args()
print('Loading data')

transform = transforms.Compose([
    transforms.CenterCrop(500),
    transforms.Resize(224),
    transforms.ToTensor(),
])

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



filename = 'resnet_{}_singleview_{}'.format(args.depth,args.model_suffix)
dset_val = SingleViewDataSetMeta(args.data, args.set_type, transform=transform)

print('Dataset size: ', len(dset_val))
classes = dset_val.classes
print(len(classes), classes)

val_loader = DataLoader(dset_val, batch_size=args.batch_size, num_workers=2)

if args.depth == 18:
    model = resnet18(pretrained=args.pretrained, num_classes=len(classes),pooling=args.pooling)
elif args.depth == 34:
    model = resnet34(pretrained=True, num_classes=len(classes))
elif args.depth == 50:
    model = resnet50(pretrained=args.pretrained, num_classes=len(classes))
elif args.depth == 101:
    model = resnet101(pretrained=args.pretrained, num_classes=len(classes))
elif args.depth == 152:
    model = resnet152(pretrained=args.pretrained, num_classes=len(classes))
else:
    raise Exception('Specify number of layers for resnet in command line. --resnet N')
print('Using resnet' + str(args.depth))


model.to(device)
cudnn.benchmark = True

print('Running on ' + str(device))

# Loss and Optimizer
criterion = nn.CrossEntropyLoss()

# Helper functions
def load_checkpoint(model_path):
    global best_acc, start_epoch
    # Load checkpoint.
    print('\n==> Loading checkpoint..')
    assert os.path.isfile(model_path), 'Error: no checkpoint file found!'

    checkpoint = torch.load(model_path)
    best_acc = checkpoint['best_acc']
    start_epoch = checkpoint['epoch']
    model.load_state_dict(checkpoint['state_dict'])

model_name = 'resnet_{}_singleview_{}'.format(args.depth,args.model_suffix)

# Validation and Testing
def eval(data_loader):
    set_len = len(val_loader)
    total = 0.0
    correct = 0.0

    total_loss = 0.0
    n = 0
    results = {'gt':[],'pred':[],'obj_ind':[],'view':[],'conf':[],'correct':[],'path':[]}
    for i, (inputs, targets,obj_inds,views,paths) in enumerate(data_loader):
        with torch.no_grad():    
            inputs, targets = inputs.to(device), targets.to(device)
            
            # compute output
            outputs = model(inputs)

            loss = criterion(outputs, targets)
            total_loss += loss

            
            softmax = nn.Softmax(dim=1)
            outputs = softmax(outputs)

            conf, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            
            batch_correct = np.asarray((predicted.cpu() == targets.cpu()))
            batch_correct = [1 if cor else 0 for cor in batch_correct]
            results['correct'] = np.concatenate([results['correct'],
                batch_correct])
            results['gt'] = np.concatenate([results['gt'],
                targets.cpu().numpy()])
            results['pred'] = np.concatenate([results['pred'],
                predicted.cpu().numpy()])
            results['obj_ind'] = np.concatenate([results['obj_ind'],
                obj_inds])
            results['view'] = np.concatenate([results['view'],
                views])
            results['conf'] = np.concatenate([results['conf'],
                conf.cpu().numpy()])
            results['path'] = np.concatenate([results['path'],
                paths])
            
            correct += (predicted.cpu() == targets.cpu()).sum()
            n += 1
        if args.verbose:
            printProgressBar(i,set_len,prefix="{:.2f}".format(correct/(i*args.batch_size)))

    avg_test_acc = 100 * correct / total
    avg_loss = total_loss / n
    pd.DataFrame(results).to_csv('baseline_results/Modelnet{}_40/best_cls_vp_{}_{}.csv'.format(args.num_classes,model_name,args.set_type))
    return avg_test_acc, avg_loss

if args.resume:
    load_checkpoint(args.resume)

model.eval()
avg_test_acc, avg_loss = eval(val_loader)

print('\nEvaluation:')
print('\tVal Acc: %.2f - Loss: %.4f' % (avg_test_acc.item(), avg_loss.item()))
print('\tCurrent best val acc: %.2f' % best_acc)
