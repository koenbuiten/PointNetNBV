import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

import torchvision.transforms as transforms

import argparse
import numpy as np
import time
import os

from models.resnet import *
from models.mvcnn import *
from dataloaders.singleViewDataset import SingleViewDataSet
import pandas as pd

parser = argparse.ArgumentParser(description='MVCNN-PyTorch')
parser.add_argument('--data', metavar='DIR', help='path to dataset')
parser.add_argument('--depth', choices=[18, 34, 50, 101, 152], type=int, metavar='N', default=18, help='resnet depth (default: resnet18)')

parser.add_argument('--model_suffix', type=str, default='default', help='Suffix of model name')
parser.add_argument('--epochs', default=100, type=int, metavar='N', help='number of total epochs to run (default: 100)')
parser.add_argument('-b', '--batch_size', default=4, type=int,
                    metavar='N', help='mini-batch size (default: 4)')
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                    metavar='LR', help='initial learning rate (default: 0.001)')
parser.add_argument('--momentum', default=0.9, type=float,
                    metavar='MOMENTUM', help='Momentum value')
parser.add_argument('--lr_decay_freq', default=5, type=float,
                    metavar='W', help='learning rate decay (default: 5)')
parser.add_argument('--lr_decay', default=0.5, type=float,
                    metavar='W', help='learning rate decay (default: 0.5)')
parser.add_argument('--print-freq', '-p', default=40, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--view_num', default=40, type=int,
                    metavar='N', help='Number of views for each object')
parser.add_argument('-r', '--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--num_classes',type=str,choices=['10','4','40','MSO'],default=40,help="Number of classes, ModelNet10, ModelNet40 and MSO (Medium Sized Object)")
parser.add_argument('--optimizer',choices=['Adam','SGD'],default='Adam')
parser.add_argument('--pooling',choices=[None,'maxpool','majority'],default=None,type=str,help='Multi view pooling method')
parser.add_argument('--pretrained', dest='pretrained', action='store_true', help='use pre-trained model')
parser.add_argument('--multi_view', dest='multi_view', action='store_true', help='Use multi view dataset')
args = parser.parse_args()
print('Loading data')

transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
])

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


if args.num_classes == '10':
    classes = ['bathtub','bed','chair','desk','dresser','monitor','night_stand','sofa','table','toilet']
else:
    classes = 'all'

filename = 'resnet_{}_singleview_{}'.format(args.depth,args.model_suffix)
dset_train = SingleViewDataSet(args.data, 'train', transform=transform,classes=classes)
dset_val = SingleViewDataSet(args.data, 'test', transform=transform,classes=classes)

print('Training dataset size: ', len(dset_train))
print('Test dataset size: ', len(dset_val))
classes = dset_train.classes
print(len(classes), classes)

train_loader = DataLoader(dset_train, batch_size=args.batch_size, shuffle=True, num_workers=2, drop_last=True)
val_loader = DataLoader(dset_val, batch_size=args.batch_size, num_workers=2)

if args.depth == 18:
    model = resnet18(pretrained=args.pretrained, num_classes=len(classes),pooling=args.pooling)
elif args.depth == 34:
    model = resnet34(pretrained=args.pretrained, num_classes=len(classes))
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
n_epochs = args.epochs
criterion = nn.CrossEntropyLoss()
if args.optimizer == "Adam":
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
elif args.optimizer == 'SGD':
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_decay_freq, gamma=args.lr_decay)

best_acc = -0.1
best_loss = 0.0
start_epoch = 0

def save_checkpoint(state, loc='checkpoint'):
    filepath = os.path.join(loc, filename+'.pth.tar')
    print('\tSaving checkpoint - {}'.format(filename))
    torch.save(state, filepath)


# Helper functions
def load_checkpoint(model_path):
    global best_acc, start_epoch
    # Load checkpoint.
    print('\n==> Loading checkpoint..')
    assert os.path.isfile(model_path), 'Error: no checkpoint file found!'

    checkpoint = torch.load(model_path)
    best_acc = checkpoint['best_acc']
    start_epoch = checkpoint['epoch']
    scheduler.load_state_dict(checkpoint['scheduler'].state_dict())
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])


def train():
    train_size = len(train_loader)
    losses = []
    for i, (inputs, targets) in enumerate(train_loader):

        # Convert from list of 3D to 4D
        if args.multi_view:
            inputs = np.stack(inputs, axis=1)
            inputs = torch.from_numpy(inputs)
        
        inputs, targets = inputs.to(device), targets.to(device)   
        # compute output
        
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        losses.append(loss.item())
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return losses


# Validation and Testing
def eval(data_loader):

    # Eval
    total = 0.0
    correct = 0.0

    total_loss = 0.0
    n = 0

    for i, (inputs, targets) in enumerate(data_loader):
        with torch.no_grad():
            # Convert from list of 3D to 4D        
            if args.multi_view:
                inputs = np.stack(inputs, axis=1)
                inputs = torch.from_numpy(inputs)

            inputs, targets = inputs.to(device), targets.to(device)
            
            # compute output
            outputs = model(inputs)

            loss = criterion(outputs, targets)

            total_loss += loss
            n += 1

            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted.cpu() == targets.cpu()).sum()

    avg_test_acc = 100 * correct / total
    avg_loss = total_loss / n

    return avg_test_acc, avg_loss


# Training / Eval loop
if args.resume:
    load_checkpoint(args.resume)

if os.path.isfile('train_data/{}.csv'.format(filename)):
    train_data = pd.read_csv('train_data/{}.csv'.format(filename))
else:
    train_data = pd.DataFrame({})


for epoch in range(start_epoch, n_epochs):
    print('\n-----------------------------------')
    print('Epoch: [%d/%d]' % (epoch+1, n_epochs))
    start = time.time()

    model.train()
    losses = train()
    print('Training time taken: %.2f sec.' % (time.time() - start))

    train_data[epoch]= losses
    if not os.path.exists('train_data'):
        os.mkdir('train_data')
    pd.DataFrame(train_data).to_csv('train_data/{}.csv'.format(filename),index=False)

    scheduler.step()
    model.eval()
    avg_test_acc, avg_loss = eval(val_loader)

    print('\nEvaluation:')
    print('\tVal Acc: %.2f - Loss: %.4f' % (avg_test_acc.item(), avg_loss.item()))
    print('\tCurrent best val acc: %.2f' % best_acc)

    # Save model
    if avg_test_acc > best_acc:
        
        best_acc = avg_test_acc
        best_loss = avg_loss
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'acc': avg_test_acc,
            'best_acc': best_acc,
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler,
        },  loc='model_saves')
