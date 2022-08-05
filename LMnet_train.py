from dataloaders.LEMnetDataSet import LEMDataset
from dataloaders.LCMnetDataset import LCMDataset
from models.LMnet import LMnet, lmnetloss

import argparse
import numpy as np
import pandas as pd
import torch

from torch.utils.data import DataLoader
import os

from utils.util import printProgressBar

MODELS = ['LEMnet','LCMnet']
parser = argparse.ArgumentParser(description='VPSNet')
parser.add_argument('--data', metavar='DIR', help='path to dataset')
parser.add_argument('--model', type=str,metavar='model',choices=MODELS, default='LEMnet')
parser.add_argument('--model_path', default='', type=str, metavar='PATH',
                    help='Path to exisiting model')
parser.add_argument('--measure_file_test',type=str,metavar='PATH',help='path to file with entropy or classication performance data')
parser.add_argument('--measure_file_train',type=str,metavar='PATH',help='path to file with entropy or classication performance data')
parser.add_argument('--epochs', default=30, type=int, metavar='N', help='number of total epochs to run (default: 50)')
parser.add_argument('--batch_size', default=4, type=int, metavar='N', help='Batch size, default: 5')
parser.add_argument('--view_num', default=15, type=int, metavar='N', help='Number of views to predict')
parser.add_argument('--class_num', default=40, type=int, metavar='N', help='Number of classes in dataset')
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                    metavar='LR', help='initial learning rate (default: 0.001)')
parser.add_argument('--lr_decay_freq', default=5, type=float,
                    metavar='W', help='learning rate decay (default: 5)')
parser.add_argument('--lr_decay', default=0.1, type=float,
                    metavar='W', help='learning rate decay (default: 0.5)')
parser.add_argument('--verbose', dest='verbose', action='store_true', help='Print progressbar')

args = parser.parse_args()

NUM_EPOCHS = args.epochs
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('Running {} on {}'.format(args.model,device))

def load_model(model_path):

    model = LMnet(local_views=15)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_decay_freq, gamma=args.lr_decay)
    model.to(device)

    # Load checkpoint.
    print('\n==> Loading checkpoint..')
    assert os.path.isfile(model_path), 'Error: no checkpoint file found!'

    checkpoint = torch.load(model_path, map_location=torch.device(device))
    start_epoch = checkpoint['epoch']
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    
    scheduler.load_state_dict(checkpoint['scheduler'].state_dict())
    
    
    return model, optimizer, start_epoch, scheduler

def save_checkpoint(state, name, loc='model_saves', extension='.pth.tar'):
    filepath = os.path.join(loc,'{}{}'.format(name,extension))
    print('\tSaving checkpoint - {}'.format(name))
    torch.save(state, filepath)

if args.model_path:
    print('INFO: Resuming with existing model')
    model,optimizer,start_epoch,scheduler = load_model(args.model_path)
else:
    
    model = LMnet(local_views=15)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer,  step_size=args.lr_decay_freq, gamma=args.lr_decay)
    start_epoch = 0
    
measure_file_train = pd.read_csv(args.measure_file_train)
measure_file_test = pd.read_csv(args.measure_file_test)

print("INFO: Loading dataset")
if model == 'LEMnet':
    dset_train =LEMDataset(args.data,'train',measure_file_train)
    dset_test = LEMDataset(args.data,'test',measure_file_test)
else:
    dset_train =LCMDataset(args.data,'train',measure_file_train)
    dset_test = LCMDataset(args.data,'test',measure_file_test)

print('INFO: training dataset size: {}'.format(len(dset_train)))
print('INFO: test dataset size: {}'.format(len(dset_test)))

train_loader = DataLoader(dset_train, batch_size=args.batch_size, shuffle=True,drop_last=True)
test_loader = DataLoader(dset_test, batch_size=1, shuffle=False,drop_last=True)

def train(epoch):
    train_size = len(train_loader)
    losses = []

    for i, (part_pc,target_map,target_view,label,obj_ind) in enumerate(train_loader):

        part_pc, target_map, target_view = \
            part_pc.type(torch.FloatTensor),target_map.type(torch.FloatTensor), \
            target_view.type(torch.LongTensor)

        part_pc, target_map,target_view = \
            part_pc.to(device), target_map.to(device), target_view.to(device)
        part_pc = part_pc.permute(0,2,1)

        optimizer.zero_grad()
        
        output,  matrix3x3, matrix64x64, feature = model(part_pc)
        loss = lmnetloss(output,target_map,matrix3x3,matrix64x64)

        losses.append(loss.item())
        
        loss.backward()
        optimizer.step()
        if args.verbose:
            printProgressBar(i+1,train_size,prefix='loss: {:.4f}'.format(loss))
        
    print('DEBUG: Saving data')
    train_results = {'epoch':[epoch],'loss':[losses]}
    if epoch == 0:
        pd.DataFrame(train_results).to_csv(os.path.join(args.data,'train_results_{}_M{}.csv'.format(args.model,args.class_num)),index=False)
    else:
        pd.DataFrame(train_results).to_csv(os.path.join(args.data,'train_results_{}_M{}.csv'.format(args.model,args.class_num)),index=False, mode='a', header=False)
    return np.mean(losses)

def eval(epoch):
    test_size = len(test_loader)

    results = {'class': [],'obj_ind':[],'view':[],'loss':[]}
    with torch.no_grad():
        for i, (part_pc,target_map,target_view,label,obj_ind) in enumerate(test_loader):
            part_pc, target_map, target_view = \
                part_pc.type(torch.FloatTensor),target_map.type(torch.FloatTensor), \
                target_view.type(torch.LongTensor)

            part_pc, target_map,target_view = \
                part_pc.to(device), target_map.to(device), target_view.to(device)
            part_pc = part_pc.permute(0,2,1)
            
            output,  matrix3x3, matrix64x64, feature = model(part_pc)
            
            loss = lmnetloss(output,target_map,matrix3x3,matrix64x64)

            if len(label) == 1: # if test batch size = 1
                results['class'].append(label[0])
                results['obj_ind'].append(obj_ind[0])
                results['view'].append(target_view.cpu().numpy()[0])
                results['loss'].append(loss.item())
            else:
                results['class'] = np.concatenate([results['class'],label])
                results['obj_ind'] = np.concatenate([results['obj_ind'],np.asarray(obj_ind)])
                results['view'] = np.concatenate([results['view'],target_view.cpu().numpy()])
                results['loss'] = np.concatenate([results['loss'],[loss.item() for i in range(args.batch_size)]])

            if args.verbose:
                printProgressBar(i,test_size,prefix='loss: {:.4f}'.format(loss.item()))

    if args.verbose:
        print("Avg loss: {}".format(np.mean(results['loss'])))
    if epoch == 0:
        pd.DataFrame(results).to_csv(os.path.join(args.data,'test_results_{}_M{}.csv'.format(args.model,args.class_num)),index=False)
    else:
        pd.DataFrame(results).to_csv(os.path.join(args.data,'test_results_{}_M{}.csv'.format(args.model,args.class_num)),index=False, mode='a', header=False)

    return np.mean(results['loss'])



best_loss = 1000
for epoch in range(start_epoch,NUM_EPOCHS):
    print("Epoch [{}/{}]".format(epoch,NUM_EPOCHS))
    print("Training")
    model.train()
    
    avg_loss_train = train(epoch)

    scheduler.step()
    print("Evaluation")
    model.eval()
    avg_loss_test = eval(epoch)
    if avg_loss_test < best_loss:
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'average_loss': avg_loss_test,
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler,
        }, name='{}{}_40_entropy'.format(args.model,args.class_num), loc='model_saves')


    

