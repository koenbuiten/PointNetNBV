import numpy as np
import argparse
import os

import torch
import torchvision.transforms as transforms
import open3d as o3d
from PIL import Image
import pandas as pd

from scipy.special import softmax
from models.resnet import resnet34
from models.NBV import NBV, baselineNBV
from utils.util import printProgressBar
import time


MAP_TYPES = ['entropy','cls_performance']
NBV_TYPES = ['map','random','furthest','unidirectional']

THRESHOLDS = [0.85,0.9,0.95,0.97,0.99]
MAX_VIEWS = 5

parser = argparse.ArgumentParser(description='Test NBV map methods methods')
parser.add_argument('--data', metavar='DIR', help='path to dataset')
parser.add_argument('--nbv_type', type=str,metavar='model',choices=NBV_TYPES, default='map')
parser.add_argument('--map_type', type=str,metavar='model',choices=MAP_TYPES, default='entropy')
parser.add_argument('--cls_model_path', default='', type=str, metavar='PATH',
                    help='Path to exisiting classification model')
parser.add_argument('--nbv_model_path', default='', type=str, metavar='PATH',
                    help='Path to exisiting nbv model')
parser.add_argument('--num_classes_cls', type=int,default=40)
parser.add_argument('--num_classes_NBV_model', type=int,default=40)
parser.add_argument('--verbose', dest='verbose', action='store_true', help='Print progressbar')

args = parser.parse_args()

if args.nbv_type == 'map':
    print("Running test for {} map {}_40 NBV on the {}_40 dataset".format(args.map_type,args.num_classes_cls,args.num_classes_NBV_model))
else:
    print("Running test for {}  NBV on the {}_40 dataset".format(args.nbv_type,args.num_classes_cls ))
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.CenterCrop(500),
    transforms.Resize(224),
    transforms.ToTensor(),
])

def load_model(model_path,num_classes,device):

    model = resnet34(pretrained=True, num_classes=num_classes,pooling=None)

    print('\n==> Loading cls model on {}..'.format(device))
    assert os.path.isfile(model_path), 'Error: no checkpoint file found!'

    checkpoint = torch.load(model_path)
    best_acc = checkpoint['best_acc']
    print("INFO: Acc: {}".format(best_acc))
    model.load_state_dict(checkpoint['state_dict'])
    model.to(device)
    
    return model

cls_model = load_model(args.cls_model_path,args.num_classes_cls,device)
cls_model.eval()

if args.nbv_type == 'map':
    nbv_model = NBV(args.map_type,args.nbv_model_path,device=device)
else:
    nbv_model = baselineNBV(args.nbv_type)

labels = [dir for dir in os.listdir(os.path.join(args.data,'view_pc','test'))]
labels.sort()
label_to_idx = {labels[i]:i for i in range(len(labels))}

total_num_objects = int(np.sum([len(os.listdir(os.path.join(args.data,'view_pc','test',label))) for label in labels])/40)
np.random.seed(46)
start = time.time()
j=0 
for threshold in THRESHOLDS:
    results = {'pred':[],'gt':[],'label':[],'obj_ind':[],'correct':[],'views':[],'confidence':[]}
    for run in range(10):
        i = 0
        start_views = np.random.randint(0,40,total_num_objects)
        correct = 0
        for label in labels:
            files = [file for file in os.listdir(os.path.join(args.data,'view_pc','test',label))]
            files.sort()
            obj_ids = np.unique([file.split('.')[0].split('_')[-4] for file in files])
            
            
            for idx, obj_id in enumerate(obj_ids):
                with torch.no_grad():
                    current_vp = start_views[i]

                    polar_angle = int(current_vp/8)*18
                    azimuth_angle = int(current_vp%8)*45
                    file_name = '{}_{}_normal_{}_{}.png'.format(label,obj_id,polar_angle,azimuth_angle)

                    view = Image.open(os.path.join(args.data,'image','test',label,file_name))
                    
                    view = view.convert('RGB')
                    view = transform(view)
                    view = torch.unsqueeze(view,0).to(device)
                    class_prediction = cls_model(view)

                    cls_predictions = class_prediction.cpu().numpy()
                    explored_views = [current_vp]


                    while len(explored_views) < MAX_VIEWS and  np.max(softmax(cls_predictions)) < threshold:

                        if args.nbv_type == 'map':
                            file_name = '{}_{}_normal_{}_{}.pcd'.format(label,obj_id,polar_angle,azimuth_angle)
                            view_pc = o3d.io.read_point_cloud(os.path.join(args.data,'view_pc','test',label,file_name))
                            nbv = nbv_model.predict_next_best_view(np.asarray(view_pc.points),explored_views,label,current_vp)
                        else:
                            nbv = nbv_model.get_nbv(current_vp,explored_views)

                        
                        current_vp = nbv

                        explored_views.append(nbv)

                        polar_angle = int(current_vp/8)*18

                        azimuth_angle = int(current_vp%8)*45 

                        file_name = '{}_{}_normal_{}_{}.png'.format(label,obj_id,polar_angle,azimuth_angle)
                        view = Image.open(os.path.join(args.data,'image','test',label,file_name))
                        view = view.convert('RGB')
                        view = transform(view)
                        view = torch.unsqueeze(view,0).to(device)
                        class_prediction = cls_model(view)
                        cls_predictions+=class_prediction.detach().cpu().numpy()

                    prediction = np.argmax(softmax(cls_predictions))

                    results['gt'].append(label_to_idx[label])
                    results['pred'].append(prediction)
                    results['label'].append(label)
                    results['obj_ind'].append(obj_id)
                    results['correct'].append((np.argmax(cls_predictions) == label_to_idx[label]).sum())
                    results['views'].append(explored_views)
                    results['confidence'].append(np.max(softmax(cls_predictions)))
                    
                    correct += (np.argmax(cls_predictions) == label_to_idx[label]).sum()
                    i+=1
                    j+=1
                    if args.verbose:
                        printProgressBar(j,total_num_objects*50,prefix="{:.2f}%".format((correct/i)*100),suffix="{} {:.2f}".format(label,(time.time()-start)/j*total_num_objects*50))
                        # printProgressBar(i,len(obj_ids),prefix="{}".format(file_name),suffix="{} {}".format(label,obj_id))
    if not os.path.isdir('baseline_results/NBV_local/{}'.format(threshold)):
        os.mkdir('baseline_results/NBV_local/{}'.format(threshold))
        for map in ['random', 'unidirectional','furthest','LCM10_40','LCM40_40','LEM10_40','LEM40_40']:
            os.mkdir('baseline_results/NBV_local/{}/{}'.format(threshold,map))

    if args.nbv_type == 'map':
        if args.map_type == 'entropy':
            pd.DataFrame(results).to_pickle('baseline_results/NBV_local/{}/LEM{}_40/results_{}_40.pkl'.format(threshold,args.num_classes_NBV_model, args.num_classes_cls))
            pd.DataFrame(results).to_csv('baseline_results/NBV_local/{}/LEM{}_40/results_{}_40.csv'.format(threshold,args.num_classes_NBV_model, args.num_classes_cls),index=False)
        else:
            pd.DataFrame(results).to_pickle('baseline_results/NBV_local/{}/LCM{}_40/results_{}_40.pkl'.format(threshold,args.num_classes_NBV_model, args.num_classes_cls))
            pd.DataFrame(results).to_csv('baseline_results/NBV_local/{}/LCM{}_40/results_{}_40.csv'.format(threshold,args.num_classes_NBV_model, args.num_classes_cls),index=False)

    else:
        pd.DataFrame(results).to_pickle('baseline_results/NBV_local/{}/{}/results_{}_40.pkl'.format(threshold,args.nbv_type,args.num_classes_cls))
        pd.DataFrame(results).to_csv('baseline_results/NBV_local/{}/{}/results_{}_40.csv'.format(threshold,args.nbv_type,args.num_classes_cls),index=False)