import argparse
import os
import numpy as np

import torch
import torchvision.transforms as transforms
import open3d as o3d
from PIL import Image
from scipy.special import softmax

from dataloaders.LCMnetDataset import LCMDataset
from dataloaders.LEMnetDataSet import LEMDataset
from models.resnet import resnet34
from NBV import NBV


NBV_METHODS = ['LEM','LCM']
MAX_VIEWS = 5
LABELS10 = ['bathtub','bed','chair','desk','dresser','monitor','night_stand','sofa','table','toilet']
parser = argparse.ArgumentParser(description='Next best view prediction from a point-cloud')
parser.add_argument('--object_data',default='data/table_0444')
parser.add_argument('--nbv_model_path', type=str, default='LEMnet10_40_entropy.pth.tar', help='Path to pointnet model')
parser.add_argument('--cls_model_path', type=str, default='resnet_34_singleview_M10_40_b32.pth.tar', help='Path to classification model')
parser.add_argument('--nbv_method',type=str,default='LEM',choices=NBV_METHODS)
parser.add_argument('--threshold',default=0.99, type=float)
parser.add_argument('--max_views',default=5,type=int)
args = parser.parse_args()

def load_model(model_path,device,num_classes=10):

    model = resnet34(pretrained=True, num_classes=num_classes,pooling=None)

    print('\n==> Loading cls model on {}..'.format(device))
    assert os.path.isfile(model_path), 'Error: no checkpoint file found!'

    checkpoint = torch.load(model_path)
    best_acc = checkpoint['best_acc']
    print("INFO: Acc: {}".format(best_acc))
    model.load_state_dict(checkpoint['state_dict'])
    model.to(device)
    
    return model

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  
nbv_model = NBV(args.nbv_method,args.nbv_model_path,device=device)
cls_model = load_model(args.cls_model_path,device)
cls_model.eval()

transform = transforms.Compose([
    transforms.CenterCrop(500),
    transforms.Resize(224),
    transforms.ToTensor(),
])

current_vp = np.random.randint(0,40)
label = args.object_data.split('/')[1].split('_')[0]
obj_id = args.object_data.split('/')[1].split('_')[1]

def get_view(polar_angle,azimuth_angle):
    view_file_name = '{}_{}_{}_{}.png'.format(label,obj_id,polar_angle,azimuth_angle)
    view = Image.open(os.path.join(args.object_data,'views',view_file_name))

    view = view.convert('RGB')
    view = transform(view)
    view = torch.unsqueeze(view,0).to(device)
    return view

view = get_view(int(current_vp/8)*18,int(current_vp%8)*45)
with torch.no_grad():
    class_prediction = cls_model(view)

    cls_predictions = class_prediction.cpu().numpy()
    explored_views = [current_vp]

    while len(explored_views) < args.max_views and  np.max(softmax(cls_predictions)) < args.threshold:

        polar_angle = int(current_vp/8)*18
        azimuth_angle = int(current_vp%8)*45 
        pcd_file_name = '{}_{}_{}_{}.pcd'.format(label,obj_id,polar_angle,azimuth_angle)
        view_pc = o3d.io.read_point_cloud(os.path.join(args.object_data,'pcds',pcd_file_name))

        nbv = nbv_model.predict_next_best_view(np.asarray(view_pc.points),explored_views,label,current_vp)

        current_vp = nbv

        explored_views.append(nbv)

        view = get_view(polar_angle,azimuth_angle)

        class_prediction = cls_model(view)
        cls_predictions+=class_prediction.detach().cpu().numpy()

    prediction = np.argmax(softmax(cls_predictions))
    print("Trajectory: {}".format(explored_views))
    print("Prediction: {}".format(LABELS10[prediction]))