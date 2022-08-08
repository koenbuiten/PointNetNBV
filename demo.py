import argparse
import os
import numpy as np

import torch
import torchvision.transforms as transforms
import open3d as o3d
from PIL import Image
from scipy.special import softmax

from models.resnet import resnet34
from models.NBV import NBV
import matplotlib.pyplot as plt

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
explored_images = []

def get_view(polar_angle,azimuth_angle):
    view_file_name = '{}_{}_{}_{}.png'.format(label,obj_id,polar_angle,azimuth_angle)
    view = Image.open(os.path.join(args.object_data,'views',view_file_name))
    view = view.convert('RGB')
    explored_images.append(view)
    view = transform(view)
    view = torch.unsqueeze(view,0).to(device)
    return view

view = get_view(int(current_vp/8)*18,int(current_vp%8)*45)


with torch.no_grad():
    class_prediction = cls_model(view)

    cls_predictions = class_prediction.cpu().numpy()
    explored_views = [current_vp]
    predictions = [np.argmax(softmax(class_prediction.cpu().numpy()))]
    confidence_predictions = [np.max(softmax(class_prediction.cpu().numpy()))]
    softmax_predictions = [softmax(class_prediction.cpu().numpy())]
    polars = [int(current_vp/8)*18]
    azimuthals = [int(current_vp%8)*45]

    while len(explored_views) < args.max_views and  np.max(softmax(cls_predictions)) < args.threshold:

        polar_angle = int(current_vp/8)*18
        azimuth_angle = int(current_vp%8)*45 
        pcd_file_name = '{}_{}_{}_{}.pcd'.format(label,obj_id,polar_angle,azimuth_angle)
        view_pc = o3d.io.read_point_cloud(os.path.join(args.object_data,'pcds',pcd_file_name))

        nbv = nbv_model.predict_next_best_view(np.asarray(view_pc.points),explored_views,label,current_vp)

        current_vp = nbv

        explored_views.append(nbv)

        polar_angle = int(current_vp/8)*18
        azimuth_angle = int(current_vp%8)*45 
        polars.append(polar_angle)
        azimuthals.append(azimuth_angle)

        view = get_view(polar_angle,azimuth_angle)

        class_prediction = cls_model(view)
        cls_predictions+=class_prediction.cpu().numpy()

        softmax_predictions.append(softmax(class_prediction.cpu().numpy()))
        predictions.append(np.argmax(softmax(class_prediction.cpu().numpy())))
        confidence_predictions.append(np.max(softmax(class_prediction.cpu().numpy())))
        
        

    prediction = np.argmax(softmax(cls_predictions))
    print("Trajectory: {}".format(explored_views))
    print("Predidction per view {}".format([LABELS10[pred] for pred in predictions]))
    print("Confidence per prediction {}".format(confidence_predictions))

    print("\nAccumulated raw prediction output:\n{}".format(cls_predictions[0]))
    print("Accumulated softmax prediction output:\n{}".format([np.round(pred,2) for pred in softmax(cls_predictions)[0]]))
    print("Accumulated confidence score: {}".format(np.max(softmax(cls_predictions))))
    print("Final prediction: {}".format(LABELS10[prediction]))

    if len(explored_views) > 1:
        fig, axis = plt.subplots(1,len(explored_views),figsize=(3*len(explored_views),8.5))
        axis[0].text(0,460,"Combined confidence",verticalalignment='top',fontsize=14)

        axis[0].text(0,720,"Final prediction",verticalalignment='top',fontsize=14)
        axis[0].text(0,750,LABELS10[prediction],verticalalignment='top',fontsize=12)

        axis[0].text(0,780,"Correct prediction",verticalalignment='top',fontsize=14)
        axis[0].text(0,810,label,verticalalignment='top',fontsize=12)

        axis[0].text(122,500,"Confidence",verticalalignment='top')
        axis[0].text(0,500,"Label",verticalalignment='top')
        axis[0].text(0,520,"\n".join(LABELS10),verticalalignment='top')

        max_pred_idx = np.argmax(softmax(cls_predictions)[0])
        conf_string = [str(np.round(conf,2)) for conf in softmax(cls_predictions)[0]]
        conf_string[max_pred_idx] = r"$\bf{" + conf_string[max_pred_idx] + "}$"
        
        conf_string = "\n".join(conf_string)
        axis[0].text(122,520,conf_string,verticalalignment='top')

        for i, view in enumerate(explored_images):
            axis[i].imshow(view)
            axis[i].tick_params(bottom=False,left=False)
            axis[i].set_yticklabels([])
            axis[i].set_xticklabels([])
            axis[i].set_title('view {}: ({},{})'.format(explored_views[i],polars[i],azimuthals[i]))
            axis[i].xaxis.set_label_coords(0, 0)

            bbox = axis[i].get_position()
            bbox.y0 = 0.9- (bbox.y1-bbox.y0)
            bbox.y1 = 0.9
            
            axis[i].set_position(bbox)
            axis[i].text(122,240,"Confidence",verticalalignment='top')
            axis[i].text(0,240,"Label",verticalalignment='top')
            axis[i].text(0,260,"\n".join(LABELS10),verticalalignment='top')

            max_pred_idx = np.argmax(softmax_predictions[i][0])
            conf_string = [str(np.round(conf,2)) for conf in softmax_predictions[i][0]]
            conf_string[max_pred_idx] = r"$\bf{" + conf_string[max_pred_idx] + "}$"
            
            conf_string = "\n".join(conf_string)

            axis[i].text(122,260,conf_string,verticalalignment='top')
    
    else:
        fig, axis = plt.subplots(1,len(explored_views),figsize=(3*len(explored_views),8.5))
        axis.imshow(explored_images[0])
        axis.tick_params(bottom=False,left=False)
        axis.set_yticklabels([])
        axis.set_xticklabels([])
        axis.set_title('view {}: ({},{})'.format(explored_views[0],polars[0],azimuthals[0]))
        axis.xaxis.set_label_coords(0, 0)

        bbox = axis.get_position()
        bbox.y0 = 0.9- (bbox.y1-bbox.y0)
        bbox.y1 = 0.9
            
        axis.set_position(bbox)
        axis.text(0,260,"Combined confidence",verticalalignment='top',fontsize=14)

        axis.text(0,520,"Final prediction",verticalalignment='top',fontsize=14)
        axis.text(0,550,LABELS10[prediction],verticalalignment='top',fontsize=12)

        axis.text(0,580,"Correct prediction",verticalalignment='top',fontsize=14)
        axis.text(0,610,label,verticalalignment='top',fontsize=12)

        axis.text(122,300,"Confidence",verticalalignment='top')
        axis.text(0,300,"Label",verticalalignment='top')
        axis.text(0,320,"\n".join(LABELS10),verticalalignment='top')

        max_pred_idx = np.argmax(softmax(cls_predictions)[0])
        conf_string = [str(np.round(conf,2)) for conf in softmax(cls_predictions)[0]]
        conf_string[max_pred_idx] = r"$\bf{" + conf_string[max_pred_idx] + "}$"
        
        conf_string = "\n".join(conf_string)
        axis.text(122,320,conf_string,verticalalignment='top')
    plt.show()
