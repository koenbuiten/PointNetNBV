import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

from classification_baselines import classification_test


def create_confusion_matrix(results,out_path,depth,num_views, sampling_method,set_view_num):
    
    results['pred'] = [int(item) for item in results['pred'].values]
    results['gt'] = [int(item) for item in results['gt'].values]   

    label_inds, label_count = np.unique(results['gt'].values,return_counts=True)
    num_labels = len(label_inds)

    instance_acc = np.asarray(results['pred'].values == results['gt'].values).sum()
    instance_acc /= np.sum(label_count)/100

    class_acc = np.zeros(num_labels)
    for index, row in results.iterrows():
        class_acc[row['gt']] += np.asarray(row['gt']==row['pred']).sum()
    class_acc /= (label_count/100)
    class_acc = np.mean(class_acc)

    if num_labels == 10:
        labels = ['bathtub', 'bed', 'chair', 'desk', 'dresser', 'monitor', 'night_stand', 'sofa', 'table', 'toilet']
    else:
        labels = ['airplane', 'bathtub', 'bed', 'bench', 'bookshelf', 'bottle', 'bowl', 'car', 'chair', 'cone', 'cup', 'curtain', 'desk', 'door', 'dresser', 'flower_pot', 'glass_box', 'guitar', 'keyboard', 'lamp', 'laptop', 'mantel', 'monitor', 'night_stand', 'person', 'piano', 'plant', 'radio', 'range_hood', 'sink', 'sofa', 'stairs', 'stool', 'table', 'tent', 'toilet', 'tv_stand', 'vase', 'wardrobe', 'xbox']
    
    fig  = ConfusionMatrixDisplay.from_predictions(results['gt'].values, results['pred'].values,
        normalize='true',display_labels =labels,xticks_rotation="vertical",
        colorbar=False)
    if num_labels == 10:
        fig.figure_.set_size_inches(12,12)
    else:
        fig.figure_.set_size_inches(20,20)
    plt.title('Instance acc: {:.2f} Class accuracy: {:.2f}'.format(instance_acc,class_acc))

    if not os.path.isdir(os.path.join(out_path,'Modelnet{}_{}_Resnet{}'.format(num_labels,set_view_num,depth))):
        os.mkdir(os.path.join(out_path,'Modelnet{}_{}_Resnet{}'.format(num_labels,set_view_num,depth)))
    plt.savefig(os.path.join(out_path,'Modelnet{}_{}_Resnet{}'.format(num_labels,set_view_num,depth),
        '{}_{}_views'.format(sampling_method,num_views)))

SAMPLING_METHODS = ['best','normal','random','peak', 'local_peak', 'ascending', 'descending', 'descending_spec','full']


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Classification baseline comparison')
    parser.add_argument('--data', metavar='DIR', help='path to dataset')
    parser.add_argument('--model_path', metavar='DIR', help='Path to model file')
    parser.add_argument('--cm_path', metavar='DIR', help='Path to store confusion matrices')
    parser.add_argument('--pred_data_path', metavar='DIR', help='Path to store prediction data')
    parser.add_argument('--sampling_methods', choices=SAMPLING_METHODS, nargs='*',help='Sampling methods to compare')
    parser.add_argument('--depth', type=int,default=18)
    parser.add_argument('--set_view_num', type=int,default=40)
    parser.add_argument('--verbose',dest='verbose',action='store_true', help='Use progressbar')
    args = parser.parse_args()

    if not os.path.isdir(args.cm_path):
        os.mkdir(args.cm_path)
    
    for sampling_method in args.sampling_methods:
        
        if sampling_method == 'normal':
            sampling_data_path = os.path.join(args.data,'entropy_measure_test.csv')
            cls_results = classification_test(args.data+'/image',args.model_path,args.set_view_num,
                args.depth,1,sampling_method=sampling_method,sampling_data_path=sampling_data_path,verbose=True,out_path=args.pred_data_path)
            create_confusion_matrix(cls_results,args.cm_path,args.depth, view,sampling_method,args.set_view_num)
        elif sampling_method == 'best':
            for view in range(3,5):
                sampling_data_path = 'baseline_results/Modelnet40_40_Resnet34/best_cls_vp_resnet_34_singleview_M40_40_b32.csv'
                cls_results = classification_test(args.data+'/image',args.model_path,args.set_view_num,
                    args.depth,view,sampling_method=sampling_method,sampling_data_path=sampling_data_path,verbose=True,out_path=args.pred_data_path)
                create_confusion_matrix(cls_results,args.cm_path,args.depth,view,sampling_method,args.set_view_num)
        else:
            for view in range(1,6):
                sampling_data_path = os.path.join(args.data,'entropy_measure_test.csv')
                cls_results = classification_test(args.data+'/image',args.model_path,args.set_view_num,
                    args.depth,view,sampling_method=sampling_method,sampling_data_path=sampling_data_path,verbose=True,out_path=args.pred_data_path)
                create_confusion_matrix(cls_results,args.cm_path,args.depth, view,sampling_method,args.set_view_num)

