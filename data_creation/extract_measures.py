import os
import argparse
import pandas as pd
from skimage.measure import shannon_entropy
import matplotlib.pyplot as plt
from data_creation_util import printProgressBar
import numpy as np
import cv2

parser = argparse.ArgumentParser(description="Generates a dataset in CSV format for viewpoint measure: entropy, silhouette length, visible points.")
parser.add_argument("--data", help="Specify root directory to data folder")
parser.add_argument("--out", help="Select a desired output directory.", default="./")
parser.add_argument("--split_set", choices=['test','train'], help="Select train or test set", default="test")
parser.add_argument("--entropy", help="Create entropy dataset", action='store_true')
parser.add_argument("--visible_points", help="Create visible points dataset", action='store_true')
parser.add_argument("--silhouette_length", help="create silhouette_length dataset.", action='store_true')
parser.add_argument("--resume", metavar='DIR',help='Path to dataset csv file')
parser.add_argument('--verbose', dest='verbose', action='store_true', help='Print progressbar')
args = parser.parse_args()


OUT_DIR = args.out
DATA_PATH = args.data

split_set = args.split_set

''' View dataset folder hierarchy
        |__ depth image view_pcd
        |   |__test train
        |       |__class label 1 ... class label n
        |           |__ label_objInd_pose_angleh_anglev.extension
        |__ full_pc
            |__ test train
                |__ class label 1 ... class label n
                    |__ label_objInd_pose.pcd

'''

# for split_set in ['train', 'test']:
labels = []

for label in os.listdir(os.path.join(DATA_PATH,'depth',split_set)):
    if os.path.isdir(os.path.join(DATA_PATH,'depth',split_set,label)):
        labels.append(label)


def save_data(data,label,measure):
    if label == labels[0]:
        print('INFO: Saving data')
        pd.DataFrame(data).to_csv(os.path.join(OUT_DIR, "{}_measure_{}.csv".format(measure,split_set)), index=False)
    else:
        pd.DataFrame(data).to_csv(os.path.join(OUT_DIR, "{}_measure_{}.csv").format(measure,split_set), index=False, mode='a', header=False)

if args.resume:
    data= pd.read_csv(args.resume)

if os.path.isfile(os.path.join(OUT_DIR, "{}_measure_{}.csv".format('entropy',split_set))):
    df = pd.read_csv(os.path.join(OUT_DIR, "{}_measure_{}.csv".format('entropy',split_set)))
    [labels.remove(label) for label in np.unique(df['label'].values)]

for label in labels:
    i = 0

    data_label = []
    data_index = []
    angle_h = []
    angle_v = []

    entropy = []
    silhouette_length = []
    silhouette_area = []
    visible_points = []
    poses = []

    files = os.listdir(os.path.join(DATA_PATH,'depth', split_set,label))
    number_of_files = len(files)
    files.sort()
    current_object = ''
    entropies_object = []
    for file in files:

        file_name = file[0:-4]
        OBJECT_INDEX = file_name.split('_')[-4]
        pose = file_name.split('_')[-3]
        h = file_name.split('_')[-2]
        v = file_name.split('_')[-1]


        data_label.append(label)
        data_index.append(OBJECT_INDEX)
        angle_h.append(h)
        angle_v.append(v)
        poses.append(pose)

        # Calculate entropy on depth image
        # Entropy is calculated by using -pk * log(pk) where pk is the probibilty of a pixels value in the depth image
        # Calculated by the occurrence of the pixel values divided by the number of total pixels

        if args.entropy:
            try:
                depth_image = plt.imread(os.path.join(DATA_PATH,'depth',split_set,label, file))
                entropy.append(shannon_entropy(depth_image))
            except:
                print("Error: not able to open file: {}".format(file))

        # Calculate visible points from from view in the point cloud
        if args.visible_points:
            with open(os.path.join(DATA_PATH,'view_pc',split_set,label, '{}.npy'.format(file_name)), 'rb') as f:
                part_pc_ids = np.load(f)
            visible_points.append(len(part_pc_ids))

        
        # Silhouette length and area from image
        if args.silhouette_length:
            image = cv2.imread(os.path.join(DATA_PATH,'image',split_set,label, file))

            im_bw = cv2.threshold(image, 1, 255, cv2.THRESH_BINARY)[1]
            kernel = np.ones((15, 15), np.uint8)
            silhouette_im = cv2.morphologyEx(im_bw, cv2.MORPH_CLOSE, kernel)
            silhouette_pixels = np.where(silhouette_im != [0,0,0])

            area_perc = len(silhouette_pixels[0])/(image.shape[0]*image.shape[1])
            silhouette_area.append(area_perc)

            silhouette_edges = cv2.Canny(image=silhouette_im, threshold1=100, threshold2=200) # Canny Edge Detection
            silhouette_edge_length = len(np.where(silhouette_edges == 255)[0])
            silhouette_length.append(silhouette_edge_length)

        i+=1
        if args.verbose:
            printProgressBar(i,number_of_files,prefix=label)

    base_dict = {"label": data_label,
                            "obj_ind": data_index,
                            "pose": poses,
                            "angle_h": angle_h,
                            "angle_v": angle_v}

    if args.entropy:
        data = base_dict.copy()
        data['entropy'] = entropy
        save_data(data,label,'entropy')

    if args.visible_points:
        data = base_dict.copy()
        data['visible_points'] = visible_points
       
        save_data(data,label,'visible_points')

    if args.silhouette_length:
        data = base_dict.copy()
        data['silhouette_length'] = silhouette_length
        data['silhouette_area'] = silhouette_area
        save_data(data,label,'silhouette_length_area')


    




