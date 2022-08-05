
import open3d as o3d
from torch.utils.data.dataset import Dataset
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

def obj_ind_to_string(obj_idn):
    obj_idn = int(obj_idn)
    if obj_idn < 10:
        return '000' + str(obj_idn)
    elif obj_idn < 100:
        return '00' + str(obj_idn)
    elif obj_idn <1000:
        return '0' + str(obj_idn)
    return str(obj_idn)

class LCMDataset(Dataset):
    def find_classes(self, dir):
        classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}

        return classes, class_to_idx
    def __init__(self, root, set_type, measure_file):
        self.root=root
        self.set_type = set_type

        self.maps = []
        self.obj_inds = []
        self.labels = []
        self.views = []
        self.part_pcs = []
        self.map_index = []

        pcd_path = os.path.join(root,'view_pc',set_type)
        self.classes, self.class_to_idx= self.find_classes(pcd_path)

        self.h_step_angle = 18
        self.v_step_angle = 45
        obj_ind = ''
        measure_map = []
        map_index = 0
        points = []

        measure_file = measure_file.sort_values(['gt','obj_ind','view'])
        object_files = []
        object_views = []
        for idx,row in measure_file.iterrows():

            if obj_ind == '':
                obj_ind = obj_ind_to_string(row['obj_ind'])

            if int(obj_ind) != row['obj_ind']:
                if np.sum(measure_map) >= 1 and len(measure_map) == 40:
                    self.maps.append(measure_map)
                    measure_map = []

                    for i in range(40):
                        self.map_index.append(map_index)
                        self.obj_inds.append(obj_ind)
                        self.labels.append(self.classes[int(row['gt'])])
                    self.part_pcs.extend(object_files)
                    self.views.extend(object_views)

                    obj_ind = obj_ind_to_string(row['obj_ind'])
                    map_index += 1
                    object_files = []
                    object_views = []
                    
            polar_angle = int(row['view']/8)*18
            azimuth_angle = int(row['view']%8)*45
            ppc_file_name = '{}_{}_{}_{}_{}.pcd'.format(self.classes[int(row['gt'])],obj_ind,'normal',polar_angle,azimuth_angle)
            ppc_file_path = os.path.join(root,'view_pc',set_type,self.classes[int(row['gt'])],ppc_file_name)

            object_files.append(ppc_file_path)
            object_views.append(int(row['view']))
            measure_map.append(row['conf']*row['correct'])
        self.maps.append(measure_map)


    def __getitem__(self, index):
        label = self.labels[index]
        obj_ind = self.obj_inds[index]
        view = self.views[index]

        part_pc = o3d.io.read_point_cloud(self.part_pcs[index])
        part_pc = np.asarray(part_pc.points)
        part_pc = np.asarray([(point- np.min(part_pc))/(np.max(part_pc)-np.min(part_pc)) for point in part_pc])



        map = np.asarray(self.maps[self.map_index[index]])
        map = np.reshape(map,(5,8))
        angle_v = int(view%8)*45
        map = np.roll(map,int(1-angle_v/45),axis=1) # Shift map so view is in the 2th column, point (angle_h/18,1)
        part_map = map[:,0:3] # Get part map with view in center and left and right column
        part_map = part_map.flatten()

        return part_pc,part_map,view,label,obj_ind

    def __len__(self):
        return len(self.obj_inds)