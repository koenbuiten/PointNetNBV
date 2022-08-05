
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

class LEMDataset(Dataset):
    def find_classes(self, dir):
        classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}

        return classes, class_to_idx
    def __init__(self, root, set_type,measure_file,normalize_entropy=True):
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
        
        for idx,row in measure_file.iterrows():
            if obj_ind == '':
                obj_ind = obj_ind_to_string(row['obj_ind'])

            if int(obj_ind) != row['obj_ind']:
                if normalize_entropy:
                    measure_map /= np.max(measure_map)
                if np.sum(measure_map) < 1:
                    print("entropy map is smaller then 1")
                    exit()
                self.maps.append(measure_map)
                measure_map = []
                obj_ind = obj_ind_to_string(row['obj_ind'])
                map_index += 1
                
            measure_map.append(row['entropy'])
            
            ppc_file_name = '{}_{}_{}_{}_{}.pcd'.format(row['label'],obj_ind,'normal',row['angle_h'],row['angle_v'])
            ppc_file_path = os.path.join(root,'view_pc',set_type,row['label'],ppc_file_name)

            self.part_pcs.append(ppc_file_path)

            hor_split_n = int(row['angle_h'])/self.h_step_angle
            ver_split_n = int(row['angle_v'])/self.v_step_angle
            self.views.append(int(hor_split_n*8 + ver_split_n))

            self.map_index.append(map_index)
            self.obj_inds.append(obj_ind)
            self.labels.append(row['label'])
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