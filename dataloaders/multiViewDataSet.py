
from turtle import st
from torch.utils.data.dataset import Dataset
import os
from PIL import Image
import random
import time
import numpy as np
import pandas as pd
from skimage.feature import peak_local_max
from skimage.morphology import local_minima 
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import random
# from skimage.feature import peak_local_max

def string_to_tuple(string):
    string = string.replace('(',"")
    string = string.replace(')',"")
    string = string.split(',')
    string = [int(value) for value in string]
    # print(tuple(string))
    return tuple(string)

def string_to_array(string,obj):
    string = string.replace(',)',")")
    string = string.replace('(',"")
    string = string.replace(')',"")
    string = string.replace('[ ',"")
    string = string.replace('] ',"")
    string = string.replace('  '," ")
    string = string.replace('[',"")
    string = string.replace(']',"") 
    string = string.split(',')
    
    try:
        string = [int(value) for value in string]
    except:
        print(string)
        print(obj)
        exit()
    # print(tuple(string))
    return string

def get_obj_idn(label,file):
    if label == 'night_stand':
        return file.split('_')[2]
    else:
        return file.split('_')[1]

class MultiViewPairDataSet(Dataset):

    def find_classes(self, dir):
        classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}

        return classes, class_to_idx

    def __init__(self, root, set_type,view_num=18, transform=None):
        print("creating " + str(set_type) + " dataset")
        self.x = []
        self.y = []
        self.pairs = []
        self.objs = []
        self.poses = []

        set_labels = []
        set_objs = []
        set_views = []
        set_poses = []

        self.root = root

        self.classes, self.class_to_idx = self.find_classes(os.path.join(root,set_type))
        self.transform = transform
        data_df = pd.read_pickle(os.path.join(root,set_type,'dataset_{}.pkl'.format(set_type)))

        data_df = data_df.sort_values(['label','obj_ind','pose','view'])

        view_set = []
        for idx, row in data_df.iterrows():
            view_set.append(row['path'])           
            if int(row['view']) == view_num-1:
                set_views.append(view_set)
                set_labels.append(self.class_to_idx[row['label']])
                set_objs.append(row['obj_ind'])
                set_poses.append(row['pose'])
                view_set = []

        label_idx = 0
        for v_set in set_views:
            
            for i in range(len(v_set)-1):
                for j in range(i+1,len(v_set)):
                    self.x.append([v_set[i],v_set[j]])
                    self.y.append(set_labels[label_idx])
                    self.pairs.append((i,j))
                    self.objs.append(set_objs[label_idx])
                    self.poses.append(set_poses[label_idx])
            label_idx += 1


    # Override to give PyTorch access to any image on the dataset
    def __getitem__(self, index):
        original_input = self.x[index]
        new_input = []
        for view in original_input:
            im = Image.open(view)
            im = im.convert('RGB')

            if self.transform is not None:
                im = self.transform(im)
            new_input.append(im)
        return new_input, self.y[index], self.pairs[index], self.objs[index],self.poses[index]

    # Override to give PyTorch size of dataset
    def __len__(self):
        return len(self.x)
def get_local_views(start_view):

    full_map = np.asarray(range(40))
    full_map = np.reshape(full_map,(5,8))
    start_view_loc = np.where(full_map == start_view)
    shift = 3-start_view_loc[1]
    full_map = np.roll(full_map,shift,axis=1)
    start_view_loc = np.where(full_map == start_view)
    local_map = full_map[:,start_view_loc[1][0]-1:start_view_loc[1][0]+2]

    local_map = local_map.flatten()
    
    np.delete(local_map,np.where(local_map == 9))
    return local_map


class MultiViewDataSet(Dataset):

    def find_classes(self, classes):
        
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}

        return classes, class_to_idx

    def __init__(self, root, set_type,view_num=18,set_view_num=40, transform=None,random_vps=False,local=False,classes='all'):
        self.x = []
        self.y = []
        self.root = root
        if classes == 'all':
            classes = [d for d in os.listdir(os.path.join(root,set_type)) if os.path.isdir(os.path.join(os.path.join(root,set_type), d))]
            
        self.classes, self.class_to_idx = self.find_classes(classes)
        # self.classes, self.class_to_idx = self.find_classes(os.path.join(root,set_type))

        self.transform = transform
        start = time.time()
        data_df = pd.read_pickle(os.path.join(root,set_type,'dataset_{}.pkl'.format(set_type)))
        # data_df = data_df.loc[data_df['pose'] == 'normal']
        data_df = data_df.sort_values(['label','obj_ind','pose','view'])
        self.poses = np.unique(data_df['pose'].values)
        view_set = []
        path_set = []
        for idx, row in data_df.iterrows():
            if row['label'] not in self.classes:
                continue
            view_set.append(int(row['view'])) 
            path_set.append(row['path'])          
            if int(row['view']) == set_view_num-1:
                if len(view_set) != set_view_num:
                    print('Not enough views')
                if random_vps:
                    if local:
                        start_view = 0
                        
                        NRV_ids = [start_view] # Next-Random-View path set
                        for i in range(view_num-1):
                        # start_view = np.random.randint(0,set_view_num)
                            local_views = get_local_views(start_view)
                            while local_views[start_view] in NRV_ids:
                                start_view = np.random.randint(0,15)
                            NRV_ids.append(local_views[start_view])
                            
                        
                        # np.random.shuffle(local_views)
                        
                        
                        # local_path_set = np.append(local_path_set,path_set[start_view])
                        # print(NRV_path_set)
                        # exit()
                        self.x.append(np.asarray(path_set)[NRV_ids])
                        

                else:
                    self.x.append(path_set)
                
                self.y.append(self.class_to_idx[row['label']])
                view_set = []
                path_set = []
                

    # Override to give PyTorch access to any image on the dataset
    def __getitem__(self, index):
        original_input = self.x[index]
        new_input = []
        for view in original_input:
            im = Image.open(view)
            im = im.convert('RGB')

            if self.transform is not None:
                im = self.transform(im)
            new_input.append(im)
        return new_input, self.y[index]

    # Override to give PyTorch size of dataset
    def __len__(self):
        return len(self.x)

def obj_idn_to_string(obj_idn):
    obj_idn = int(obj_idn)
    if obj_idn < 10:
        return '000' + str(obj_idn)
    elif obj_idn < 100:
        return '00' + str(obj_idn)
    elif obj_idn <1000:
        return '0' + str(obj_idn)
    return str(obj_idn)

def get_view_num(angle_h,angle_v):
    hor_split_n = int(angle_h)/18
    ver_split_n = int(angle_v)/45
    return int(hor_split_n*8 + ver_split_n)

class MultiBestViewDataset(Dataset):
    def find_classes(self, dir):
        classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}

        return classes, class_to_idx
    # selection methods: peak, local_peak, ascending, descending, descending_spec
    def __init__(self, root, set_type,descriptor_data,set_view_num,view_num, transform=None):
        self.x = []
        self.y = []


        self.transform = transform

        self.classes, class_to_idx = self.find_classes(os.path.join(root,set_type))
        # root / <train/test> / <label> / <view>.png
        self.root = root
        self.set_type = set_type
        print(self.classes)
        descriptor_data = descriptor_data.sort_values(['gt','obj_ind','correct','conf'],ascending=False)

        descriptor_data = descriptor_data.reset_index(drop=True)


        object_paths =[]
        current_class = ''
        print("View num: {}".format(view_num))
        for index, row in descriptor_data.iterrows():
            # if int(row['correct']) == 1:
                # correct += 1
            object_paths.append(row['path'])
            current_class = int(row['gt'])

            if (index+1)%set_view_num == 0:

                self.x.append(object_paths[0:view_num])
                self.y.append(current_class)
                object_paths =[]
            if index % 10000 == 0:
                print(index, end='\r')

    # Override to give PyTorch access to any image on the dataset
    def __getitem__(self, index):
        original_input = self.x[index]
        new_input = []
        for view in original_input:
            im = Image.open(view)
            im = im.convert('RGB')

            if self.transform is not None:
                im = self.transform(im)
            new_input.append(im)

        return new_input, self.y[index] 

    # Override to give PyTorch size of dataset
    def __len__(self):
        return len(self.x)

    
class DescriptorBasedMultiViewDataSet(Dataset):
    def find_classes(self, classes):
        
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}

        return classes, class_to_idx
    # selection methods: peak, local_peak, ascending, descending, descending_spec
    def __init__(self, root, set_type,descriptor, descriptor_data,set_view_num,view_num, transform=None,selection_method='full',classes='all'):
        self.x = []
        self.y = []
        self.entropy_values = []
        # root / <train/test> / <label> / <view>.png
        self.root = root
        self.set_type = set_type
        self.selection_method = selection_method
        # self.classes, self.class_to_idx = self.find_classes(np.unique(descriptor_data['label'].values))
        if classes == 'all':
            classes = np.unique(descriptor_data['label'].values)

        self.classes, self.class_to_idx = self.find_classes(classes)
        self.transform = transform
        descriptor_data = descriptor_data.sort_values(['label','obj_ind','pose','angle_h','angle_v'])
        current_pose = descriptor_data['pose'].values[0]
        ascending_classes = ['chair','cone','curtain','cup','curtain','desk','guitar','mantel','monitor','piano','sink','table','tent','toilet','vase']


        views = []
        measure_values = []
        number_of_views = []
        for index, row in descriptor_data.iterrows():  

            if row['label'] not in self.classes:
                continue
            measure_values.append(row['entropy'])
            obj_ind = obj_idn_to_string(row['obj_ind'])
            label = row['label']
            current_pose = row['pose']
            view_file = '{}_{}_{}_{}_{}.png'.format(label,obj_ind,current_pose,row['angle_h'],row['angle_v'])
            if os.path.isfile(os.path.join(self.root,self.set_type,label,view_file)):
                views.append(os.path.join(self.root,self.set_type,label,view_file))
            else:
                print('Not a file')

            if len(views) == set_view_num:
                self.y.append(self.class_to_idx[label])
                    
                # Create full entropy maps
                if selection_method == 'full':
                    self.x.append(views)
                    self.entropy_values.append(measure_values)

                # Peak viewpoints
                elif selection_method == 'peak':
                    if set_view_num == 40:
                        entropies = np.asarray(measure_values).reshape((5,8))
                        views = np.asarray(views).reshape(5,8)
                        if row['label'] == 'table':
                            
                            view_ids = local_minima(entropies)
                            view_ids = np.where(view_ids == True)
                            view_ids = [[view_ids[0][i],view_ids[1][i]] for i in range(len(view_ids[0]))]
                            
                        else:
                            view_ids = peak_local_max(entropies, min_distance=1, exclude_border=False)

                        temp_views = [views[x][y] for [x,y] in view_ids]
                        
                        
                    else:
                        view_ids = find_peaks(measure_values)[0]
                        temp_views = np.asarray(views)[view_ids]
                    number_of_views.append(len(temp_views))
                    self.x.append(temp_views)
                elif selection_method == 'local_peak':
                    ascending = True
                    # start_view = np.random.randint(0,set_view_num)
                    start_view = 0
                    nbvs = [views[start_view]] #Next-Best-Views
                    nbv_ids = [start_view]

                    for i in range(view_num-1):
                        
                        local_view_ids = get_local_views(start_view)
                        
                        entropies = np.asarray(measure_values)[local_view_ids].reshape(5,3)
                        if row['label'] in ascending_classes:
                            ascending= False
                            view_ids = local_minima(entropies)
                            view_ids = np.where(view_ids == True)
                            view_ids = [[view_ids[0][i],view_ids[1][i]] for i in range(len(view_ids[0]))]
                            if len(view_ids) == 0:
                                view_ids = np.argsort(entropies.flatten())

                                view_ids = [id for id in  view_ids if id not in nbv_ids]
                            nbv_id = view_ids[0]


                        else:
                            view_ids = peak_local_max(entropies, min_distance=1, exclude_border=False)
                            view_ids = [id[0]*3 + id[1] for id in view_ids if id[0]*3 + id[1] not in nbv_ids]

                            if len(view_ids) == 0:

                                view_ids = np.argsort(entropies.flatten())
                                view_ids = [id for id in  view_ids if id not in nbv_ids]
                                nbv_id = view_ids[-1]

                            else:
                                nbv_id = view_ids[0]
                        nbvs.append(views[nbv_id])
                        nbv_ids.append(nbv_id)
                        start_view = nbv_id

                    self.x.append(nbvs)
                else:
                    temp_dict = pd.DataFrame({'views':views,'entropy': measure_values})
                    if selection_method == 'ascending':
                        temp_dict = temp_dict.sort_values('entropy',ascending=True)
                    elif selection_method == 'descending':
                        temp_dict = temp_dict.sort_values('entropy',ascending=False)
                    elif selection_method == 'descending_spec':
                        if row['label'] in ascending_classes:
                            temp_dict = temp_dict.sort_values('entropy',ascending=True)
                        else:
                            temp_dict = temp_dict.sort_values('entropy',ascending=False)
                    # temp_dict = temp_dict.sort_values(descriptor,ascending=False)
                    self.x.append(temp_dict['views'].values[:int(view_num)])
                views = []
                measure_values = []
            if index % 10000 == 0:
                print(index, end='\r')
        # print(np.asarray(self.x).shape)
        if selection_method == 'peak':
            print("Avergae number of views used: {}".format(np.mean(number_of_views)))
        # np.save('test_data/number_of_views_M{}_{}.npy'.format(len(self.classes),set_view_num),number_of_views)
        # exit()
    # Override to give PyTorch access to any image on the dataset
    def __getitem__(self, index):
        original_input = self.x[index]
        new_input = []
        for view in original_input:
            im = Image.open(view)
            im = im.convert('RGB')

            if self.transform is not None:
                im = self.transform(im)
            new_input.append(im)

        return new_input, self.y[index] 

    # Override to give PyTorch size of dataset
    def __len__(self):
        return len(self.x)