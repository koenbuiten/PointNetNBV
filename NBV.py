"""
Local Map PointNet Next-Best-View prediction.
Description: Predicts the values for the local neighbouring views for entropy of cls performance and 
selects the Next-Best-View for classification

Input: Point-cloud of current view
Ouput: Next best view
Parameters: Map type (Entropy, cls performance)

"""

import os
import argparse
import numpy as np
import pandas as pd

from models.LMnet import LMnet

from skimage.feature import peak_local_max
from skimage.morphology import local_minima 
from scipy.spatial import distance

import torch

MAP_TYPES = ['entropy','cls_performace']
BASELINE_TYPES = ['random','furthest','unidirectional']
class baselineNBV():
    def random_nbv(self,current, explored):
        dummy_map = np.asarray(range(40)).reshape(5,8)
        dummy_map = np.roll(dummy_map,1-np.where(dummy_map == current)[1])
        local_map = dummy_map[:,0:3].flatten()
        random_view = local_map[np.random.randint(15)]
        while random_view in explored:
            local_map = np.delete(local_map,np.where(local_map == random_view))
            random_view = local_map[np.random.randint(len(local_map))]
        return random_view

    def furthest_nbv(self,current,explored):
        dummy_map = np.asarray(range(40)).reshape(5,8)
        dummy_map = np.roll(dummy_map,1-np.where(dummy_map == current)[1])
        local_map = dummy_map[:,0:3]
        distances = []
        for x in range(5):
            for y in range(3):
                distances.append(distance.euclidean((x,y),(np.where(local_map==current))))
        sorted_inds = np.argsort(distances)
        furthest_view = local_map.flatten()[sorted_inds[-1]]
        while furthest_view in explored:
            sorted_inds = np.delete(sorted_inds,-1)
            furthest_view = local_map.flatten()[sorted_inds[-1]]
        return furthest_view

    def unidirectional_nbv(self,current,explored):
        dummy_map = np.asarray(range(40)).reshape(5,8)
        dummy_map = np.roll(dummy_map,-np.where(dummy_map == current)[1],axis=1)
        current_row = np.where(dummy_map == current)[0].item()
        return dummy_map[current_row][1]

    def __init__(self, baseline_type):
        if baseline_type == 'random':
            self.nbv = self.random_nbv
        elif baseline_type == 'furthest':
            self.nbv = self.furthest_nbv
        else:
            self.nbv = self.unidirectional_nbv
        
    def get_nbv(self,current,explored):
        return self.nbv(current,explored)
        
class NBV():
    def load_model(self,model_path,device):
        model = LMnet(local_views=15)
        # Load checkpoint.
        print('==> Loading NBV model..')
        assert os.path.isfile(model_path), 'Error: no model file found!'

        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint['state_dict'])
        model.to(device)
        model.eval()
        return model

    def __init__(self, map_type,model_path,device="cuda:0"):
        self.device=device
        self.model = self.load_model(model_path,self.device)
        self.map_type = map_type
        self.ascending_classes = ['chair', 'cone', 'cup', 'curtain', 'guitar', 'mantel', 'monitor', 'piano', 'sink', 'table', 'tent', 'toilet', 'vase']

    def predict_map(self,view_pc): # View_pc in shape (3,1024) as numpy array
        # Normalize point cloud
        view_pc = np.asarray([(point- np.min(view_pc))/(np.max(view_pc)-np.min(view_pc)) for point in view_pc])
        view_pc = torch.from_numpy(view_pc).type(torch.FloatTensor)
        view_pc = torch.unsqueeze(view_pc,0).permute(0,2,1).to(self.device)

        output,  matrix3x3, matrix64x64, feature = self.model(view_pc)
        predicted_map = output.detach().cpu().numpy().reshape(5,3)
        return predicted_map
    
    def predict_feature(self,view_pc):
        view_pc = np.asarray([(point- np.min(view_pc))/(np.max(view_pc)-np.min(view_pc)) for point in view_pc])
        view_pc = torch.from_numpy(view_pc)
        view_pc = torch.unsqueeze(view_pc,0)
        view_pc = view_pc.to(self.device)

        output,  matrix3x3, matrix64x64, feature = self.model(view_pc)
        return feature.detach().cpu().numpy()
    
    def predict_next_best_view(self,view_pc,explored_views,label,current_view):
        predicted_map = self.predict_map(view_pc)
        dummy_map = np.asarray(range(40)).reshape(5,8)
        dummy_map = np.roll(dummy_map,1-np.where(dummy_map == current_view)[1],axis=1)
        local_dummy_map = dummy_map[:,0:3].flatten()

        if self.map_type == 'entropy' and (label in self.ascending_classes):
            view_ids_raw = local_minima(predicted_map)
            view_ids = np.where(view_ids_raw == True)

            view_ids = [view_ids[0][i]*3+view_ids[1][i] for i in range(len(view_ids[0])) if  local_dummy_map[view_ids[0][i]*3+view_ids[1][i]] not in explored_views]

            if len(view_ids) == 0:
                view_ids = np.argsort(predicted_map.flatten())
                view_ids = [id for id in  view_ids if local_dummy_map[id] not in explored_views]
            next_best_view = local_dummy_map[view_ids[0]]

        else:
            view_ids = peak_local_max(predicted_map, min_distance=1, exclude_border=False)
            view_ids = [id[0]*3 + id[1] for id in view_ids if local_dummy_map[id[0]*3 + id[1]] not in explored_views]

            if len(view_ids) == 0:
                view_ids = np.argsort(predicted_map.flatten())
                view_ids = [id for id in  view_ids if local_dummy_map[id] not in explored_views]
                next_best_view = local_dummy_map[view_ids[-1]]
            else:
                next_best_view = local_dummy_map[view_ids[0]]
        return next_best_view
