# -*- coding: utf-8 -*-
"""
Created on Thu Jun 17 14:31:21 2021

@author: user
"""

import pickle
import numpy
import math
import helper
import numpy as np
import torch

#checked 
datasets = ['RH_ASDNC', 'LH_ASDNC', 'RH_ADLMCI', 'LH_ADLMCI']
#args_dataset = 'RH_ASDNC'
dataset_idx = 0
n_rois = 35

predicted_multigraphs = []
labels = []

for cv_idx in range(5):
    # collect predicted view 1 generated from src 0
    with open('data predicted/'+datasets[dataset_idx]+'_v_01_src_'+str(0)+'_cv_'+str(cv_idx)+'_predicted.pickle','rb') as f:
        adjs_predicted_view_1 = pickle.load(f)['test'][0].detach().cpu().numpy()
    # collect predicted view 0 generated from src 1
    with open('data predicted/'+datasets[dataset_idx]+'_v_01_src_'+str(1)+'_cv_'+str(cv_idx)+'_predicted.pickle','rb') as f:
        adjs_predicted_view_0 = pickle.load(f)['test'][0].detach().cpu().numpy()
    
    with open('data predicted/'+datasets[dataset_idx]+'_v_01_src_'+str(0)+'_cv_'+str(cv_idx)+'_predicted.pickle','rb') as f:
        labels_cv = pickle.load(f)['labels']
    adjs_mat_view_0 = []
    adjs_mat_view_1 = []
    adjs_mats_cv = []
    for i in range(len(adjs_predicted_view_0)):
        #adj = np.zeros
        view0_elt = helper.vec_to_mat(adjs_predicted_view_0[i,:], n_rois)
        view1_elt = helper.vec_to_mat(adjs_predicted_view_1[i,:], n_rois)
        views_elt = np.stack((view0_elt,view1_elt), axis = -1)
        adjs_mats_cv.append(views_elt)
    
    predicted_multigraphs.extend(adjs_mats_cv)
    labels.extend(labels_cv)

with open('multigraphs predicted/'+datasets[dataset_idx]+'_generated_edges', 'wb') as f:
    pickle.dump(predicted_multigraphs, f)
with open('multigraphs predicted/'+datasets[dataset_idx]+'_generated_labels', 'wb') as f:
    pickle.dump(labels, f)

print("es")