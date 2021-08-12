# -*- coding: utf-8 -*-
"""
Created on Tue Jun  8 17:13:23 2021

@author: user
"""

import pickle
import numpy
import math
import helper
import numpy as np 


''' Goal: Generate a train list of n_views elements with np(280, 595) each
          and a test list of n_views elements with np(30, 595) each'''
              
datasets = ['RH_ASDNC', 'LH_ASDNC', 'RH_ADLMCI', 'LH_ADLMCI']
len_views = [6, 6, 4, 4]
n_rois = 35
cv_number = 5
args_dataset = datasets[3]
args_view = [0,1]
with open('data original/'+args_dataset+'/'+args_dataset+'_edges','rb') as f:
    adjs = pickle.load(f)
with open('data original/'+args_dataset+'/'+args_dataset+'_labels','rb') as f:
    labels = pickle.load(f)
    
graphs_0 = []
for i in range(len(adjs)):
    multigraph_dict = {}
    multigraph_dict = {'adj':adjs[i][:,:,args_view[0]], 'label': labels[i], 'id':i}
    graphs_0.append(multigraph_dict)
folds_graphs_0 = helper.stratify_splits(graphs_0, cv_number)

graphs_1 = []
for i in range(len(adjs)):
    multigraph_dict = {}
    multigraph_dict = {'adj':adjs[i][:,:,args_view[1]], 'label': labels[i], 'id':i}
    graphs_1.append(multigraph_dict)
folds_graphs_1 = helper.stratify_splits(graphs_1, cv_number)
  
'''
sample_example = folds_graphs_1[0][0]['adj']
sample_example_vec = helper.mat_to_vec(sample_example, n_rois)
sample_example_mat = helper.vec_to_mat(sample_example_vec, n_rois)
'''

folds_vecs_0 = []
for cv in range(len(folds_graphs_0)):
    vecs = []
    for i in range(len(folds_graphs_0[cv])):
        vec = helper.mat_to_vec(folds_graphs_0[cv][i]['adj'], n_rois)
        vecs.append(vec)
    folds_vecs_0.append(vecs)
    
folds_vecs_1 = []
for cv in range(len(folds_graphs_1)):
    vecs = []
    for i in range(len(folds_graphs_1[cv])):
        vec = helper.mat_to_vec(folds_graphs_1[cv][i]['adj'], n_rois)
        vecs.append(vec)
    folds_vecs_1.append(vecs)
    
'''    
ab_1 = folds_graphs[1][3]['adj']
ab_2 = helper.vec_to_mat(folds_vecs[1][3], n_rois)
'''

train_tests_0 = []
for cv in range(cv_number):
    train_test={}
    folds_vecs_copy = folds_vecs_0.copy() 
    test = []
    train = []
    test = folds_vecs_copy.pop(cv)
    for i in range(len(folds_vecs_copy)): 
        train.extend(folds_vecs_copy[i])
    print(str("train: "+str(len(train)))+" test: "+str(len(test)))
    train_test['train'] = numpy.stack(train, axis=0 )
    train_test['test'] = numpy.stack(test, axis=0 )
    train_tests_0.append(train_test)
    
train_tests_1 = []
for cv in range(cv_number):
    train_test={}
    folds_vecs_copy = folds_vecs_1.copy() 
    test = []
    train = []
    test = folds_vecs_copy.pop(cv)
    for i in range(len(folds_vecs_copy)): 
        train.extend(folds_vecs_copy[i])
    train_test['train'] = numpy.stack(train, axis=0 )
    train_test['test'] = numpy.stack(test, axis=0 )
    train_tests_1.append(train_test)
#vect_rh_adlmci_view_2_cv_3_train_real
train_combined = []
for i in range(cv_number):
    train_combined_elt = []
    train_combined_elt.append(train_tests_0[i]['train'])
    train_combined_elt.append(train_tests_1[i]['train'])
    train_combined.append(train_combined_elt)

test_combined = []
for i in range(cv_number):
    test_combined_elt = {}
    test_combined_list = []
    label_cv = []
    for g in range(len(folds_graphs_0[i])):
        label_cv.append(folds_graphs_0[i][g]['label'])
    test_combined_list.append(train_tests_0[i]['test'])
    test_combined_list.append(train_tests_1[i]['test'])
    test_combined_elt['labels'] = label_cv
    test_combined_elt['test'] = test_combined_list
    test_combined.append(test_combined_elt)

ps = "3"
'''
for i in range(cv_number):
    train = train_combined[i]
    test = test_combined[i]
    with open('data processed/'+args_dataset+'_v_01_cv'+str(i)+'_train.pickle', 'wb') as f:
        pickle.dump(train, f)
    with open('data processed/'+args_dataset+'_v_01_cv'+str(i)+'_test.pickle', 'wb') as f:
        pickle.dump(test, f)
    
'''