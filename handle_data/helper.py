# -*- coding: utf-8 -*-
"""
Created on Tue Jun  8 18:02:44 2021

@author: user
"""

import pickle
import numpy as np
import math


def stratify_splits(graphs, cv_number):
    graphs_0 = []
    graphs_1 = []
    for i in range(len(graphs)):
        if graphs[i]['label'] == 0:
            graphs_0.append(graphs[i])
        if graphs[i]['label'] == 1:
            graphs_1.append(graphs[i])
    graphs_0_folds = []
    graphs_1_folds = []
    pop_0_fold_size = math.ceil(len(graphs_0) / cv_number)
    pop_1_fold_size = math.ceil(len(graphs_1) / cv_number)
    graphs_0_folds = [graphs_0[i:i + pop_0_fold_size] for i in range(0, len(graphs_0), pop_0_fold_size)]
    graphs_1_folds = [graphs_1[i:i + pop_1_fold_size] for i in range(0, len(graphs_1), pop_1_fold_size)]
    folds = []
    for i in range(cv_number):
        fold = []
        fold.extend(graphs_0_folds[i])
        fold.extend(graphs_1_folds[i])
        folds.append(fold)
    return folds
'''    
def vec_to_mat(vec, n):
    a = np.zeros((n, n)) # Initialize nxn matrix
    triu = np.triu_indices(n, k=1) # Find upper right indices of a triangular nxn matrix
    tril = np.tril_indices(n, k=1) # Find lower left indices of a triangular nxn matrix
    a[tril] = vec
    a[triu] = a.T[triu] # Make the matrix symmetric
    return a    
'''
def vec_to_mat(vec, n):
    a = np.zeros((n,n))
    k = 0
    for i in range(n):
        for j in range(n):
            if i > j :
                a[i,j] = vec[k]
                k += 1
    a = a + np.transpose(a)
    return a

def mat_to_vec(mat, n):
    vec = mat[np.tril_indices(n,k=-1)]
    return vec


