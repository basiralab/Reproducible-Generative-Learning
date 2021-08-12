# -*- coding: utf-8 -*-
"""
Created on Wed Jun 23 15:10:59 2021

@author: Mohammed Amine
"""
import pickle
import os 
import numpy as np 
import sklearn.metrics as metrics
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import seaborn as sns

datasets = ['Demo']
#datasets = ['RH_ADLMCI', 'LH_ADLMCI']
dataset_idx = 3
for dataset_idx in range(len(datasets)): 
    fig = figure(num=None, figsize=(17, 10))
    
    '''
    sub = 1
    '''
    view = 0
    plt.subplot(2, 1, 1)
    with open('rep_dict_'+datasets[dataset_idx]+'_view_'+str(view)+'.pickle', 'rb') as f:
        rep_dict = pickle.load(f)
    rep_real = rep_dict['real']
    rep_gen = rep_dict['generated']
    maxs = [max(rep_real), max(rep_gen)]
    mins = [min(rep_real), min(rep_gen)]
    width = 0.3
    x = np.arange(10)
    low = min(mins)
    high = max(maxs)
    plt.bar(x - 0.5*width, rep_real, width, color = ('purple'), label = "Ground-truth")
    plt.bar(x + 0.5*width, rep_gen, width, color = ('orange'), label = "Generated")
    plt.xticks(x, ["Diffpool", "Diffpool\nfew shot", "GAT", "GAT\nfew shot", "GCN", "GCN\nfew shot", "SAG", "SAG\nfew shot", "GUNET", "GUNET\nfew shot"])
    plt.ylabel("A|   ", fontsize=24, rotation=0)
    plt.ylim([low-0.2*(high-low),high+0.1*(high-low)])
    plt.title('View 1', fontsize=24)
    plt.legend()
    
    '''
    sub = 2
    '''
    view = 1
    plt.subplot(2, 1, 2)
    with open('rep_dict_'+datasets[dataset_idx]+'_view_'+str(view)+'.pickle', 'rb') as f:
        rep_dict = pickle.load(f)
    rep_real = rep_dict['real']
    rep_gen = rep_dict['generated']
    maxs = [max(rep_real), max(rep_gen)]
    mins = [min(rep_real), min(rep_gen)]
    width = 0.3
    x = np.arange(10)
    low = min(mins)
    high = max(maxs)
    plt.bar(x - 0.5*width, rep_real, width, color = ('purple'), label = "Ground-truth")
    plt.bar(x + 0.5*width, rep_gen, width, color = ('orange'), label = "Generated")
    plt.xticks(x, ["Diffpool", "Diffpool\nfew shot", "GAT", "GAT\nfew shot", "GCN", "GCN\nfew shot", "SAG", "SAG\nfew shot", "GUNET", "GUNET\nfew shot"])
    #plt.ylabel("view 2", fontsize=18)
    plt.ylim([low-0.2*(high-low),high+0.1*(high-low)])
    plt.title('View 2', fontsize=24)
    plt.legend()
    
    fig.suptitle(datasets[dataset_idx],fontsize=24)    
    
    plt.show()

#mymodel = np.poly1d(np.polyfit(x_d, all_d, 3))
#sns.lineplot(x=x_d, y=mymodel(all_d))