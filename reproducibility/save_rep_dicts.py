# -*- coding: utf-8 -*-
"""
Created on Wed Jun 23 15:03:27 2021

@author: Mohammed Amine
"""


import pickle
import numpy as np
import os 
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sn
import matplotlib
import sklearn.metrics as metrics
from matplotlib.pyplot import figure
import pandas as pd
import seaborn as sns
import Analysis

def generate_rep_lists(dataset,view):
    rep = Analysis.Rep_scores(dataset+'_real', view)
    rep_gen = Analysis.Rep_scores(dataset+"_generated", view)
    return {'real':rep, 'generated':rep_gen}

datasets = ['Demo']

for dataset_idx in range(len(datasets)):
    for view in range(2):
        dataset = datasets[dataset_idx]
        dict_df = generate_rep_lists(dataset,view)
        with open('rep_dict_'+datasets[dataset_idx]+'_view_'+str(view)+'.pickle', 'wb') as f:
            pickle.dump(dict_df, f)

'''
rep_real = rep_dict['real']
rep_gen = rep_dict['generated']

maxs = [max(rep_real), max(rep_gen)]
mins = [min(rep_real), min(rep_gen)]

width = 0.3

x = np.arange(10)
low = min(mins)
high = max(maxs)

figure(figsize=(8, 6), dpi=80)
#fig, axs = plt.subplots(1, 1, figsize=(40, 3), sharey=True)

plt.bar(x - 0.5*width, rep_real, width, color = ('purple'), label = "Ground-truth")
plt.bar(x + 0.5*width, rep_gen, width, color = ('orange'), label = "Generated")

plt.xticks(x, ["Diffpool", "Diffpool\nfew shot", "GAT", "GAT\nfew shot", "GCN", "GCN\nfew shot", "SAG", "SAG\nfew shot", "GUNET", "GUNET\nfew shot"])

plt.ylabel("reproducibility score", fontsize=12)
plt.ylim([low-0.2*(high-low),high+0.1*(high-low)])
plt.legend()

#plt.savefig("Bar_plot_dataset_{}_view_{}.png".format(dataset, view))
plt.title(dataset + ' view ' + str(view))
plt.show()
'''    