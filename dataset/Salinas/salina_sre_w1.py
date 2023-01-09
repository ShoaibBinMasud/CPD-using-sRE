# -*- coding: utf-8 -*-
from scipy.io import loadmat
import pandas as pd
import pickle
# from cpd_utils import cpdWindowStat

data = loadmat('salinas_not_removed_8.mat')

X = data['Y']
gtLabel = data['L']
for i in [1, 2,3]:
    print(16, 32, 64) 

# n_List = [10, 20]
# eps_List = [0, 0.1, 1, 2, 10, 'W1']
# sRE_stat = dict()
# for n in n_List:
#     sRE_stat_eps = pd.DataFrame()
#     for eps in eps_List:
#         sRE_stat_eps[eps] = cpdWindowStat(X, n = n, eps = eps)
#     sRE_stat[n] = sRE_stat_eps
# with open('salina_pixel_remove_sre_w1.pkl', 'wb') as f: pickle.dump(sRE_stat, f)