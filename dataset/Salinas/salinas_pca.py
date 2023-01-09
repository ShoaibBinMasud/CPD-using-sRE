# -*- coding: utf-8 -*-
"""
Created on Wed Jan  4 20:44:03 2023

@author: shakt
"""
import os
from scipy.io import loadmat, savemat
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import pickle

ncol, nrow =  4, 1
fig, axs = plt.subplots(ncol, nrow, figsize = (32, 20), sharex = True)
plt.subplots_adjust(top=0.92, bottom=0.013, left=0.04, right=0.98, hspace=.2, wspace=.1)
plt.rc('xtick', labelsize=20)  
plt.rc('ytick', labelsize=20) 

X = loadmat('SalinasA.mat')['salinasA'].transpose(1, 0, 2)
y = loadmat('SalinasA_gt.mat')['salinasA_gt'].T

def extract_pixels(X, y):
    q = X.reshape(-1, X.shape[2])
    df = pd.DataFrame(data = q)
    df = pd.concat([df, pd.DataFrame(data = y.ravel())], axis=1)
    df.columns= [f'band{i}' for i in range(1, 1+X.shape[2])]+['class']
    # df.to_csv('Dataset.csv')
    return df
df = extract_pixels(X, y)

pca = PCA(n_components = 75)

principalComponents = pca.fit_transform(df.iloc[:, :-1].values)

ev=pca.explained_variance_ratio_

# plt.figure(figsize=(12, 6))
# plt.plot(np.cumsum(ev))
# plt.xlabel('Number of components')
# plt.ylabel('Cumulative explained variance')
# plt.show()
no_comp = 16
pca = PCA(n_components = no_comp )
dt = pca.fit_transform(df.iloc[:, :-1].values)
q = pd.concat([pd.DataFrame(data = dt), pd.DataFrame(data = y.ravel())], axis = 1)
q.columns = [f'PC-{i}' for i in range(1,no_comp+1)]+['class']



gt = q.iloc[:, -1].values
gtLabel = np.zeros_like(gt)
diff = np.array([int(y)-int(x) for x, y in zip(gt[:-1], gt[1:])])
diff[diff != 0] = 1
ind = np.where(np.array(diff)== 1)[0]
gtLabel[ind+1 ] = 1
# xcoords = np.where(gtLabel == 1)[0]

x_pca = dict()
x_pca['Y'] = q.iloc[:, :-1].values[:500, :]
x_pca['L'] = gtLabel[:500] #q.iloc[:, -1].values.ravel()
xcoords = np.where(x_pca['L'] == 1)[0]

# savemat('salinas_not_removed_8.mat', x_pca )


T = np.arange(500)
axs[0].set_xlim(0, 500)
axs[0].plot(T, df.iloc[:500, 0].values, 'b-', linewidth = 3)
[axs[0].axvline(x=xc,color = 'red',linestyle='--', linewidth = 3) for xc in xcoords]
for i in range(1, 4):
    [axs[i].axvline(x=xc, color = 'red',linestyle='--',linewidth = 3) for xc in xcoords]
    axs[i].plot(T, x_pca['Y'][:, i-1],'b-', linewidth = 3)
    axs[i].set_xlim(0, 500)
    axs[i].set_ylabel('Comp '+ str(i), fontsize = 30)
axs[3].set_xlabel('Samples', fontsize = 30)
axs[0].set_ylabel('Original image data', fontsize = 24)
axs[0].set_title('PCA dimension = '+ str(no_comp), fontsize = 40)
# with open( os.getcwd()+'salinas_result', 'wb') as f:
#     pickle.dump(x_pca, f)
# fig = plt.figure(figsize = (20, 10))
# for i in range(1, 1+8):
#     fig.add_subplot(2,4, i)
#     plt.imshow(q.loc[:, f'PC-{i}'].values.reshape(86, 83), cmap='nipy_spectral')
#     plt.axis('off')
#     plt.title(f'Band - {i}')
    

