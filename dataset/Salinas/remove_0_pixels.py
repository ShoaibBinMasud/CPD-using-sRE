# -*- coding: utf-8 -*-
"""
Created on Tue Jan  3 09:56:24 2023

@author: shakt
"""

from scipy.io import loadmat, savemat
import numpy as np
import matplotlib.pyplot as plt
plt.rc('xtick', labelsize=30) 
plt.rc('ytick', labelsize=30) 


data = loadmat('SalinasA.mat')['salinasA']
gt = loadmat('SalinasA_gt.mat')['salinasA_gt']

X = data.transpose(1, 0, 2).reshape(data.shape[0]* data.shape[1], -1)[:500, :]
gt = gt.T.reshape(-1)[:500]
zeroIndex = np.where(gt==0)[0]

gt_remove =  np.delete(gt, zeroIndex)
X_remove = np.delete(X, zeroIndex, axis = 0)

 fig, axs = plt.subplots(2, 1, figsize =(24, 5))
plt.subplots_adjust(top=0.92, bottom=0.013, left=0.04, right=0.98, hspace=.50, wspace=.04)


axs[0].plot(X[:, 0], color = 'blue', label = 'Original')
ax0_twin = axs[0].twinx()
ax0_twin.plot(gt, color = 'red', label = 'ground truth')
ax0_twin.legend(fontsize = 20)
ax0_twin.set_ylabel('Class', fontsize = 20)
axs[0].set_ylabel('Pixel values', fontsize = 20)
axs[0].legend(fontsize = 20)
axs[0].set_title('Wtih unlabeled', fontsize = 24)


axs[1].plot(X_remove[:, 0], color = 'blue', label = 'Original')
ax1_twin = axs[1].twinx()
ax1_twin.plot(gt_remove, color = 'red', label = 'ground truth')
ax1_twin.legend(fontsize = 20)
ax1_twin.set_ylabel('Class', fontsize = 20)
axs[1].set_ylabel('Pixel values', fontsize = 20)
axs[1].legend(fontsize = 20)
axs[1].set_title('Wtihout unlabeled', fontsize = 24)


gtLabel = np.zeros_like(gt_remove)
diff = np.array([int(y)-int(x) for x, y in zip(gt_remove[:-1], gt_remove[1:])])
diff[diff != 0] = 1
ind = np.where(np.array(diff)== 1)[0]
gtLabel[ind+1 ] = 1
xcoords = np.where(gtLabel == 1)[0]

data = dict()
data['Y'] = X_remove
data['Lc'] = gt_remove
data['L'] = gtLabel
# savemat('salinas_pixel_remove.mat', data)

