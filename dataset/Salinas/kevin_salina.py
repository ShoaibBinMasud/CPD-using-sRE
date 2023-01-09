# -*- coding: utf-8 -*-
"""
Created on Mon Jul 18 00:14:12 2022

@author: shakt
"""
# kevins

import glob
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
import os
from scipy.io import loadmat
# from metrics import compute_auc, f1_score
import scipy.signal as scisig

    
def F_GInv(F,G):
    F=F.flatten()
    G=G.flatten()
    n = len(F)
    m = len(G)
    Fsort = np.sort(F)
    Gsort = np.sort(G)
    outX = np.zeros(m)
    outY = np.zeros(m)
    for i in range(m):
        dist = np.argwhere(Fsort <= Gsort[i])
        outY[i] = len(dist)/n
        outX[i] = (i+1)/m # cdf jumps at points in Gm
    return (outX, outY)

def DistanceToUniform(p, cdf):
    #This only computes distance, we need squared distance. 
    # we assume that [0,0], and [1,1] are not included
    p=np.append(p,1)
    cdf = np.append(cdf,1)
    prevX = 0
    prevY = 0
    total = 0
    overUnder = 0

    for i in range(len(p)):
        if (cdf[i] < p[i]): # we are under
            if overUnder == 1: # we were over
                total += (np.abs(prevX-prevY) + (p[i]-prevY))/2 * (p[i]-prevX) # trapezoid
            elif overUnder !=-1: # we stayed under
                total += (np.abs(prevX-prevY) + (cdf[i]-prevY))/2 *(p[i]-prevX)
            overUnder = 1
        elif (cdf[i] < p[i]): # we are over
            if overUnder == -1: # and now we are under
                total += (np.abs(prevX-prevY) + 0)/2 * (prevY - prevX)
                total += (0 + (p[i]-cdf[i]))/2 * (p[i] - prevY)
            elif overUnder !=-1: # we are still over
                # we need to check if we fell under for some part
                if (p[i] < prevY): # if we did we have to integrate 2 smaller triangles
                    total += (np.abs(prevX-prevY))/2 * (prevY-prevX)
                    total += (p[i]-prevY)/2 * (p[i]-prevY)
                else:
                    total += (np.abs(prevY-prevX)+(cdf[i] - p[i]))/2 * (p[i]-prevX)
            overUnder = 0
        else:
            total+= np.abs(prevY-prevX)/2*(p[i]-prevX)
        
        prevX=p[i]
        prevY = cdf[i]
    return total

def DistanceSquaredToUniform(pp,cdf,step= 0.01):
    pp=np.append(pp,1)
    cdf = np.append(cdf,1)
    xAll = np.linspace(0,1,int(1/step)+1)
    total = 0 
    for x in xAll:
        argX = np.argwhere(pp>=x)
        total += (x - cdf[argX[0]])*(x - cdf[argX[0]])*step
    return total

def TwoSampleWTest(sampA, sampB, step = None):
    lenA = len(sampA)
    lenB = len(sampB)
    (cdfX, cdfY) = F_GInv(sampA,sampB)
            
    if (step is None):
        distOut = DistanceSquaredToUniform(cdfX, cdfY)*(lenA*lenB)/(lenA+lenB)
    else:
        distOut = DistanceSquaredToUniform(cdfX, cdfY, step=step)*(lenA*lenB)/(lenA+lenB)
    return distOut

def Compute2SampleWStat(dat, window, stride):
    lenDat = len(dat)
    dim=len(dat[0])
    out = np.zeros((int(np.floor(lenDat/stride)), dim))
    count = 0
    for i in tqdm(range(0,lenDat-stride,stride)):
        for j in range(dim):
            if (i<window or i >= lenDat-window):
                out[count,j]=0
            else:        
                win1 = dat[i-window:i,j]
                win2 = dat[i:i+window,j]
                out[count,j] = TwoSampleWTest(win1, win2)
        count = count+1
    outSingle = np.mean(out,axis=1)
    return outSingle 


'''Salinas images'''
# data = loadmat('SalinasA.mat')['salinasA']
# gt = loadmat('SalinasA_gt.mat')['salinasA_gt']
# data = loadmat('salinas_pixel_remove.mat')
data = loadmat('salinas_not_removed_8.mat')

X = data['Y']
gtLabel = data['L']
# X = data.transpose(1, 0, 2).reshape(data.shape[0]* data.shape[1], -1)[:500, :]
# X = data['Y']
cpd_stat =  Compute2SampleWStat(X, 10, 1) 
# with open('salina_pixel_remove_w2stat.pkl', 'wb') as f: pickle.dump(cpd_stat, f)
with open('salina_w2stat_not_removed_pca_8.pkl', 'wb') as f: pickle.dump(cpd_stat, f)
'''UCR dataset'''
# dataset = 'UCR'
# dataDir = os.getcwd()+'/datasets/'+ dataset 
# files = glob.glob(dataDir + '/*.mat')
# data = loadmat(files[0])
# X, gtLabel = data['Y'], data['L'].T
# # n_List = [25, 50, 100]
# # cpd_stat = dict()  
# # for n in n_List:
# #    cpd_stat[n] =  Compute2SampleWStat(X, n, 1) 
# # with open('ecg_w2stat.pkl', 'wb') as f: pickle.dump(cpd_stat, f)

# with open('ecg_w2stat.pkl', 'rb') as f: result = pickle.load(f)
# z = result[50]
# aucVal, fpr, tpr, thresholds = compute_auc(gtLabel, z)
# f1_temp = 0
# th_temp = 0
# thr_List = np.linspace(z.min(), z.max(), 100)

# for th in thr_List:
#     f1, precision, recall = f1_score(gtLabel, z, thr = th, tol_dist = 40, detect_margin = 30)
#     if f1>= f1_temp:
#         f1_temp = f1
#         th_temp = th
# peakIndex = scisig.find_peaks(z, height =th_temp ,  distance = 50)[0]
# print(aucVal, f1_temp)

'''Libras Movement dataset'''
# dataset = 'libras'
# dataDir = os.getcwd()+'datasets'+ dataset 

# data =  pd.read_csv('movement_libras.data', sep=",")
# X  = np.array(data.iloc[: , :90])
# label  = data.iloc[: , -1]
# y = np.zeros_like(label)
# for i in range(1, 15):
#     ind = np.where(label== i)[0][-1]
#     y[ind+1] = 1
# gtLabel = y

# n_List =[10, 20, 30]
# cpd_stat = pd.DataFrame()
# for n in n_List:
#     cpd_stat[n] = Compute2SampleWStat(X, n, 1) 
# with open('libras_w2stat.pkl', 'wb') as f: pickle.dump(cpd_stat, f)
# n = 20
# tol_margin = 10
# detect_margin = 10

# z =  Compute2SampleWStat(X, 50, 1) 
# thr_List = np.linspace(z.min(), z.max(), 1000)
# f1_temp = 0
# th_temp = 0
# for th in thr_List:
#     f1, precision, recall = f1_score(gtLabel, z, thr = th, tol_dist = tol_margin, detect_margin = detect_margin)
#     if f1>= f1_temp:
#         f1_temp = f1
#         th_temp = th
# aucVal, fpr, tpr, thresholds = compute_auc(gtLabel, z)
# # _, _, aucVal, _, _  = ChangePointMetrics_AUCpeaks(z, gtLabel.ravel(),margin = detect_margin,
#                                                                             # tol = tol_margin)
# print('w2stat:', aucVal, f1_temp)   


'''Beedance dataset'''
# dataset = 'beedance'
# dataDir = os.getcwd()+'/datasets/'+ dataset
# files = glob.glob(dataDir + '/*.mat')
# cpd_stat = dict()
# for file in files:
#     data = loadmat(file)
#     X = data['Y']
#     gtLabel = data['L']
#     if dataset == 'beedance':
       
#        s, e = np.where(gtLabel == 1)[0][0], np.where(gtLabel == 1)[0][-1]+1
#        gtLabel = np.concatenate((gtLabel[:s], gtLabel[:s], gtLabel[:s], gtLabel[:s], gtLabel,
#                                  gtLabel[e:], gtLabel[e:], gtLabel[e:], gtLabel[e:]), axis = 0)
#        X = np.concatenate(( X[:s], X[:s], X[:s], X[:s], X, X[e:], X[e:],X[e:], X[e:]), axis = 0 )
#     w2stat = dict()
#     for n in n_List:
#         w2stat[n] =  Compute2SampleWStat(X, n, 1)      
#     cpd_stat[file[-15:-4]] =w2stat
#     print(file, '\n')
# with open('beedance_w2stat.pkl', 'wb') as f: pickle.dump(cpd_stat, f)

# dataset = 'UCR'
# dataDir = os.getcwd()+'/datasets/'+ dataset 
# files = glob.glob(dataDir + '/*.mat')
# data = loadmat(files[0])
# X, gtLabel = data['Y'], data['L'].T
# xcoords = np.where(gtLabel== 1)[0]
# T = np.arange(len(X))

# tol_margin = 25
# detect_margin = 20

# z =  Compute2SampleWStat(X, 50, 1) 
# thr_List = np.linspace(z.min(), z.max(), 1000)
# f1_temp = 0
# th_temp = 0
# for th in thr_List:
#     f1, precision, recall = f1_score(gtLabel, z, thr = th, tol_dist = tol_margin, detect_margin = detect_margin)
#     if f1>= f1_temp:
#         f1_temp = f1
#         th_temp = th
# aucVal, fpr, tpr, thresholds = compute_auc(gtLabel, z)
# # _, _, aucVal, _, _  = ChangePointMetrics_AUCpeaks(z, gtLabel.ravel(),margin = detect_margin,
#                                                                             # tol = tol_margin)
# print('w2stat:', aucVal, f1_temp)   



  








