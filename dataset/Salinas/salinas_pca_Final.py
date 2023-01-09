# -*- coding: utf-8 -*-
"""
Created on Wed Jan  4 22:31:54 2023

@author: shakt
"""

from scipy.io import loadmat,savemat
import pandas as pd
import pickle
import numpy as np
from sklearn.decomposition import PCA
from cpd_utils import cpdWindowStat
import torch
import ot
from tqdm import tqdm
from sinkhorn_div import sinkhorn_divergence
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
def extract_pixels(X, y):
    q = X.reshape(-1, X.shape[2])
    df = pd.DataFrame(data = q)
    df = pd.concat([df, pd.DataFrame(data = y.ravel())], axis=1)
    df.columns= [f'band{i}' for i in range(1, 1+X.shape[2])]+['class']
    # df.to_csv('Dataset.csv')
    return df

X = loadmat('SalinasA.mat')['salinasA'].transpose(1, 0, 2)
y = loadmat('SalinasA_gt.mat')['salinasA_gt'].T

no_comp = 8
df = extract_pixels(X, y)

pca = PCA(n_components = no_comp)
dt = pca.fit_transform(df.iloc[:, :-1].values)
q = pd.concat([pd.DataFrame(data = dt), pd.DataFrame(data = y.ravel())], axis = 1)
q.columns = [f'PC-{i}' for i in range(1,9)]+['class']


gt = q.iloc[:, -1].values
gtLabel = np.zeros_like(gt)
diff = np.array([int(y)-int(x) for x, y in zip(gt[:-1], gt[1:])])
diff[diff != 0] = 1
ind = np.where(np.array(diff)== 1)[0]
gtLabel[ind+1 ] = 1
xcoords = np.where(gtLabel == 1)[0]
x_pca = dict()
x_pca['Y'] = q.iloc[:, :-1].values[:500, :]
x_pca['L'] = gtLabel[:500]
savemat('salinas_not_removed_pca_'+str(no_comp)+'.mat', x_pca )

X = x_pca['Y']
n = 10
### sre, W1
eps_List = [0, 0.1, 1, 2, 10, 'W1']
sRE_stat = pd.DataFrame()
for eps in eps_List:
    sRE_stat[eps] = cpdWindowStat(X, n = n, eps = eps)
  

# ## sINKDIV
def sinkdiv(data, n = 250, eps = 1, s = 1):
    '''
      n= window size
      eps = entropic regulaizer parameter
      s: amount of sliding
    '''
    cpd_stat = np.zeros(len(data)// s, )
    count = 0
    for i in tqdm(range(0, len(data)-s, s),  desc = 'n: ' + str(n)+' eps:'+str(eps)):
        if i<n or i>= len(data)- n:
            cpd_stat[count] = 0
        else:
            a, b = ot.unif(n), ot.unif(n)
            x, y = data[i-n:i, :], data[i: i+n, :]

            a, b = torch.from_numpy(a).to(device), torch.from_numpy(b).to(device)
            x, y = torch.from_numpy(x).to(device), torch.from_numpy(y).to(device)
            cpd_stat[count] = sinkhorn_divergence(a, x, b, y, 2, eps).cpu().detach().numpy()
        count += 1
    return cpd_stat 

eps_List = [0.1, 1, 10]
div_stat = pd.DataFrame()
for eps in eps_List:
    div_stat[eps] = sinkdiv(X, n = n, eps = eps)


## MMD
def mmd(x, y, sigma = 0.5):
    n, d = x.shape
    m, d2 = y.shape
    assert d == d2
    x = torch.from_numpy(x)
    y = torch.from_numpy(y)
    xy = torch.cat([x.detach(), y.detach()], dim=0)
    dists = torch.cdist(xy, xy, p=2.0)
    k = torch.exp((-1/(2*sigma**2)) * dists**2) + torch.eye(n+m)*1e-5
    k_x = k[:n, :n]
    k_y = k[n:, n:]
    k_xy = k[:n, n:]
    mmd = k_x.sum() / (n * (n - 1)) + k_y.sum() / (m * (m - 1)) - 2 * k_xy.sum() / (n * m)
#     print(mmd)
    return mmd

def mmd_stat_f(data, n = 250, sigma = 1, s = 1):
    '''
     n= window size
     eps = entropic regulaizer parameter
     s: amount of sliding
    '''
    
    cpd_stat = np.zeros(len(data)// s, )
    count = 0
    for i in tqdm(range(0, len(data)-s, s),  desc = 'n: ' + str(n)+' sigma:'+str(sigma)):
        if i<n or i>= len(data)- n:
            cpd_stat[count] = 0
        else:
            x, y = data[i-n:i, :], data[i: i+n, :]
            cpd_stat[count] = mmd(x,y,sigma)
#         print(cpd_stat[count])
        count += 1
    return cpd_stat  

simga_List = [0.01, 0.1, 0.5, 1, 10]
X = X.astype(float)
mmd_stat = pd.DataFrame()
for sigma in simga_List:
    mmd_stat[sigma] = mmd_stat_f(X, n = n, sigma = sigma)
    
# WQT
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
wqt_stat =  Compute2SampleWStat(X, 10, 1) 
##

result = dict()
result['sRE'] = sRE_stat
result['div'] = div_stat
result['mmd'] = mmd_stat
result['wqt'] = wqt_stat
with open('salina_not_removed_pca'+ str(no_comp) +'.pkl', 'wb') as f: pickle.dump(result, f)