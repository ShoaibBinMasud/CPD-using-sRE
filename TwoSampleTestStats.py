import torch
import numpy as np
import ghalton
import ot
from tqdm import tqdm
from get_plan import plan
from sinkhorn_div import divergence
from sklearn.metrics.pairwise import euclidean_distances
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def pairwiseDistance(X, Y):
    '''calculating pairwise distance'''
    x_col = X.unsqueeze(1)
    y_lin = Y.unsqueeze(0)
    M = torch.sqrt(torch.sum((x_col - y_lin)**2 , 2))
    return M
 
def testStatistics(X, Y):
    '''
    RankEnergy(X, Y) = (2/mn)E||R(X)-R(Y)|| - (1/n^2) E||R(X)- R(X')|| - (1/m^2) E||R(Y)- R(Y')||
    '''
    n, m = X.shape[0], Y.shape[0]
    assert n==m
    coefficient = n * m / (n + m)
    xx = pairwiseDistance(X + 1e-16, X) # to avoid 'divide by zero error'
    yy = pairwiseDistance(Y + 1e-16 , Y)
    xy = pairwiseDistance(X, Y)
    rank_energy = coefficient * ( 2 * torch.mean(xy) - torch.mean(xx) - torch.mean(yy))
    return rank_energy

def RankEenrgy(x, y):

    assert x.shape == y.shape
    n, d = x.shape
    concat_xy = np.concatenate((x, y), axis  = 0)
    sequencer = ghalton.Halton(d)
    h = np.array(sequencer.get(2 * n))
    a , b = ot.unif(2 * n), ot.unif(2 *n)
    M = ot.dist(concat_xy, h, metric ='sqeuclidean')
    M /= M.max()  
    otPlan = ot.emd(a, b, M)  
    R = np.argmax(otPlan, axis = 1)
    Rx, Ry = torch.from_numpy(h[R[:n]]), torch.from_numpy(h[R[n:]])
    RE = testStatistics (Rx, Ry)
    return RE.cpu().detach().numpy() 

'''RANK ENERGY'''
def RE(data, n = 250, s = 1):
    '''
     n= window size
     eps = entropic regulaizer parameter
     s: amount of sliding
    '''
    cpd_stat = np.zeros(len(data)// s, )
    count = 0
    for i in tqdm(range(0, len(data)-s, s),  desc = 'RE=>   n: ' + str(n)):
        if i<n or i>= len(data)- n:
            cpd_stat[count] = 0
        else:
            x, y = data[i-n:i, :], data[i: i+n, :]
            cpd_stat[count] = RankEenrgy(x, y)
        count += 1
    return cpd_stat  

"""SOFT RANK ENERGY"""
def SoftRankEenrgy(x, y, eps = 1):
    assert x.shape == y.shape
    n, d = x.shape
    concat_xy = np.concatenate((x, y), axis  = 0)
    sequencer = ghalton.Halton(d)
    h = np.array(sequencer.get(2 * n))
    a , b = ot.unif(2 * n), ot.unif(2 *n)
    a, b = torch.from_numpy(a).to(device), torch.from_numpy(b).to(device)
    concat_xy = torch.from_numpy(concat_xy).to(device)
    h = torch.from_numpy(h).to(device)
    entropicOtPlan = plan(a, concat_xy, b, h, p=2, eps = eps)
    row_sum = entropicOtPlan.sum(axis = 1)
    scaled_plan = entropicOtPlan / row_sum[:, np.newaxis]
    R = torch.mm(scaled_plan, h)
    sRE = testStatistics (R[:n], R[n:])
    return sRE.cpu().detach().numpy() 

def sRE(data, n = 250, eps = 1, s = 1):
    cpd_stat = np.zeros(len(data)// s, )
    count = 0
    for i in tqdm(range(0, len(data)-s, s),  desc = 'sRE=>    n: ' + str(n)+' eps:'+str(eps)):
        if i<n or i>= len(data)- n:
            cpd_stat[count] = 0
        else:
            x, y = data[i-n:i, :], data[i: i+n, :]
            cpd_stat[count] = SoftRankEenrgy(x, y, eps)
        count += 1
    return cpd_stat
 
"""Wasserstein-1"""

def Wasserstein_1(x, y):
    #https://pythonot.github.io/quickstart.html#computing-wasserstein-distance
    assert x.shape == y.shape
    n, d = x.shape
    a, b = ot.unif(n), ot.unif(n)
    M = ot.dist(x, y, metric = 'euclidean')
    W1 = ot.emd2(a, b, M)
    return W1

def W1(data, n = 250, s = 1):
    '''
     n= window size
     eps = entropic regulaizer parameter
     s: amount of sliding
    '''
    cpd_stat = np.zeros(len(data)// s, )
    count = 0
    for i in tqdm(range(0, len(data)-s, s),  desc = 'W1=>  n: ' + str(n)):
        if i<n or i>= len(data)- n:
            cpd_stat[count] = 0
        else:
            x, y = data[i-n:i, :], data[i: i+n, :]
            cpd_stat[count] = Wasserstein_1(x, y)
        count += 1
    return cpd_stat   

"""a distribution-free version of Wasserstein distance"""
# https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9246231

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

def WQT(data, n = 250, s = 1):
    lenDat = len(data)
    dim=len(data[0])
    out = np.zeros((int(np.floor(lenDat/s)), dim))
    count = 0
    for i in tqdm(range(0,lenDat-s,s), desc = 'WQT=>   n: ' + str(n)):
        for j in range(dim):
            if (i<n or i >= lenDat-n):
                out[count,j]=0
            else:        
                x = data[i-n:i,j]
                y = data[i:i+n,j]
                out[count,j] = TwoSampleWTest(x, y)
        count = count+1
    cpd_stat = np.mean(out,axis=1)
    return cpd_stat 

"""SINKHORN DIVERGENCE"""

def Sinkdiv(data, n = 250, eps = 1, s = 1):
    cpd_stat = np.zeros(len(data)// s, )
    count = 0
    for i in tqdm(range(0, len(data)-s, s),  desc = 'SinkDiv=>   n: ' + str(n)):
        if i<n or i>= len(data)- n:
            cpd_stat[count] = 0
        else:
            x, y = data[i-n:i, :], data[i: i+n, :]
            cpd_stat[count] = divergence(x, y, eps)
        count += 1
    return cpd_stat 

"""MAXIMUM MEAN DISCREPANCY"""

def median_heuristic(X):
    max_n = min(30000, X.shape[0])
    D2 = euclidean_distances(X[:max_n], squared=True)
    med_sqdist = np.median(D2[np.triu_indices_from(D2, k=1)])
    return med_sqdist

def MaximumMeanDis(x, y, sigma):

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
    return mmd

def MMD(data, n = 250, eps = 1, s = 1):
    '''
     n= window size
     eps = entropic regulaizer parameter
     s: amount of sliding
    '''
    cpd_stat = np.zeros(len(data)// s, )
    count = 0
    sigma = median_heuristic(data)
    for i in tqdm(range(0, len(data)-s, s),  desc = 'MMD=>   n: ' + str(n)):
        if i<n or i>= len(data)- n:
            cpd_stat[count] = 0
        else:
            x, y = data[i-n:i, :], data[i: i+n, :]
            cpd_stat[count] = MaximumMeanDis(x, y, sigma)
        count += 1
    return cpd_stat 