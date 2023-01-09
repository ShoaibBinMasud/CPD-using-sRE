import numpy as np
import ghalton
from tqdm import tqdm
import ot
import torch
from get_plan import plan

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
    
def rankEnergy(x, y, eps = 1):
    '''
     x: left window
     y: right window
     eps: regularization parameter
             ## Rank energy eps ==0, soft rank energy: eps>0 
    '''
    assert x.shape == y.shape
    n, d = x.shape
    concat_xy = np.concatenate((x, y), axis  = 0)
    sequencer = ghalton.Halton(d)
    h = np.array(sequencer.get(2 * n))
    a , b = ot.unif(2 * n), ot.unif(2 *n)
    if eps ==0:
        M = ot.dist(concat_xy, h, metric ='sqeuclidean')
        M /= M.max()  
        otPlan = ot.emd(a, b, M)  
        R = np.argmax(otPlan, axis = 1)
        Rx, Ry = torch.from_numpy(h[R[:n]]), torch.from_numpy(h[R[n:]])
        RE = testStatistics (Rx, Ry)
        return RE.cpu().detach().numpy() 
    
    elif eps == 'W1':
        #https://pythonot.github.io/quickstart.html#computing-wasserstein-distance
        a, b = ot.unif(n), ot.unif(n)
        M = ot.dist(x, y, metric = 'euclidean')
        W1 = ot.emd2(a, b, M)
        return W1
    else:
        a, b = torch.from_numpy(a).to(device), torch.from_numpy(b).to(device)
        concat_xy = torch.from_numpy(concat_xy).to(device)
        h = torch.from_numpy(h).to(device)
        entropicOtPlan = plan(a, concat_xy, b, h, p=2, eps = eps)
        row_sum = entropicOtPlan.sum(axis = 1)
        scaled_plan = entropicOtPlan / row_sum[:, np.newaxis]
        R = torch.mm(scaled_plan, h)
        sRE = testStatistics (R[:n], R[n:])
        return sRE.cpu().detach().numpy() 
    
def cpdWindowStat(data, n = 250, eps = 1, s = 1):
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
            x, y = data[i-n:i, :], data[i: i+n, :]
            cpd_stat[count] = rankEnergy(x, y, eps = eps)
        count += 1
    return cpd_stat   