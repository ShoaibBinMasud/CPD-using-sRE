# -*- coding: utf-8 -*-
import numpy as np
from scipy.io import savemat
from scipy.stats import multivariate_normal as mvn

no_instance = 2
d = 10
xcoords = [300, 700, 1200, 1500, 1900, 2200, 2400, 2700, 2900]
def generate_sequence():
    mu, cov, cov_7  = np.ones(d), np.eye(d), np.eye(d)
    cov_7[cov_7!=1]= 0.2  
    segment_0 = mvn.rvs(0 * mu, 0.001 * cov, 300)
    segment_1 = mvn.rvs(0 * mu, 0.01 * cov, 400)
    segment_2 = mvn.rvs(mu, cov, 500)
    segment_3 = np.array([np.random.laplace(0, 1, 300)  for i in range(d)]).T
    segment_4 = mvn.rvs(mu, cov, 400)
    segment_5 = np.array([np.random.gamma(2, 2, 300)  for i in range(d)]).T
    segment_6 = mvn.rvs(0 * mu, 0.1 * cov, 200)
    segment_7 = mvn.rvs(mu, cov_7, 300)
    segment_8 = mvn.rvs(0 * mu, 0.01 * cov, 200)
    segment_9 = mvn.rvs(0 * mu, 0.001 *cov, 400)
    X = np.concatenate((segment_0, segment_1, segment_2, segment_3, segment_4,segment_5,
                        segment_6,segment_7, segment_8, segment_9), axis = 0)
    return X


for i in range(no_instance):
    data = dict()
    data['Y'] = generate_sequence()
    gtLabel = np.zeros(len(data['Y']))
    gtLabel[xcoords] = 1
    data['L'] = gtLabel
    savemat('synthetic_'+str(i)+'.mat', data)
    

