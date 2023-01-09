# -*- coding: utf-8 -*-
import argparse
import utils
import numpy as np
import pandas as pd
from scipy.io import loadmat
from TwoSampleTestStats import RE, sRE, W1, WQT, MMD, Sinkdiv
import warnings
warnings.simplefilter("ignore")

parser = argparse.ArgumentParser()
parser = utils.get_args(parser)
args = parser.parse_args()
dataset = utils.get_data(args)
result = pd.DataFrame(columns=['Method'])

methods = ['RE','sRE', 'W1', 'WQT', 'MMD', 'Sinkdiv']
for method in methods:
    func = globals()[method]
    print(func)
    auc_pr_List, f1_List = list(), list()
    for sequence in dataset:
        data = loadmat(sequence)
        X, gtLabel = data['Y'], data['L']
        if method =='sRE': z =func(X, n = args.window, eps = args.epsilon)
        else:              z =func(X, n = args.window) 
        auc_pr, f1 = utils.compute_auc_f1(gtLabel.ravel(), z,   DELTA = args.DELTA, xi = args.xi)
        auc_pr_List.append(auc_pr)
        f1_List.append(f1)
    result = result.append({'Method': method, 'AUC-PR':  np.mean(auc_pr_List),
                                  'F1-score':np.mean(f1_List)}, ignore_index = True)

with pd.ExcelWriter('result.xlsx', mode='a', engine='openpyxl', if_sheet_exists='replace') as writer:  
    result.to_excel(writer, sheet_name=args.dataset )
