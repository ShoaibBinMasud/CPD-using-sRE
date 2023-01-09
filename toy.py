# -*- coding: utf-8 -*-
import argparse
import utils
import os
import glob
import numpy as np
import pandas as pd
from scipy.io import loadmat
from TwoSampleTestStats import RE, sRE, W1T, W2T, MMD, Sinkdiv

def get_args(parser):
    parser.add_argument('--dataset', required=True, help='synthetic')
    parser.add_argument('--window', type=int,  nargs='+', default= [ 25, 50], help='input window size')
    parser.add_argument('--epsilon',type=float,nargs='+', default=[0.1, 1, 10], help='input regularizer')
    parser.add_argument('--xi',  type=int,             default= 20, help='input detection margin')
    parser.add_argument('--DELTA',  type=int,  nargs='+', default=[25, 50, 100, 200], help='input minimum horizontal distance')
    return parser

def get_data(args):
    if args.dataset== 'synthetic':
        dataroot = os.getcwd()+'/dataset/Synthetic' 
        dataset = glob.glob(dataroot + '/*.mat')
    else:
        raise ValueError("Unknown dataset %s" % (args.dataset))
    return dataset

parser = argparse.ArgumentParser()
parser = get_args(parser)
args = parser.parse_args()
dataset = get_data(args)

auc_pr_table = pd.DataFrame(columns=['Method'])
f1_table = pd.DataFrame(columns=['Method'])
methods = ['MMD', 'Sinkdiv','W1', 'WQT','RE','sRE']
for i in range(len(args.window)):
    auc_pr_df, f1_df = pd.DataFrame(columns=['Method_']), pd.DataFrame(columns=['Method_'])
    n = args.window[i]
    DELTA = args.DELTA[i]
    for method in methods:
        print(method)
        func = globals()[method]
        auc_pr_List, f1_List = list(), list()  
        if method =='sRE': 
            for eps in args.epsilon:
                for sequence in dataset:
                    data = loadmat(sequence)
                    X, gtLabel = data['Y'], data['L'] 
                    z =func(X, n = n, eps = eps)
                    auc_pr, f1 = utils.compute_auc_f1(gtLabel.ravel(), z,  DELTA = DELTA, xi = args.xi)
                    auc_pr_List.append(auc_pr)
                    f1_List.append(f1)
                auc_pr_df = auc_pr_df.append({'Method_': method + str(eps), 'AUC-PR':  np.mean(auc_pr_List),},
                                             ignore_index = True)
                f1_df = f1_df.append({'Method_': method + str(eps), 'F1-score':  np.mean(f1_List),},
                                     ignore_index = True)
        else:   
            for sequence in dataset:
                data = loadmat(sequence)
                X, gtLabel = data['Y'], data['L'] 
                z =func(X, n = n) 
                auc_pr, f1 = utils.compute_auc_f1(gtLabel.ravel(), z,   DELTA = DELTA, xi = args.xi)
                auc_pr_List.append(auc_pr)
                f1_List.append(f1)
            auc_pr_df = auc_pr_df.append({'Method_': method , 'AUC-PR':  np.mean(auc_pr_List),},
                                         ignore_index = True)
            f1_df = f1_df.append({'Method_': method , 'F1-score':  np.mean(f1_List),}, 
                                 ignore_index = True)

    auc_pr_table['Method'] = auc_pr_df.iloc[:, 0]
    auc_pr_table[n] = auc_pr_df.iloc[:, 1]
    f1_table['Method'] = f1_df.iloc[:, 0]
    f1_table[n] = f1_df.iloc[:, 1]

with pd.ExcelWriter('sythetic.xlsx') as writer:  
    auc_pr_table.to_excel(writer, sheet_name='AUC_-PR')
    f1_table.to_excel(writer, sheet_name='Best F1-Score')    