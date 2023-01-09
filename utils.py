# -*- coding: utf-8 -*-
import os
import glob
import scipy.signal as scisig
import numpy as np
from sklearn.metrics import auc

def get_args(parser):
    parser.add_argument('--dataset', required=True, help='hascpac2016 | hasc2011 | beedance | salinas | ecg |')
    parser.add_argument('--window', type=int, default=64, help='input window size')
    parser.add_argument('--epsilon', type=float, default=0.1, help='input regularizer')
    parser.add_argument('--xi', type=int, default=10, help='input detection margin')
    parser.add_argument('--DELTA', type=int, default=10, help='input minimum horizontal distance')
    return parser

def get_data(args):
    if args.dataset== 'hascpac2016':
        dataroot = os.getcwd()+'/dataset/HASC-PAC2016' 
        dataset = glob.glob(dataroot + '/*.mat')

    elif args.dataset== 'hasc2011':
        dataroot = os.getcwd()+'/dataset/HASC2011' 
        dataset = glob.glob(dataroot + '/*.mat')
        
    elif args.dataset== 'beedance':
        dataroot = os.getcwd()+'/dataset/beedance' 
        dataset = glob.glob(dataroot + '/*.mat')
        
    elif args.dataset== 'ecg':
        dataroot = os.getcwd()+'/dataset/UCR' 
        dataset = glob.glob(dataroot + '/*.mat')
    
    elif args.dataset == 'salinas':
        dataroot = os.getcwd()+'/dataset/Salinas'
        dataset = glob.glob(dataroot+'/*.mat')

    else:
        raise ValueError("Unknown dataset %s" % (args.dataset))
        
    return dataset


"""PERFORMANCE METRICS"""

def f1_score(gtLabel, cpd_stat, thr = 0.5, tol_dist = 100,  detect_margin = 250):
    peakIndex = scisig.find_peaks(cpd_stat,  height = thr, distance = tol_dist)[0]
    predicted_cp = np.zeros(len(cpd_stat))
    predicted_cp[peakIndex] = 1
    no_cp_predicted = np.sum(predicted_cp)
    tp = 0
    for i in np.argwhere(predicted_cp == 1):
        if np.max(gtLabel[np.maximum(1, i[0]-detect_margin): np.minimum(len(gtLabel), i[0]+detect_margin)]) == 1:
            tp+=1
    no_cp_true = np.sum(gtLabel) 
    tp_2 = 0 # another definition of TP
    for i in np.argwhere(gtLabel == 1):
        if np.max(predicted_cp[np.maximum(0, i[0]-detect_margin):np.minimum(len(gtLabel)-1, i[0]+detect_margin)]==1):
            tp_2+=1
    if (no_cp_predicted==0):
        precision=0
    if (no_cp_true==0):
        recall=0
    
    precision = tp/ no_cp_predicted
    recall = tp_2 / no_cp_true 

    if (precision + recall == 0):
        f1=0
    else:
        f1 = 2*precision*recall/(precision+recall)

    return f1, precision, recall

def compute_auc_f1(gtLabel, z, DELTA = 2, xi = 2):
    f1_temp = 0
    th_temp = 0
    precision_list, recall_list, f1_list =[],[], []
    thr_List = np.linspace(z.min(), z.max(), 1000)
    for th in thr_List:
        f1, precision, recall = f1_score(gtLabel, z, thr = th, tol_dist = DELTA, detect_margin = xi)
        precision_list.append(precision)
        recall_list.append(recall)
        f1_list.append(f1)
        if f1>= f1_temp:
            th_temp = th
            f1_temp = f1
    auc_pr = auc(recall_list, precision_list)
    f1, precision, recall = f1_score(gtLabel, z, thr = th_temp, tol_dist = DELTA, detect_margin = xi)
    return  auc_pr, f1

    

