#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 10 11:53:50 2022

@author: bruzewskis
"""

import numpy as np
import os
import matplotlib.pyplot as plt
import tracemalloc
from scipy.io import netcdf
import h5py



A = np.random.random((1000,4))
B = np.random.random((1000,))
# B[np.random.choice(1000, 10)] = np.nan

AB_dot = A.T.dot(B)
AB_dot = np.nansum(np.einsum('ij,j->ij', A.T, B), axis=1)
AA_dot = A.T.dot(A)
AA_dot = np.nansum(np.einsum('ij,jk->ijk', A.T, A), axis=1)


AB_accum = np.zeros(np.size(A.T,0))
AA_accum = np.zeros((np.size(A.T,0),)*2)

regions = np.arange(B.size).reshape(-1,10)
for region in regions:
    AB_accum += np.nansum(np.einsum('ij,j->ij', A.T[:,region], B[region]), axis=1)
    
    AA_accum += np.nansum(np.einsum('ij,jk->ijk', A[region].T, A[region]), axis=1)
    
print('A.T dot B matches:', np.allclose(AB_accum, AB_dot))
print('A.T dot A matches:', np.allclose(AA_accum, AA_dot))

################
A = np.random.random((1000,4))
B = np.random.random((1000,20))

isgood = np.random.choice([True, False], (1000,20), p=[0.8, 0.2])
B[~isgood] = np.nan

ATA1 = np.zeros((4,4,20))
ATB1 = np.zeros((4,20))
for i in range(20):
    igl = isgood[:,i]
    Al = A[igl]
    Bl = B[igl,i]
    
    ATA1[:,:,i] = np.dot(Al.T, Al)
    ATB1[:,i] = np.dot(Al.T, Bl)

ATA2 = np.einsum('ij,jk,jl->ikl', A.T, A, isgood)
ATB2 = np.nansum(np.einsum('ij,jk->ijk', A.T, B), axis=1)
print(np.allclose(ATA1, ATA2))
print(np.allclose(ATB1, ATB2))