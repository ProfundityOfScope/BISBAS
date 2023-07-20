#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 19 14:51:05 2023

@author: bruzewskis
"""

import cupy as cp
from cupyx.scipy.linalg import lu_factor
import numpy as np
import scipy as sp
import h5py
from time import time

coi = [ (2291, 2270), (2291, 2271), (2291, 2273), (2285, 2274)]
notes = ['normal', 'det0, tsbig', 'det0, tsnormal', 'det0, tsflagged']

G = np.load('test_gmat.npy')
med = np.load('test_med.npy')

with h5py.File('ifgramStack.h5', 'r') as fo:
    
    nd = np.size(G, 1)
    out = np.zeros((len(coi), nd, nd))
    for i in range(len(coi)):
        ci, cj = coi[i]
        
        phases = fo['unwrapPhase'][:,ci,cj]
        cohers = fo['coherence'][:,ci,cj]
        
        # Set up
        phases[cohers<0.4] = np.nan
        phases -= med
        
        print(f'Coordinate: ({ci}, {cj}) with notes: "{notes[i]}"')
        mask = (cohers<0.4).astype(int)
        
        # pure numpy
        M = np.diag(~np.isnan(phases))
        A = np.linalg.multi_dot([G.T, M, G])
        out[i] = A
        
        start = time()
        dettest = np.isinf(np.linalg.slogdet(A)[1])
        dettime = time() - start
        start = time()
        lu,piv = sp.linalg.lu_factor(A)
        lutest = np.any(np.diag(np.triu(lu))<1e-10)
        lutime = time() - start
        
        print('\tPure numpy(64?):', dettest, dettime, lutest, lutime)
    
        # pure cupy 64
        Mc = cp.asarray(M)
        Gc = cp.asarray(G)
        Ac = cp.asarray(A)
        
        start = time()
        dettestc = cp.isinf(cp.linalg.slogdet(Ac)[1])
        dettimec = time() - start
        start = time()
        lu,piv = lu_factor(Ac)
        lutestc = cp.any(cp.diag(cp.triu(lu))<1e-10)
        lutimec = time() - start
        print('\tPure Cupy(64?):', dettestc, dettimec, lutestc, lutimec)
            
np.save('parityMatrices.npy', out)