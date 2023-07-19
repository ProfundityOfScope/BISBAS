#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 19 14:51:05 2023

@author: bruzewskis
"""

import cupy as cp
import cupyx as cpx
import numpy as np
import h5py

coi = [ (2291, 2270), (2291, 2271), (2291, 2273), (2285, 2274)]
notes = ['normal', 'det0, tsbig', 'det0, tsnormal', 'det0, tsflagged']

G = np.load('test_gmat.npy')
med = np.load('test_med.npy')
conv = -4.4

with h5py.File('ifgramStack.h5', 'r') as fo:
    
    for i in range(len(coi)):
        ci,cj = coi[i]
        
        phases = fo['unwrapPhase'][:,ci,cj]
        cohers = fo['coherence'][:,ci,cj]
        
        # Set up
        phases[cohers<0.4] = np.nan
        phases -= med
        phases *= conv
        
        print(f'Coordinate: ({ci}, {cj}) with notes: "{notes[i]}"')
        
        # pure numpy
        M = np.diag(~np.isnan(phases))
        A = np.linalg.multi_dot([G.T, M, G]).astype('float32')
        det = np.linalg.det(A)
        sign, logdet = np.linalg.slogdet(A)
        rank = np.linalg.matrix_rank(A, hermitian=True)
        print('\tPure numpy(32):', det, logdet, rank)
        
        # pure numpy
        M = np.diag(~np.isnan(phases))
        A = np.linalg.multi_dot([G.T, M, G])
        det = np.linalg.det(A)
        sign, logdet = np.linalg.slogdet(A)
        rank = np.linalg.matrix_rank(A, hermitian=True)
        print('\tPure numpy(64?):', det, logdet, rank)
        
        # pure cupy
        Mc = cp.diag(~cp.isnan(cp.asarray(phases)))
        Gc = cp.asarray(G)
        Ac = cp.dot(Gc.T, Mc).dot(Gc).astype('float32')
        det = cp.linalg.det(Ac)
        sign, logdet = cp.linalg.slogdet(Ac)
        rank = cp.linalg.matrix_rank(Ac)
        print('\tPure Cupy(32):', det, logdet, rank)
    
        # pure cupy 64
        Mc = cp.diag(~cp.isnan(cp.asarray(phases)))
        Gc = cp.asarray(G)
        Ac = cp.dot(Gc.T, Mc).dot(Gc)
        det = cp.linalg.det(Ac)
        sign, logdet = cp.linalg.slogdet(Ac)
        rank = cp.linalg.matrix_rank(Ac)
        print('\tPure Cupy(64?):', det, logdet, rank)
        
        # cupy with errstate
        with cpx.errstate(linalg='raise'):
            Mc = cp.diag(~cp.isnan(cp.asarray(phases)))
            Gc = cp.asarray(G)
            Ac = cp.dot(Gc.T, Mc).dot(Gc).astype('float32')
            det = cp.linalg.det(Ac)
            sign, logdet = cp.linalg.slogdet(Ac)
            rank = cp.linalg.matrix_rank(Ac)
            print('\tErrstate Cupy(32):', det, logdet, rak)
        
        # cupy with errstate 64
        with cpx.errstate(linalg='raise'):
            Mc = cp.diag(~cp.isnan(cp.asarray(phases)))
            Gc = cp.asarray(G)
            Ac = cp.dot(Gc.T, Mc).dot(Gc)
            det = cp.linalg.det(Ac)
            sign, logdet = cp.linalg.slogdet(Ac)
            rank = cp.linalg.matrix_rank(Ac)
            print('\tErrstate Cupy(64):', det, logdet, rank)
            