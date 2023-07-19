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

coi = [ (2291, 2270), (2291, 2271), (2291, 2273) ]
notes = ['normal', 'det0, tsbig', 'det0, tsnormal']

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
        
        print(f'Coordinate: ({ci, cj}) with notes: "{notes[i]}"')