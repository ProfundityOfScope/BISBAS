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

AB_dot = A.T.dot(B)
AA_dot = A.T.dot(A)


AB_accum = np.zeros(np.size(A.T,0))
AA_accum = np.zeros((np.size(A.T,0),)*2)

regions = np.arange(B.size).reshape(-1,10)
for region in regions:
    AB_accum += np.nansum(np.einsum('ij,j->ij', A.T[:,region], B[region]), axis=1)
    
    AA_accum += np.nansum(np.einsum('ij,jk->ijk', A.T[:,region], A[region]), axis=1)
    
print('A.T dot B matches:', np.allclose(AB_accum, AB_dot))
print('A.T dot A matches:', np.allclose(AA_accum, AA_dot))

tgt = 'testing.hdf5'
nt, ny, nx = 10, 50, 50
if os.path.exists(tgt):    
    os.remove(tgt)
with h5py.File(tgt, 'a') as f:
    
    fd = f.create_dataset('data', data=np.random.random((nt, nx, ny)))
    
    ft = f.create_dataset('time', data=np.sort(np.random.randint(0,500,nt)))
    ft.make_scale('days from first measurement')
    fd.dims[0].attach_scale(ft)
    fd.dims[0].label = 'Time'
    
    
    flon = f.create_dataset('lon', data=np.linspace(298.1, 298.2, nx))
    flon.make_scale('degrees east')
    fd.dims[1].attach_scale(flon)
    fd.dims[1].label = 'Longitude'
    
    flat = f.create_dataset('lat', data=np.linspace(-42.8, -42.6, ny))
    flat.make_scale('degrees north')
    fd.dims[2].attach_scale(flat)
    fd.dims[2].label = 'Latitude'
    
    print('Datasets:', list(f))
    print('Data dimension labels:', [dim.label for dim in fd.dims])
    for i,dim in enumerate(fd.dims):
        print(f'Data dim[{i}] scales:', dim.keys())
    timerecov = fd.dims[0].values()[0]
    # print(timerecov[])