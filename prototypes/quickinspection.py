#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  6 10:51:43 2023

@author: bruzewskis
"""

import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import transforms
from matplotlib.colors import LogNorm
from astropy.time import Time
from scipy.optimize import curve_fit
from matplotlib.patches import Rectangle
from tqdm import trange
from time import sleep, time

import scipy
        
def intf2ts(px_intf, G, med, wave=0.05546576):

    conv = -1000*wave/(4*np.pi)
    px_ref = px_intf - med
    px_ref *= conv

    M = ~np.isnan(px_ref)
    A = np.linalg.multi_dot([G.T, np.diag(M), G]).astype('float32')
    B = np.nansum(np.dot(G.T, np.diag(M)) * px_ref, axis=1)
    
    sign, abslogdet = np.linalg.slogdet(A)
    cond = np.linalg.cond(A)
    rank = np.linalg.matrix_rank(A, hermitian=True)
    
    return abslogdet, cond, rank

if __name__=='__main__':
    
    image = np.load('../image_mostbig.npy')
    yb, xb = np.where(np.abs(image)>1e10)
    
    G = np.load('../test_gmat.npy').astype('int32')
    med = np.load('../test_med.npy')
    
    fig = plt.figure(figsize=(15,15), dpi=1920/15)
    
    with h5py.File('../ifgSlim.h5', 'r') as fo:
        
        goodmap = np.sum(fo['coherence'][:]>0.4, axis=0)/len(fo['coherence'])
        
        data = np.where(fo['coherence'][:]>0.4, fo['unwrapPhase'], np.nan)
        ax0 = plt.subplot2grid((2,2),(0,0))
        ax0.imshow(np.sum(~np.isnan(data), axis=0))
        ax0.invert_xaxis()
        
        start = 1255
        end = start + 200
        dr = data.reshape(513, -1).T[start:end]
        M = (~np.isnan(dr))
        A = np.matmul(G.T[None, :, :], M[:, :, None] * G[None, :, :])
        B = np.nansum(G.T[:, :, None] * (M*dr).T[None, :, :], axis=1).T
        
        for i in range(start, end):
            plt.scatter(i%50, i//50, ec='k', fc='w', s=30, marker='s')
        print(A.dtype, M.dtype)
        
        sign, logdet = np.linalg.slogdet(A)
        low = np.isinf(logdet)
        A[low] = np.eye(172)
        B[low] = np.full(172, np.nan)
        
        M = np.linalg.solve(A, B)
        plt.subplot2grid((2,2),(0,1))
        plt.hist(M.ravel(), bins=20)
        plt.semilogy()
        
    
    ax3 = plt.subplot2grid((2,2),(1,1), fig=fig)
    ax3.imshow(image, interpolation='nearest', cmap='Spectral', vmin=-45, vmax=60)
    ax3.scatter(xb,yb, ec='k', fc='w', marker='s', s=30)
    
        
    for i in range(start, end):
        plt.scatter(2250+i%50, 2275+i//50, ec='k', fc='m', s=30, marker='s')
    ax3.invert_xaxis()
    plt.xlim(2300, 2250)
    plt.ylim(2325, 2275)
    plt.title('Timeseries')
    plt.show()
    
    
    arrs = np.load('../parityMatrices.npy')
    B = np.ones(172)
    
    start = time()
    sign, logdet = np.linalg.slogdet(arrs)
    is_singular = np.isinf(logdet)
    print('Det', is_singular, time()-start)
    
    S = np.linalg.svd(arrs, compute_uv=False, hermitian=True)
    print(np.any(np.isclose(S,0), axis=1))
    
    a = np.ones(10)*10
    test = np.diag(a) + np.diag(a,-1)[:10,:10]/2 + np.diag(a,1)[:10,:10]/2
    print(test)
    
    plt.show()
    # plt.imshow(test)
    print(np.linalg.det(test))