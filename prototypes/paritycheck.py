#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  5 10:48:37 2022

@author: bruzewskis
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 12 12:56:00 2022

@author: bruzewskis
"""

from scipy.io import netcdf_file
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.patches import Rectangle
import pickle
from tqdm import trange,tqdm
from time import time
import h5py

def imviz(x,y,z,norm=None,title='',fig=None,ax=None):

    fig = plt.figure(figsize=(6,5), dpi=1080/5) if fig is None else fig
    ax = fig.add_subplot() if ax is None else ax
        
    if norm is None:
        norm = Normalize(np.nanmean(z)-np.nanstd(z),
                         np.nanmean(z)+np.nanstd(z))

    ax.pcolormesh(x, y, z, shading='auto', norm=norm, cmap='Spectral')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title)

def get_data_near_h5(x,y,z, x0, y0, min_points=10, max_size=20):

    # We don't need to check smaller chunks
    min_size = np.ceil(np.sqrt(min_points)).astype(int)

    # This is how we would deal with a non-uniform spacing
    xp = np.interp(x0, x, np.arange(len(x)))
    yp = np.interp(y0, y, np.arange(len(y)))

    # We need to find a good chunk size
    for chunk_size in np.arange(min_size, max_size):

        # Check if the position is outside of image
        if any([xp <= chunk_size, xp >= len(x)-chunk_size,
                yp <= chunk_size, yp >= len(y-chunk_size)]):
            raise ValueError('This position too close to edge of image')

        # Find corners
        xmin = np.ceil(xp - chunk_size/2).astype(int)
        ymin = np.ceil(yp - chunk_size/2).astype(int)
        xmax = xmin + chunk_size
        ymax = ymin + chunk_size

        # Grab that bit of the images
        zarr = z[ymin:ymax, xmin:xmax]

        # Check if what we grabbed is nice enough
        good_count = np.sum(~np.isnan(zarr))
        if good_count >= min_points:
            # Skip lugging around the meshgrid
            ym, xm = np.mgrid[ymin:ymax, xmin:xmax]
            xarr = x[xm]
            yarr = y[ym]
            break
    else:
        raise ValueError('Couldn\'t find a good chunk, try a different reference')

    return xarr, yarr, zarr

def make_model(x, y, z, nt=3, gps=None):

    # Good pixels
    yind, xind = np.where(~np.isnan(z))
    d = z.ravel()
    d = d[~np.isnan(d)]
    xg = np.array(x)[xind]
    yg = np.array(y)[yind]

    # Generate stuff and solve
    ones = np.ones_like(xg)
    A = np.column_stack([ones, xg, yg, xg**2, yg**2, xg*yg])
    
    # Figure out original fit
    m0, res, rank, sing = np.linalg.lstsq(A[:, :nt], d, rcond=None)

    # Compose GPS stuff
    ng = len(gps)
    xgps = gps[:, 0]
    ygps = gps[:, 1]
    pgps = gps[:, 2]
    zgps = gps[:, 3:]

    Gg = np.zeros((ng, 6))
    dg = np.zeros((ng, 1))
    for i in range(ng):
        # Find a good chunk of data
        xa, ya, za = get_data_near_h5(x, y, z, xgps[i], ygps[i], pgps[i])
        isgood = ~np.isnan(za)
        numgood = np.sum(isgood)

        # Record it's bulk properties
        Gg[i] = np.column_stack([numgood,
                                 np.sum(xa,    where=isgood),
                                 np.sum(ya,    where=isgood),
                                 np.sum(xa**2, where=isgood),
                                 np.sum(ya**2, where=isgood),
                                 np.sum(xa*ya, where=isgood)])
        dg[i] = (np.nanmean(za) - zgps[i]) * numgood
    
    # Build up
    GTG = np.dot(A.T, A)
    GTd = np.dot(A.T, d)
    
    # Assemble K matrix
    K = np.zeros((nt+ng, nt+ng))
    K[:nt, :nt] = 2 * GTG[:nt, :nt]
    K[:nt, nt:] = Gg[:,:nt].T
    K[nt:, :nt] = Gg[:,:nt]

    # Assemble D matrix
    D = np.zeros((ng+nt))
    D[:nt] = 2 * GTd[:nt]
    D[nt:] = np.squeeze(dg)
    
    # Solve
    m1, res, rank, sing = np.linalg.lstsq(K, D, rcond=None)
    m1 = m1[:nt]
    
    test = np.dot(np.linalg.pinv(K), D)

    # Plot model
    xl = np.linspace(np.min(x), np.max(x), 10)
    yl = np.linspace(np.min(y), np.max(y), 10)
    XL, YL = np.meshgrid(xl, yl)
    ZL = np.array([np.ones_like(XL), XL, YL, XL**2, YL**2, XL*YL])
    ZL0 = np.sum(m0[:, None, None]*ZL[:nt], axis=0)
    ZL1 = np.sum(m1[:, None, None]*ZL[:nt], axis=0)

    top = max([np.max(ZL0), np.max(ZL1)])
    bot = min([np.min(ZL0), np.min(ZL1)])
    fig = plt.figure()
    ax1 = plt.subplot2grid((1,2),(0,0), fig=fig, projection='3d')
    ax1.plot_surface(XL, YL, ZL0)
    ax1.set_zlim(bot, top)
    ax2 = plt.subplot2grid((1,2),(0,1), fig=fig, projection='3d')
    ax2.plot_surface(XL, YL, ZL1)
    ax2.set_zlim(bot, top)
    
    print(m0)
    print(m1)
    print(test)

    # Return model
    X, Y = np.meshgrid(x, y)
    ZM = np.array([np.ones_like(X), X, Y, X**2, Y**2, X*Y])
    ZM = np.sum(m1[:, None, None]*ZM[:nt], axis=0)
    return ZM

with h5py.File('/Users/bruzewskis/Downloads/timeseries.h5', 'r') as fo:

    i = 1
    date = fo['t'][i]

    gt = '/Users/bruzewskis/Downloads/isbas_ground_truth'
    real = f'{gt}/timeseries_nodetrend/ts_mm_{date:04d}.grd'
    realc = f'{gt}/timeseries_detrended/ts_mm_{date:04d}.grd'
    with netcdf_file(real) as fr:
        im_true = fr.variables['z'][:]
    with netcdf_file(realc) as fr:
        im_true_corr = fr.variables['z'][:]
        
    gps = np.array([[255.285,36.675,10,0],[255.286,36.677,10,0]])

    zm = make_model(fo['x'][:], fo['y'][:], im_true, gps=gps)
    corr = fo['displacements'][:,:,i] - zm
    
    norm = Normalize(np.nanmean(im_true_corr)-np.nanstd(im_true_corr),
                     np.nanmean(im_true_corr)+np.nanstd(im_true_corr))

    fig = plt.figure(figsize=(10,10), dpi=72)
    ax = plt.subplot2grid((2,2),(0,0), fig=fig)
    imviz(fo['x'], fo['y'], im_true, norm, 'ISBAS', fig=fig, ax=ax)
    
    ax = plt.subplot2grid((2,2),(1,0), fig=fig)
    imviz(fo['x'], fo['y'], im_true_corr, norm, 'ISBAS Corrected', fig=fig, ax=ax)
    
    ax = plt.subplot2grid((2,2),(0,1), fig=fig)
    imviz(fo['x'], fo['y'], fo['displacements'][:,:,i], norm, 'Bifrost', fig=fig, ax=ax)
    
    ax = plt.subplot2grid((2,2),(1,1), fig=fig)
    imviz(fo['x'], fo['y'], fo['detrended'][:,:,i], norm, 'Bifrost Corrected', fig=fig, ax=ax)
