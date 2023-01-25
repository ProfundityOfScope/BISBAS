#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Temp text up here

Created on Thu Sep 8 13:22:33 2022
@author: bruzewskis
"""

import numpy as np
import os
import sys
import time
import logging
import h5py

__version__ = 0.1

helperslogger = logging.getLogger('__main__')

def make_gmatrix(ids, dates):
    # Fast generates G matrix
    
    idarr = np.arange(1, len(dates))

    # Pre-calculates the date differences
    diffdates = np.roll(dates, -1)[:-1] - dates[:-1]
    
    # Numpifies the creation of of the boolean arrays
    arr_less = ids[:,0][:,None] < idarr
    arr_greq = ids[:,1][:,None] >= idarr
    bool_arr = np.logical_and(arr_less, arr_greq)
    
    G = bool_arr * diffdates
    G = np.array(G)
    
    if not np.linalg.matrix_rank(G) == len(dates)-1:
        readerslogger.error('G is of incorrect order')
        raise ValueError('G is of incorrect order')
    return G

def get_data_near_h5(file, x0, y0, min_points=10, max_size=20):
    
    # We don't need to check smaller chunks
    min_size = np.ceil(np.sqrt(min_points)).astype(int)

    # Grab variables
    x = file['x']
    y = file['y']
    z = file['displacements']
    #X, Y = np.meshgrid(x, y)

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
        xmin = np.ceil( xp - chunk_size/2 ).astype(int)
        ymin = np.ceil( yp - chunk_size/2 ).astype(int)
        xmax = xmin + chunk_size
        ymax = ymin + chunk_size
        
        # Grab that bit of the images
        zarr = z[ymin:ymax, xmin:xmax,:]

        # Check if what we grabbed is nice enough
        good_count = np.sum(~np.isnan(zarr), axis=(0,1))
        if np.all(good_count>min_points):
            # Skip lugging around the meshgrid
            ym, xm = np.mgrid[ymin:ymax, xmin:xmax]
            xarr = np.broadcast_to(x[xm, None], zarr.shape)
            yarr = np.broadcast_to(y[ym, None], zarr.shape)
            break
        
    return xarr, yarr, zarr

def generate_model(filename, gps, GTG, GTd, constrained=True, trendparams=3):

    # Warnings
    if len(gps)<trendparams:
        helperslogger.warning('Less GPS points than requested trendparams')
    
    # Grab the bits
    xg = gps[:,0]
    yg = gps[:,1]
    ng = gps[:,2]
    zg = gps[:,3:]

    # Open file and do stuff with it
    with h5py.File(filename, 'r') as fo:
        # Grab data around that point
        ndates = fo['t'].size
        ngps = len(gps)
        Gg = np.zeros((ngps, 6, ndates))
        dg = np.zeros((ngps, ndates))
        for i in range(ngps):
            # Find a good chunk of data
            xa, ya, za = get_data_near_h5(fo, xg[i], yg[i], ng[i])
            isgood = ~np.isnan(za)
            numgood = np.sum(isgood, axis=(0, 1))

            # Record it's bulk properties
            Gg[i] = np.array([numgood,
                              np.sum(xa,    axis=(0, 1), where=isgood),
                              np.sum(ya,    axis=(0, 1), where=isgood),
                              np.sum(xa**2, axis=(0, 1), where=isgood),
                              np.sum(ya**2, axis=(0, 1), where=isgood),
                              np.sum(xa*ya, axis=(0, 1), where=isgood)])
            dg[i] = (np.nanmean(za, axis=(0, 1)) - zg[i]) * numgood

    if constrained:
        # Build K-matrix
        K = np.zeros((6,6))

        # Solve for model params
        if np.log10(np.linalg.cond(K)):
            pass
    else:
        # Solve for model params (only gps)
        Gt = Gg[:,:trendparams,:]
        m, res, rank, sng = np.linalg.lstsq(Gt, dg, None)
    
    return m

if __name__=='__main__':
    # Grab or generate some testing cases
    gpsref = np.array([[255.285, 36.675, 10, 0]])
    gpstest1 = np.column_stack([np.random.uniform(254.8, 255.8, 5),
                                np.random.uniform(36.2, 37.2, 5),
                                np.full(5, 10),
                                np.random.normal(0, 10, (5,1))])
    gpstest2 = np.column_stack([np.random.uniform(254.8, 255.8, 5),
                                np.random.uniform(36.2, 37.2, 5),
                                np.full(5, 10),
                                np.random.normal(0, 10, (5,20))])

    GTG = np.fromfile('testing_gtg.dat')
    GTd = np.fromfile('testing_gtd.dat')

    m1n = generate_model('timeseries.h5', gpsref, GTG, GTd, False, 4)
    m1c = generate_model('timeseries.h5', gpsref, GTG, GTd, True, 4)
    print('No GPS:', m1n, m1c)

    m2n = generate_model('timeseries.h5', gpsref, GTG, GTd, False, 4)
    m2c = generate_model('timeseries.h5', gpsref, GTG, GTd, True, 4)
    print('GPS once:', m2n, m2c)

    m3n = generate_model('timeseries.h5', gpsref, GTG, GTd, False, 4)
    m3c = generate_model('timeseries.h5', gpsref, GTG, GTd, True, 4)
    print('GPS multiple:', m3n, m3c)


