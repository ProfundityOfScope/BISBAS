#!/usr/bin/env python3
"""
Helper functions that are not themselves full BISBAS blocks
"""

import os
import sys
import time
import logging
from datetime import datetime

import h5py
import numpy as np

__author__ = "Seth Bruzewski"
__credits__ = ["Seth Bruzewski", "Jayce Dowell", "Gregory Taylor"]

__license__ = "MIT"
__version__ = "1.0.2"
__maintainer__ = "Seth Bruzewski"
__email__ = "bruzewskis@unm.edu"
__status__ = "development"

helperslogger = logging.getLogger('__main__')

def make_gmatrix(datepairs):
    
    # Conver to numbers, find differences
    dates = np.sort(np.unique(datepairs))
    t0 = datetime.strptime(dates[0], '%Y%m%d')
    dr = np.array([(datetime.strptime(d, '%Y%m%d')-t0).days for d in dates])
    diffdates = np.roll(dr, -1)[:-1] - dr[:-1]

    # Convert datestrs to indices
    indmap = {dates[i]: i for i in range(len(dates))}
    indpairs = np.vectorize(lambda x: indmap[x])(datepairs)
    idarr = np.arange(1, len(dates))

    # Numpifies the creation of of the boolean arrays
    arr_less = indpairs[:, 0][:, None] < idarr
    arr_greq = indpairs[:, 1][:, None] >= idarr
    bool_arr = np.logical_and(arr_less, arr_greq)

    G = bool_arr * diffdates
    if not np.linalg.matrix_rank(G) == len(dates)-1:
        raise ValueError('G is of incorrect order')
    
    return G

def data_near(data, x0, y0, min_points=10, max_size=20):
    
    # We don't need to check smaller chunks
    min_size = np.ceil(np.sqrt(min_points)).astype(int)

    # For broadcasting
    x = np.arange(np.size(data, 2))
    y = np.arange(np.size(data, 1))

    # We need to find a good chunk size
    for chunk_size in np.arange(min_size, max_size):

        # Check if the position is outside of image
        if any([x0 <= chunk_size, x0 >= len(x)-chunk_size,
                y0 <= chunk_size, y0 >= len(y-chunk_size)]):
            raise ValueError('This position too close to edge of image')
    
        # Find corners
        xmin = np.ceil( x0 - chunk_size/2 ).astype(int)
        ymin = np.ceil( y0 - chunk_size/2 ).astype(int)
        xmax = xmin + chunk_size
        ymax = ymin + chunk_size

        # Grab that bit of the images
        zarr = data[:, ymin:ymax, xmin:xmax]

        # Check if what we grabbed is nice enough
        good_count = np.sum(~np.isnan(zarr), axis=(1,2))
        if np.all(good_count>=min_points):
            # Skip lugging around the meshgrid
            ym, xm = np.mgrid[ymin:ymax, xmin:xmax]
            break
    else:
        raise ValueError('Couldn\'t find a good chunk, try a different reference')
        
    return xm, ym, zarr

def generate_model(filename, gps, GTG, GTd, constrained=True, nt=3):

    # Warnings
    ng = len(gps)
    if not constrained and len(gps)<nt:
        helperslogger.warning('Less GPS points than requested trendparams')
    
    # Grab the bits
    xg = gps[:,0]
    yg = gps[:,1]
    pg = gps[:,2]
    zg = gps[:,3:]

    # Open file and do stuff with it
    with h5py.File(filename, 'r') as fo:
        # Grab data around that point
        nd = fo['t'].size
        Gg = np.zeros((ng, 6, nd))
        dg = np.zeros((ng, nd))
        for i in range(ng):
            # Find a good chunk of data
            xa, ya, za = get_data_near_h5(fo, xg[i], yg[i], pg[i])
            isgood = ~np.isnan(za)
            numgood = np.sum(isgood, axis=(0, 1))

            # Record it's bulk properties
            Gg[i] = np.row_stack([numgood,
                                     np.sum(xa,    axis=(0, 1), where=isgood),
                                     np.sum(ya,    axis=(0, 1), where=isgood),
                                     np.sum(xa**2, axis=(0, 1), where=isgood),
                                     np.sum(ya**2, axis=(0, 1), where=isgood),
                                     np.sum(xa*ya, axis=(0, 1), where=isgood)])
            dg[i] = (np.nanmean(za, axis=(0, 1)) - zg[i]) * numgood

    helperslogger.debug(f'GPS Matrix:  {Gg.shape} and {dg.shape}')
    helperslogger.debug(f'Data Matrix: {GTG.shape} and {GTd.shape}')
    m = np.zeros((nt, nd))
    if constrained:

        # Assemble K matrix
        K = np.zeros((nt+ng, nt+ng, nd))
        K[:nt, :nt] = 2 * GTG[:nt, :nt]
        K[:nt, nt:] = np.transpose(Gg[:,:nt], (1,0,2))
        K[nt:, :nt] = Gg[:,:nt]

        # Assemble D matrix
        D = np.zeros((ng+nt, nd))
        D[:nt] = 2 * GTd[:nt]
        D[nt:] = dg
    else:
        # If not constrain just use data
        K = Gg[:,:nt]
        D = dg

    # Solve for model params
    for i in range(nd):
        md, res, rank, sng = np.linalg.lstsq(K[:,:,i], D[:,i], None)
        m[:,i] = md[:nt]
    
    return m

if __name__=='__main__':
    # Grab or generate some testing cases
    gpstest1 = np.array([[255.285, 36.675, 10, 0]])
    gpstest2 = np.column_stack([np.random.normal(255.3, 0.1, 5),
                                np.random.normal(36.7, 0.1, 5),
                                np.full(5, 10),
                                np.random.normal(0, 10, (5,1))])
    gpstest3 = np.column_stack([np.random.normal(255.3, 0.1, 5),
                                np.random.normal(36.7, 0.1, 5),
                                np.full(5, 10),
                                np.random.normal(0, 10, (5,20))])

    GTG = np.fromfile('testing_gtg.dat').reshape((6,6,20))
    GTd = np.fromfile('testing_gtd.dat').reshape((6,20))

    print('='*10, 'Test 1', '='*10)
    m1a = generate_model('timeseries_backup.h5', gpstest1, GTG, GTd, True, 4)
    print('No GPS, constrained:\n', m1a)

    print('='*10, 'Test 2', '='*10)
    m2a = generate_model('timeseries_backup.h5', gpstest2, GTG, GTd, True, 4)
    m2b = generate_model('timeseries_backup.h5', gpstest2, GTG, GTd, False, 4)
    print('GPS once, constrained:\n', m2a)
    print('GPS once, not constrained:\n', m2b)
    
    print('='*10, 'Test 3', '='*10)
    m3a = generate_model('timeseries_backup.h5', gpstest3, GTG, GTd, True, 4)
    m3b = generate_model('timeseries_backup.h5', gpstest3, GTG, GTd, False, 4)
    print('GPS multiple, constrained:\n', m3a)
    print('GPS multiple, not constrained:\n', m3b)
    


