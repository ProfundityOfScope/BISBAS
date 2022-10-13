#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This file contains various blocks for the Bifrost-ISBAS pipeline

Created on Sun Aug 14 23:38:03 2022
@author: bruzewskis
"""

import numpy as np
import logging

__version__ = 0.1

blockslogger = logging.getLogger('__main__')

def ReferenceBlock(data, ref_arr):
    reffed_data = data.copy()
    reffed_data[:,:,2] -= ref_arr[:,None]
    return reffed_data

def CheckBlock(data):
    return data

def GenTimeseriesBlock(data, dates, G, pool):

    # Perform linear algebra solve
    # http://alexhwilliams.info/itsneuronalblog/2018/02/26/censored-lstsq/

    # Generate matrices
    dataz = data[:,:,2]
    M = ~np.isnan(dataz)
    Tbig = np.matmul(G.T[None, :, :], M.T[:, :, None] * G[None, :, :])
    rbig = np.nansum(G.T[:, :, None] * (M*dataz)[None, :, :], axis=1).T

    # Mask out low-rank values
    lowrank = np.linalg.matrix_rank(Tbig) != len(dates) - 1
    Tbig[lowrank] = np.eye(len(dates)-1)
    rbig[lowrank] = np.full(len(dates)-1, np.nan)

    # Solve
    model = np.linalg.solve(Tbig, rbig).T

    # Build timeseries
    nt = len(dates)
    ntmodel,nz=np.shape(model)   
    ts=np.zeros((nt,nz))
    for k in range(1,nt):
        ts[k]=ts[k-1]+model[k-1]*(dates[k]-dates[k-1])

    return ts

def ConvertUnitsBlock(data, factor):
    return data*factor

    