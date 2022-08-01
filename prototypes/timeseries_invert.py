#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 21 18:12:12 2022

@author: bruzewskis
"""

import numpy as np
import matplotlib.pyplot as plt
from time import time
import pickle

def make_g(ids, dates):
    # Fast generates G matrix
    
    idarr = np.arange(1, len(dates))

    # Pre-calculates the date differences
    diffdates = np.roll(dates, -1)[:-1] - dates[:-1]
    
    # Numpifies the creation of of the boolean arrays
    arr_less = igram_ids[:,0][:,None] < idarr
    arr_greq = igram_ids[:,1][:,None] >= idarr
    bool_arr = np.logical_and(arr_less, arr_greq)
    
    G = bool_arr * diffdates
    G = np.array(G)
    
    if not np.linalg.matrix_rank(G) == len(dates)-1:
        raise ValueError('Sorry, I messed up G')
    return G

def isbas_invert(G, data, mingood=0):
    
    ni,nx,ny = data.shape
    nt = G.shape[1]
    model = np.zeros((nt, nx, ny))*np.nan
    numgood = 0
    
    Gstore = {}
    for i in range(nx):
        for j in range(nx):
            Igood = np.where(~np.isnan(data[:,i,j]))[0]
            
            if len(Igood)>0:
                Ggood = G[Igood,:]
                
                # print(i,j, Ggood)
                cond1 = mingood > 0 and len(Igood) > mingood
                cond2 = mingood <= 0 and np.linalg.matrix_rank(Ggood) == nt 
                if cond1 or cond2:
                    Ikey=hash(Igood.tobytes())
                    if Ikey in Gstore:
                        Ginv=Gstore[Ikey]
                    else:
                        Ginv=np.linalg.pinv(Ggood)
                        Gstore[Ikey]=Ginv
                    model[:,i,j] = np.dot(Ginv, data[Igood,i,j])
                    numgood += 1
    return model

def censored_least_squares(G, data):
    # http://alexhwilliams.info/itsneuronalblog/2018/02/26/censored-lstsq/
    
    M = ~np.isnan(data)
    
    rhs = np.dot(G.T, M * data).T[:,:,None]
    T = np.matmul(G.T[None,:,:], M.T[:,:,None] * G[None,:,:])
    return np.squeeze(np.linalg.solve(T, rhs)).T

def reconstruct_ts(model, dates):
    nt = len(dates)
    ntmodel,nx,ny=np.shape(model)   
    ts=np.zeros((nt,nx,ny))
    for k in range(1,nt):
        ts[k]=ts[k-1]+model[k-1]*(dates[k]-dates[k-1])
    return ts


if __name__=='__main__':
    
    dates = pickle.load(open('pickled/dates.p', 'rb'))
    igram_ids = np.array(pickle.load(open('pickled/igramids.p', 'rb')))
    data = pickle.load(open('pickled/datamatrix.p', 'rb'))
    # data[2,10,14] = np.nan
    
    G = make_g(igram_ids, dates)
    
    # Old method
    model = isbas_invert(G, data)
    ts = reconstruct_ts(model, dates)
    
    # New method
    datar = data.reshape((6, 2500))
    rmodel = censored_least_squares(G, data.reshape((6,2500)))
    new_ts = reconstruct_ts(model.reshape((4-1,50,50)), dates)
    
    pickle.dump(new_ts, open('pickled/timeseries.p', 'wb'))

    # Little assertion test
    assert np.allclose(new_ts, ts), 'New timeseries does not match old timeseries'
    
        