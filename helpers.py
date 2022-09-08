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

__version__ = 0.1

readerslogger = logging.getLogger('__main__')

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

def create_file_order(path, igrams, unwrap, jdates):

	return files