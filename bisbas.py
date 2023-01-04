#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Bifrost implementation of ISBAS algorithms

Created on Tue Jul 26 14:18:33 2022
@author: bruzewskis
"""

#system-wide imports
import os
import sys
import time
import numpy as np
import logging
import argparse
import configparser
import multiprocessing as mp
import pickle

import bifrost as bf

import bisblocks
import fakeblocks
import readers
import helpers

__version__ = 0.1


def main(args):
    
    ##### Setup a logger #####
    logger = logging.getLogger(__name__)
    logFormat = logging.Formatter('%(asctime)s [%(levelname)-8s] %(message)s', 
                                  datefmt='%Y-%m-%d %H:%M:%S')
    logFormat.converter = time.gmtime
    
    # Decide to write to file or stdout
    if args.log is None:
        logHandler = logging.StreamHandler(sys.stdout)
    else:
        logHandler = logging.FileHandler(args.log)
    logHandler.setFormatter(logFormat)
    logger.addHandler(logHandler)
    
    # Decide what level to report
    if args.debug:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)
        
    logger.info(f'Starting bisbas.py with PID {os.getpid()}')
    logger.info(f'Version: {__version__}')
        
    ##### Read in the config file #####
    logger.info(f'Attempting to use {args.config} as config file')
    if not os.path.exists(args.config):
        logger.error('Could not find provided')
    config = configparser.ConfigParser()
    config.optionxform = str #make the config file case-sensitive
    config.read(args.config)
    
    # timeseries options
    ts_type   = config.get('timeseries-config','ts_type')
    mingood   = config.getint('timeseries-config','nsbas_min_intfs')
    
    # ref. pixel options
    reflat    = config.getfloat('timeseries-config','reflat')
    reflon    = config.getfloat('timeseries-config','reflon')
    refnum    = config.getint('timeseries-config','refnum')
   
    # file input options
    SAT       = config.get('timeseries-config','sat_name')
    unwrapfile= config.get('timeseries-config','unwrap_file')
    grdnaming = config.get('timeseries-config','grdnaming') # this is really unnecessary but we keep it for now
    prmfile   = config.get('timeseries-config','prm_file')
    intfs     = config.get('timeseries-config','intf_file')
    baselines = config.get('timeseries-config','baseline_file')
    
    #unwrapping check options
    unw_check_threshold = config.getfloat('timeseries-config','unw_check_threshold')
    
    # rate (velocity) options
    calcrate  = config.getboolean('timeseries-config','calcrate')
   
    # de-trending options
    detrend   = config.getboolean('timeseries-config','detrend')
    trendparams = config.getint('timeseries-config','trendparams')
    gpsfile   = config.get('timeseries-config','gps_file')
    constrainedtrend = config.getboolean('timeseries-config','constrainedtrend')
    
    # plotting options
    makeplots = config.getboolean('timeseries-config','makeplots')
    
    ##### Read and set up data #####
    
    # Read in the baseline table
    path = os.getcwd()
    bl_dir = os.path.join(path, baselines)
    ids, jdates, dates, bperp = readers.read_baselines(bl_dir)
    logger.info(f'Read {len(ids)} baselines from {baselines}')
    
    # Read dates of all interferogram pairs and convert to integer indexes
    intfs_dir = os.path.join(path, intfs)
    igram_ids = readers.read_igram_ids(SAT, intfs_dir, ids)
    logger.info(f'Read {len(igram_ids)} interferogram ids from {intfs}')

    prm_dir = os.path.join(path, prmfile)
    rad2mm_conv = readers.read_wavelength_conversion(prm_dir)
    logger.info(f'Read unit conversion from {prmfile}')

    ##### Set up tools we'll use later #####

    # Figure out the order we'll want to read in the intfs
    files = []
    for igram in igram_ids:
        jd0, jd1 = jdates[igram]
        files.append( f'intf/{jd0}_{jd1}/{unwrapfile}' )
    logger.info(f'Ordered {len(files)} files to be read')

    # Build G matrix
    G = helpers.make_gmatrix(igram_ids, dates)
    logger.info(f'Created G-matrix with shape {G.shape}')

    # Extract reference point intf stack
    read_stack = readers.DataStack.read(files)
    best_chunk_size = read_stack.find_best_chunk_size(reflon, reflat, refnum)
    ref_stack = read_stack.data_near(reflon, reflat, best_chunk_size)
    median_stack = np.nanmedian(ref_stack, axis=0)
    logger.info(f'Extracted {median_stack.size} median values to reference to')

    # Generates the timeseries
    with bf.get_default_pipeline() as PIPELINE1:
        # Do stuff blocks
        b_read = bisblocks.IntfReadBlock([path], 13_000, 'f32', files)
        b_reffed = bisblocks.ReferenceBlock(b_read, median_stack)
        b_tseries = bisblocks.GenTimeseriesBlock(b_reffed, dates, G)

        # Sink blocks
        b_write = bisblocks.WriteAndAccumBlock(b_tseries, 'timeseries.h5')

        GTG = b_write.GTG
        GTd = b_write.GTd
        PIPELINE1.run()

    print('I accumulated this:', thing)

if __name__=='__main__':
    globalstart=time.time()
    parser = argparse.ArgumentParser(description='Run ISBAS/SBAS on a GMTSAR-formatted dataset')
    parser.add_argument('-c', '--config', type=str, default='./isbas.config', 
                        help='supply name of config file to setup processing options. Required.')
    parser.add_argument('-l', '--log', type=str, 
                        help='name of logfile to write information to')
    parser.add_argument('-d', '--debug', action='store_true',
                        help='print debug messages as well as info and higher')
    args = parser.parse_args()
    
    main(args)