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

#import bifrost as bf

#from bisblocks import *
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
    igram_ids = readers.get_igram_ids(SAT, intfs_dir, ids)
    logger.info(f'Read {len(igram_ids)} interferogram ids from {intfs}')

    ##### Set up tools we'll use later #####

    # Figure out the order we'll want to read in the intfs
    files = []
    for igram in igram_ids:
        jd0, jd1 = jdates[igram]
        files.append( f'{path}/intf/{jd0}_{jd1}/{unwrapfile}' )
    logger.info(f'Ordered {len(files)} files to be read')

    # Build G matrix
    G = helpers.make_gmatrix(igram_ids, dates)
    logger.info(f'Created G-matrix with shape {G.shape}')

    # Extract reference point intf stack
    temp_reader = readers.DataStack.read(files)
    print(reflon, reflat, refnum)
    best_chunk_size = temp_reader.find_best_chunk_size(reflon, reflat, refnum)
    print(best_chunk_size)
    ref_stack = temp_reader.get_data_near(reflon, reflat, best_chunk_size)
    logger.info(f'Extracting reference stack, hopefully {ref_stack.shape}')

    # Generate regions for model creation

    ##### Timeseries pipeline #####

    # PIPELINE1: Model GPS points for detrending
    #with something as PIPELINE1:
    #    pass

    # Reference intfs to reference point

    # Optionally check data

    # Invert timeseries

    # Convert rad->mm


    # PIPELINE2: above + below 
    #with something as PIPELINE2:
    #    pass

    # Optionally detrend
    # if we don't want to detrend, we can pass a null model

    # Optionally calculate rate
    # we can maybe look at higher-order rates or peicewise stuff

    # Optionally plot

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