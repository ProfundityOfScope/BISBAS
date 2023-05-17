#!/usr/bin/env python3
"""
Bifrost implementation of ISBAS algorithms
"""

import os
import sys
import time
import pickle
import logging
import argparse
import configparser
import multiprocessing as mp

import h5py
import numpy as np
import bifrost as bf

import bisblocks
import helpers

__author__ = "Seth Bruzewski"
__credits__ = ["Seth Bruzewski", "Jayce Dowell", "Gregory Taylor"]

__license__ = "MIT"
__version__ = "1.0.2"
__maintainer__ = "Seth Bruzewski"
__email__ = "bruzewskis@unm.edu"
__status__ = "development"


def old_main(args):

    # Timekeeping
    start_time = time.time()
    
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

    # Get the parameter file
    prm_dir = os.path.join(path, prmfile)
    rad2mm_conv = readers.read_wavelength_conversion(prm_dir)
    logger.info(f'Read unit conversion from {prmfile}')

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
        b_read = bisblocks.IntfReadBlock([path], args.gulp, 'f32', files, space='system')

        # This on GPU?
        b_read_gpu = bf.blocks.copy(b_read, space='cuda')
        b_reff_gpu = bisblocks.ReferenceBlock(b_read_gpu, median_stack)
        b_tser_gpu = bisblocks.GenTimeseriesBlock(b_reff_gpu, dates, G)
        b_tsmm_gpu = bisblocks.ConvertToMillimetersBlock(b_tser_gpu, rad2mm_conv)
        b_tsmm = bf.blocks.copy(b_tsmm_gpu, space='cuda_host')

        # Sink block
        b_write = bisblocks.WriteAndAccumBlock(b_tsmm, args.outfile)

        PIPELINE1.run()

        # Keep track of accumulated values
        GTG = b_write.GTG
        GTd = b_write.GTd

    ts_time = time.time() - start_time
    logger.info(f'Finished timeseries generation in {ts_time} s')

    # If user requested detrend, we do it
    if detrend:
        logger.info('Detrend requested.')

        # Figure out GPS
        if os.path.exists(gpsfile):
            logger.info('Loading GPS file.')
            gps = np.loadtxt(gpsfile)
        else:
            logger.info('No GPS, zeroing at reference point.')
            gps = np.array([[reflon, reflat, refnum, 0]])

        model = helpers.generate_model(args.outfile, gps, GTG, GTd, True, 3)
        logger.debug(f'Generated a model: {model.shape}')

        # These will be useful for model fitting
        logger.debug('Grabbing axes from generated file')
        with h5py.File(args.outfile, 'r') as fo:
            x_axis = fo['x'][:]
            y_axis = fo['y'][:]
            t_axis = fo['t'][:]

        # Second pipeline
        logger.debug('Starting second pipeline')
        with bf.Pipeline() as PIPELINE2:
            # Read in data
            b_read = bisblocks.ReadH5Block([args.outfile], args.gulp, 'f32', space='system')

            # Apply model
            b_read_gpu = bf.blocks.copy(b_read, space='cuda')
            b_amod_gpu = bisblocks.ApplyModelBlock(b_read_gpu, model, x_axis, y_axis)
            b_rate_gpu = bisblocks.CalcRatesBlock(b_amod_gpu, t_axis)

            # Write data
            b_amod = bf.blocks.copy(b_amod_gpu, space='cuda_host')
            b_write2 = bisblocks.WriteH5Block(b_amod, args.outfile, 'detrended')

            # Load rates back to cpu and track them in RAM
            # since we cant simultaniuously write with H5py :(
            b_rate = bf.blocks.copy(b_rate_gpu, space='cuda_host')
            b_racc = bisblocks.AccumRatesBlock(b_rate)

            PIPELINE2.run()

            # Store accumulated stuff
            rates = b_racc.rates

        # Put the rates into the outfile
        with h5py.File(args.outfile, 'a') as fo:
            data = fo.create_dataset('rates', data=rates)

            # Attach scales
            data.dims[0].attach_scale(fo['y'])
            data.dims[0].label = fo['displacements'].dims[0].label
            data.dims[1].attach_scale(fo['x'])
            data.dims[1].label = fo['displacements'].dims[1].label

    if makeplots:
        logger.info('Plots requested')

    total_time = time.time() - start_time
    logger.info(f'Total runtime was {total_time} seconds')

def main(args):

    # Timekeeping
    start_time = time.time()
    
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
    logger.setLevel(logging.DEBUG if args.debug else logging.INFO)
    
    # Useful info
    logger.info(f'Starting bisbas.py with PID {os.getpid()}')
    logger.info(f'Version: {__version__}')
        
    ##### Read in the config file #####
    logger.info(f'Attempting to use {args.config} as config file')
    if not os.path.exists(args.config):
        logger.error('Could not find provided')
    config = configparser.ConfigParser()
    config.optionxform = str #make the config file case-sensitive
    config.read(args.config)
    
    # Get configuration file paramters
    mingood     = config.getint('timeseries-config','nsbas_min_intfs')
    reflat      = config.getfloat('timeseries-config','reflat')
    reflon      = config.getfloat('timeseries-config','reflon')
    refnum      = config.getint('timeseries-config','refnum')
    unw_thresh  = config.getfloat('timeseries-config','unw_check_threshold')
    calcrate    = config.getboolean('timeseries-config','calcrate')
    detrend     = config.getboolean('timeseries-config','detrend')
    trendparams = config.getint('timeseries-config','trendparams')
    gpsfile     = config.get('timeseries-config','gps_file')
    constrained = config.getboolean('timeseries-config','constrainedtrend')
    makeplots   = config.getboolean('timeseries-config','makeplots')

    # Get dates, wavelength
    # conv = (-1000)*wavelen/(4*np.pi)

    # Build G matrix

    # Get median values from ref

    # Run pipeline1

    # Build model

    # Run pipeline2

    # Make plots

if __name__=='__main__':
    globalstart=time.time()
    parser = argparse.ArgumentParser(description='Run ISBAS/SBAS on a GMTSAR-formatted dataset')
    parser.add_argument('-c', '--config', type=str, default='./isbas.config', 
                        help='supply name of config file to setup processing options. Required.')
    parser.add_argument('-l', '--log', type=str, 
                        help='name of logfile to write information to')
    parser.add_argument('-d', '--debug', action='store_true',
                        help='print debug messages as well as info and higher')
    parser.add_argument('-g', '--gulp', type=int, default=1000,
                        help='size of gulps to intake data with')
    parser.add_argument('-o', '--outfile', type=str, default='timeseries.h5',
                        help='name of archive file to write out to')
    args = parser.parse_args()
    
    main(args)