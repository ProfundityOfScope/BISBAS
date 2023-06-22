#!/usr/bin/env python3
"""
Bifrost implementation of ISBAS algorithms
"""

import os
import sys
import time
import logging
import argparse
import configparser

import h5py
import cupy as cp
import numpy as np
import bifrost as bf

import bisblocks
import helpers
import plotting

__author__ = "Seth Bruzewski"
__credits__ = ["Seth Bruzewski", "Jayce Dowell", "Gregory Taylor"]

__license__ = "MIT"
__version__ = "1.0.3"
__maintainer__ = "Seth Bruzewski"
__email__ = "bruzewskis@unm.edu"
__status__ = "development"


def main(args):

    # Setup a logger
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

    # Read in the config file
    logger.info(f'Attempting to use {args.config} as config file')
    if not os.path.exists(args.config):
        logger.error('Could not find provided')
    config = configparser.ConfigParser()
    config.optionxform = str  # make the config file case-sensitive
    config.read(args.config)

    # Get configuration file paramters
    inname      = config.get('timeseries-config', 'inname')
    outfile     = config.get('timeseries-config', 'outfile')
    outname     = config.get('timeseries-config', 'outname')
    refnum      = config.getint('timeseries-config', 'refnum')
    detrend     = config.getboolean('timeseries-config', 'detrend')
    trendparams = config.getint('timeseries-config', 'trendparams')
    constrained = config.getboolean('timeseries-config', 'constrained')
    gpsfile     = config.get('timeseries-config', 'gps_file')
    detrendname = config.get('timeseries-config', 'detrendname')
    calcrate    = config.getboolean('timeseries-config', 'calcrate')
    ratename    = config.get('timeseries-config', 'ratename')
    makeplots   = config.getboolean('timeseries-config', 'makeplots')
    ninterp     = config.getint('timeseries-config', 'ninterp')

    # Extract things from data
    with h5py.File(args.infile, 'r') as fo:
        logger.debug(f'Getting some metadata from {args.infile}')

        # Record attrs
        attrs = dict(fo.attrs)
        logger.debug(f'Copying {len(attrs)} attributes')

        # Wavelength
        wave = float(attrs['radarWavelength'])
        conv = (-1000)*wave/(4*np.pi)

        # Reference coords
        ref_x = int(attrs['REF_X'])
        ref_y = int(attrs['REF_Y'])
        logger.debug(f'Reference point: ({ref_x}, {ref_y})')

        # Get nearby data median
        _, _, ref_stack = helpers.data_near(fo[inname], ref_x, ref_y, refnum)
        median_stack = np.median(ref_stack, axis=(1, 2))
        logger.debug(f'Found {len(median_stack)} median values')

        # Get dates and date-matrix
        datepairs = fo['date'][:]
        dates = np.sort(np.unique(datepairs))
        G, dates_num = helpers.make_gmatrix(datepairs.astype(str))
        logger.debug(f'Used {len(dates)} dates to generate G-matrix {G.shape}')

    # Overwrite
    logger.debug(f'Generating output file {outfile}')
    if os.path.exists(outfile):
        os.remove(outfile)

    # Generate output file and pass along attrs
    with h5py.File(outfile, 'w') as fo:
        # Copy over attrs
        for key in attrs:
            fo.attrs[key] = attrs[key]

        # Remember dates and their order
        fo['datestr'] = dates
        fo['datenum'] = dates_num

    # Timekeeping on pipeline
    start_time = time.time()

    # Generates the timeseries
    with bf.get_default_pipeline() as PIPELINE1:
        # Read in data and move to GPU
        b_read = bisblocks.ReadH5Block(args.infile, args.gulp, inname,
                                       space='system')
        b_mask = bisblocks.ReadH5Block(args.infile, args.gulp, 'coherence',
                                       space='system')
        b_mskd = bisblocks.MaskBlock(b_read, b_mask, 0.2)
        b_mskd_gpu = bf.blocks.copy(b_mskd, space='cuda')

        # Reference, generate, and convert timeseries
        b_reff_gpu = bisblocks.ReferenceBlock(b_mskd_gpu, median_stack)
        b_tser_gpu = bisblocks.GenTimeseriesBlock(b_reff_gpu, dates_num, G)
        b_tsmm_gpu = bisblocks.ConvertToMillimetersBlock(b_tser_gpu, conv)
        b_accm_gpu = bisblocks.AccumModelBlock(b_tsmm_gpu)
        b_tsmm = bf.blocks.copy(b_tsmm_gpu, space='cuda_host')

        # Write out data and accumulate useful things
        b_write = bisblocks.WriteH5Block(b_tsmm, outfile, outname, True)

        # Start the pipeline
        PIPELINE1.run()

        # Keep track of accumulated values
        GTG = cp.asnumpy(b_accm_gpu.GTG)
        GTd = cp.asnumpy(b_accm_gpu.GTd)

    ts_time = time.time()
    ts_run = ts_time - start_time
    logger.info(f'Finished timeseries generation in {ts_run:.4f} s')

    # If user requested detrend, we do it
    if detrend:
        logger.info('Detrend requested.')

        # Figure out GPS
        if os.path.exists(gpsfile):
            logger.info('Loading GPS file.')
            gps = np.loadtxt(gpsfile)
        else:
            logger.info('No GPS, zeroing at reference point.')
            gps = np.array([[ref_x, ref_y, refnum, 0]])

        # Generate model from accumulated matrices and constraints
        model = helpers.generate_model(outfile, outname, gps, GTG, GTd, True,
                                       trendparams)
        logger.debug(f'Generated a model: {model.shape}, {model.dtype}')

        # Second pipeline
        with bf.Pipeline() as PIPELINE2:
            # Read in data and copy to GPU
            b_read = bisblocks.ReadH5Block(outfile, args.gulp, outname,
                                           space='system')
            b_read_gpu = bf.blocks.copy(b_read, space='cuda')

            # Apply the model to the data, then write to disk
            b_amod_gpu = bisblocks.ApplyModelBlock(b_read_gpu, model)
            b_amod = bf.blocks.copy(b_amod_gpu, space='cuda_host')
            b_write2 = bisblocks.WriteH5Block(b_amod, outfile, detrendname)

            # Calculate average rates, then write rate image to disk
            b_rate_gpu = bisblocks.CalcRatesBlock(b_amod_gpu, dates_num)
            b_rate = bf.blocks.copy(b_rate_gpu, space='cuda_host')
            b_rawr = bisblocks.WriteTempBlock(b_rate, f'{ratename}.dat')

            PIPELINE2.run()

            # Grab useful info
            rate_shape = b_rawr.outshape
            rate_dtype = b_rawr.dtype

        dt_time = time.time()
        dt_run = dt_time - ts_time
        logger.info(f'Finished detrending in {dt_run:.2f} s')

        # Copy temp files to outfile
        logger.debug(f'Copying temp files into {outfile}')
        with h5py.File(outfile, 'a') as fo:
            rates_mm = np.memmap(f'{ratename}.dat', mode='r', shape=rate_shape,
                                 dtype=rate_dtype)
            fo[ratename] = rates_mm[:]
            os.remove(f'{ratename}.dat')

    # Make plots
    if makeplots:
        logger.info('Plots requested')
        with h5py.File(outfile, 'r') as fo:
            plotting.make_image(fo, ratename, 'rates.png')
            plotting.make_video(fo, detrendname, 'rawdata.mp4', 5) #5
            plotting.make_video(fo, detrendname, 'intdata.mp4', 24, 30*24) #24 30*24

    total_time = time.time() - start_time
    logger.info(f'Total runtime was {total_time} seconds')


if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Run BISBAS')
    parser.add_argument('-i', '--infile', type=str, default='ifgramStack.h5',
                        help='name of file to read in and use.')
    parser.add_argument('-c', '--config', type=str, default='./isbas.config',
                        help='supply name of config file.')
    parser.add_argument('-l', '--log', type=str,
                        help='name of logfile to write information to')
    parser.add_argument('-d', '--debug', action='store_true',
                        help='print debug messages as well as info and higher')
    parser.add_argument('-g', '--gulp', type=int, default=1000,
                        help='size of gulps to intake data with')
    args = parser.parse_args()

    main(args)
