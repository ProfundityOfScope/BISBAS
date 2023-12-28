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
__version__ = "1.1.0"
__maintainer__ = "Seth Bruzewski"
__email__ = "bruzewskis@unm.edu"
__status__ = "development"


def main(args):
    '''Main function for bisbas.py'''

    # Setup a logger
    logger = logging.getLogger(__name__)
    log_format = logging.Formatter('%(asctime)s [%(levelname)-8s] %(message)s',
                                  datefmt='%Y-%m-%d %H:%M:%S')
    log_format.converter = time.gmtime

    # Decide to write to file or stdout
    if args.log is None:
        log_handler = logging.StreamHandler(sys.stdout)
    else:
        log_handler = logging.FileHandler(args.log)
    log_handler.setFormatter(log_format)
    logger.addHandler(log_handler)

    # Decide what level to report
    logger.setLevel(logging.DEBUG if args.debug else logging.INFO)

    # Useful info
    logger.info('Starting bisbas.py with PID %s', os.getpid())
    logger.info('Using bifrost version %s', bf.__version__)

    # Read in the config file
    logger.info('Attempting to use %s as config file', args.config)
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
    mincoher    = config.getfloat('timeseries-config', 'min_coherence')
    detrend     = config.getboolean('timeseries-config', 'detrend')
    constrained = config.getboolean('timeseries-config', 'constrained')
    trendparams = config.getint('timeseries-config', 'trendparams')
    gpsfile     = config.get('timeseries-config', 'gps_file')
    detrendname = config.get('timeseries-config', 'detrendname')
    ratename    = config.get('timeseries-config', 'ratename')
    makeplots   = config.getboolean('timeseries-config', 'makeplots')

    # Extract things from data
    with h5py.File(args.infile, 'r') as fo:
        logger.debug('Getting some metadata from %s', args.infile)

        # Record attrs
        attrs = dict(fo.attrs)
        logger.debug('Copying %s attributes', len(attrs))

        # Wavelength
        wave = float(attrs['radarWavelength'])
        conv = (-1000)*wave/(4*np.pi)

        # Reference coords
        ref_x = int(attrs['REF_X'])
        ref_y = int(attrs['REF_Y'])
        logger.debug('Reference point: (%s, %s)', ref_x, ref_y)

        # Get nearby data median
        _, _, ref_stack = helpers.data_near(fo[inname], ref_x, ref_y, refnum)
        median_stack = np.median(ref_stack, axis=(1, 2))
        logger.debug('Found %s median values', len(median_stack))

        # Get dates and date-matrix
        datepairs = np.array(fo['date'][:])
        dates = np.sort(np.unique(datepairs))
        date_matrix, dates_num = helpers.make_gmatrix(datepairs.astype(str))
        logger.debug('Used %s dates to generate G-matrix %s', len(dates),
                     date_matrix.shape)

        # Find best gulp size
        if args.gulp is None:
            logger.info('No gulp size provided, calculating best size')
            imsize = fo[inname][0].size
            ni = len(datepairs)
            nd = len(dates)
            gulp_size = helpers.auto_best_gulp(ni, nd, imsize, 7)
        else:
            gulp_size = args.gulp


    # Overwrite
    logger.debug('Generating output file %s', outfile)
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
    with bf.get_default_pipeline() as pipeline1:
        # Read in data and move to GPU
        b_read = bisblocks.ReadH5Block(args.infile, gulp_size, inname,
                                       space='system')
        b_mask = bisblocks.ReadH5Block(args.infile, gulp_size, 'coherence',
                                       space='system')
        b_mskd = bisblocks.MaskBlock(b_read, b_mask, mincoher)
        b_mskd_gpu = bf.blocks.copy(b_mskd, space='cuda')

        # Reference, generate, and convert timeseries
        b_reff_gpu = bisblocks.ReferenceBlock(b_mskd_gpu, median_stack)
        b_tser_gpu = bisblocks.GenTimeseriesBlock(b_reff_gpu, dates_num, date_matrix)
        b_tsmm_gpu = bisblocks.ConvertToMillimetersBlock(b_tser_gpu, conv)
        b_accm_gpu = bisblocks.AccumModelBlock(b_tsmm_gpu)
        b_tsmm = bf.blocks.copy(b_tsmm_gpu, space='cuda_host')

        # Write out data and accumulate useful things
        bisblocks.WriteH5Block(b_tsmm, outfile, outname, True)

        # Start the pipeline
        pipeline1.run()

        # Keep track of accumulated values
        gtg_matrix = cp.asnumpy(b_accm_gpu.GTG)
        gtd_matrix = cp.asnumpy(b_accm_gpu.GTd)

    ts_time = time.time()
    ts_run = ts_time - start_time
    logger.info('Finished timeseries generation in %.4f s', ts_run)

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
        model = helpers.generate_model(outfile, outname, gps, gtg_matrix,
                                       gtd_matrix, constrained, trendparams)
        logger.debug('Generated a model: %s %s', model.shape, model.dtype)

        # Second pipeline
        with bf.Pipeline() as pipeline2:
            # Read in data and copy to GPU
            b_read = bisblocks.ReadH5Block(outfile, gulp_size, outname,
                                           space='system')
            b_read_gpu = bf.blocks.copy(b_read, space='cuda')

            # Apply the model to the data, then write to disk
            b_amod_gpu = bisblocks.ApplyModelBlock(b_read_gpu, model)
            b_amod = bf.blocks.copy(b_amod_gpu, space='cuda_host')
            bisblocks.WriteH5Block(b_amod, outfile, detrendname)

            # Calculate average rates, then write rate image to disk
            b_rate_gpu = bisblocks.CalcRatesBlock(b_amod_gpu, dates_num)
            b_rate = bf.blocks.copy(b_rate_gpu, space='cuda_host')
            b_rawr = bisblocks.WriteTempBlock(b_rate, f'{ratename}.dat')

            pipeline2.run()

            # Grab useful info
            rate_shape = b_rawr.outshape
            rate_dtype = b_rawr.dtype

        dt_time = time.time()
        dt_run = dt_time - ts_time
        logger.info('Finished detrending in %.4f s', dt_run)

        # Copy temp files to outfile
        logger.debug('Copying temp files into %s', outfile)
        with h5py.File(outfile, 'a') as fo:
            rates_mm = np.memmap(f'{ratename}.dat', mode='r', shape=rate_shape,
                                 dtype=rate_dtype)
            fo[ratename] = rates_mm[:]
            os.remove(f'{ratename}.dat')

    # Make plots
    if makeplots:
        logger.info('Plots requested')
        with h5py.File(outfile, 'r') as fo:
            rates = fo[ratename][0]
            plotting.make_image(rates, outfile='rates.png')
            logger.info('Generated a rate map')
            plotting.stretch_video(fo, 'detrended', 'detrended.mp4', 10)
            logger.info('Generated a data video')

    total_time = time.time() - start_time
    logger.info('Finished in %.2f s', total_time)


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
    parser.add_argument('-g', '--gulp', type=int, default=None,
                        help='size of gulps to intake data with')
    pargs = parser.parse_args()

    main(pargs)
