#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This file contains various blocks for the Bifrost-ISBAS pipeline

Created on Sun Aug 14 23:38:03 2022
@author: bruzewskis
"""

import numpy as np
import logging
from copy import deepcopy
from datetime import datetime

import bifrost.pipeline as bfp
from bifrost.dtype import name_nbit2numpy

from readers import DataStack

__version__ = 0.1

blockslogger = logging.getLogger('__main__')

class IntfRead(object):
    '''
    File read object
    '''
    def __init__(self, filename, gulp_size, dtype, file_order):

        # Figure out order
        files = [ f'{filename}/{f}' for f in file_order ]
        self.files = files

        # Initialize our reader object
        self.step = 0
        self.reader = DataStack.read(files)
        self.dtype = dtype

        # Double check this gulp-size is acceptable
        imsize = self.reader.imsize
        if imsize%gulp_size==0:
            self.gulp_size = gulp_size
        else:
            raise ValueError('Gulp must evenly divide image size')

        # Generate regions for entire image
        self.regions = np.arange(0, imsize).reshape(-1, self.gulp_size)
        blockslogger.debug(f'Regions have shape {self.regions.shape}')


    def read(self):

        try:
            # We try to read files
            picks = self.regions[self.step]
            d = self.reader[picks]
            self.step += 1

            return d.astype(self.dtype)
        except IndexError:
            # Catch the index error if we're past the end
            return np.empty((0, len(self.files), 3), dtype=self.dtype)

    def __enter__(self):
        return self

    def close(self):
        pass

    def __exit__(self, type, value, tb):
        self.close()
        
class IntfReadBlock(bfp.SourceBlock):
    """ Block for reading binary data from file and streaming it into a bifrost pipeline

    Args:
        filenames (list): A list of filenames to open
        gulp_size (int): Number of elements in a gulp (i.e. sub-array size)
        gulp_nframe (int): Number of frames in a gulp. (Ask Ben / Miles for good explanation)
        dtype (bifrost dtype string): dtype, e.g. f32, cf32
    """
    def __init__(self, filenames, gulp_pixels, dtype, file_order, *args, **kwargs):
        super().__init__(filenames, 1, *args, **kwargs)
        self.dtype = dtype
        self.file_order = file_order
        self.gulp_pixels = gulp_pixels

    def create_reader(self, filename):
        # Log line about reading
        # Do a lookup on bifrost datatype to numpy datatype
        dcode = self.dtype.rstrip('0123456789')
        nbits = int(self.dtype[len(dcode):])
        np_dtype = name_nbit2numpy(dcode, nbits)

        return IntfRead(filename, self.gulp_pixels, np_dtype, file_order=self.file_order)

    def on_sequence(self, ireader, filename):
        ohdr = {'name': filename,
                '_tensor': {
                        'dtype':  self.dtype,
                        'shape':  [-1, self.gulp_pixels, len(self.file_order), 3], #This line needs changing
                        },
                }
        return [ohdr]

    def on_data(self, reader, ospans):
        indata = reader.read()

        if indata.shape[0] == self.gulp_pixels:
            ospans[0].data[...] = indata
            return [1]
        else:
            return [0]

class ReferenceBlock(bfp.TransformBlock):
    def __init__(self, iring, ref_stack, *args, **kwargs):
        super().__init__(iring, *args, **kwargs)
        self.ref_stack = ref_stack

    def on_sequence(self, iseq):
        ohdr = deepcopy(iseq.header)
        ohdr["name"] += "_referenced"
        return ohdr

    def on_data(self, ispan, ospan):
        in_nframe  = ispan.nframe
        out_nframe = in_nframe

        idata = ispan.data
        odata = ospan.data

        odata[...] = idata
        odata[:,:,:,2] -= self.ref_stack
        return out_nframe

class GenTimeseriesBlock(bfp.TransformBlock):
    ''' (1,npix,nintf,3) -> (1,npix,ndates,3) '''

    def __init__(self, iring, dates, G, *args, **kwargs):
        super().__init__(iring, *args, **kwargs)
        self.dates = dates
        self.G = G

    def on_sequence(self, iseq):
        ohdr = deepcopy(iseq.header)
        ohdr['name'] += '_as_ts'
        ohdr['_tensor']['shape'][2] = len(self.dates)
        return ohdr

    def on_data(self, ispan, ospan):
        in_nframe  = ispan.nframe
        out_nframe = in_nframe

        idata = ispan.data
        odata = ospan.data

        # Set up matrices to solve
        zdata = idata[0,:,:,2]
        M = ~np.isnan(zdata)
        A = np.matmul(self.G.T[None, :, :], M[:, :, None] * self.G[None, :, :])
        B = np.nansum(self.G.T[:, :, None] * (M*dataz).T[None, :, :], axis=1).T

        # Mask out low-rank values
        lowrank = np.linalg.matrix_rank(A) != len(dates) - 1
        A[lowrank] = np.eye(len(dates)-1)
        B[lowrank] = np.full(len(dates)-1, np.nan)

        # Solve
        model = np.linalg.solve(A, B)

        # Turn it into a cumulative timeseries
        datediffs = (self.dates - np.roll(self.dates, 1))[1:]
        changes = datediffs[None,:] * model
        ts = np.zeros((1,np.size(dataz,0), len(self.dates),3))
        ts[:, :, 1:, 2] = np.cumsum(changes, axis=1)

        odata[...] = ts
        return out_nframe
    
class PrintStuffBlock(bfp.SinkBlock):
    def __init__(self, iring, *args, **kwargs):
        super().__init__(iring, *args, **kwargs)
        self.n_iter = 0

    def on_sequence(self, iseq):
        self.n_iter = 0
        blockslogger.info(f'Call to on_sequence at {self.n_iter}')
        blockslogger.info(f'On sequence looking header: {iseq.header}')

    def on_data(self, ispan):
        now = datetime.now()
        blockslogger.info(f'On {self.n_iter} | {ispan.data.shape}')
        self.n_iter += 1
    