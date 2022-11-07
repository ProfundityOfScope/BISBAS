#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This file contains various blocks for the Bifrost-ISBAS pipeline

Created on Sun Aug 14 23:38:03 2022
@author: bruzewskis
"""

import numpy as np
import logging
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


    def read(self):
        # Figure out what to read and read it
        d = self.reader[self.regions[self.step]]
        self.step += 1

        return d.astype(self.dtype)

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
    def __init__(self, filenames, gulp_size, gulp_nframe, dtype, file_order, *args, **kwargs):
        super().__init__(filenames, gulp_nframe, *args, **kwargs)
        self.dtype = dtype
        self.gulp_size = gulp_size
        self.file_order = file_order

    def create_reader(self, filename):
        # Log line about reading
        # Do a lookup on bifrost datatype to numpy datatype
        dcode = self.dtype.rstrip('0123456789')
        nbits = int(self.dtype[len(dcode):])
        np_dtype = name_nbit2numpy(dcode, nbits)

        return IntfRead(filename, self.gulp_size, np_dtype, file_order=self.file_order)

    def on_sequence(self, ireader, filename):
        ohdr = {'name': filename,
                '_tensor': {
                        'dtype':  self.dtype,
                        'shape':  [-1, self.gulp_size, 3], #This line needs changing
                        },
                }
        return [ohdr]

    def on_data(self, reader, ospans):
        indata = reader.read()

        if indata.shape[1] == self.gulp_size:
            blockslogger.debug(f'BREAKS HERE {ospans[0].data.shape} and {indata.shape}')
            ospans[0].data[...] = indata
            return [1]
        else:
            return [0]
    
class PrintStuffBlock(bfp.SinkBlock):
    def __init__(self, iring, *args, **kwargs):
        super(PrintStuffBlock, self).__init__(iring, *args, **kwargs)
        self.n_iter = 0

    def on_sequence(self, iseq):
        self.n_iter = 0

    def on_data(self, ispan):
        now = datetime.now()
        if self.n_iter % 100 == 0:
            print(f'{now} | {self.n_iter} | {ispan.data.shape}')
        self.n_iter += 1
    