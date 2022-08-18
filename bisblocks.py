#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This file contains various blocks for the Bifrost-ISBAS pipeline

Created on Sun Aug 14 23:38:03 2022
@author: bruzewskis
"""

import numpy as np

import bifrost.pipeline as bfp
from bifrost.dtype import name_nbit2numpy

from reader import DataStack

class IntfRead(object):
    '''
    File read object
    '''
    def __init__(self, filename, gulp_size, dtype):
        if True: #if regions is None
            # Initialize our reader object
            self.step = 0
            self.reader = DataStack.read(filename)

            # Double check this gulp-size is acceptable
            imsize = self.reader.imsize
            if imsize%gulp_size==0:
                self.gulp_size = gulp_size
            else:
                raise ValueError('Gulp must evenly divide image size')

            # Set dtype, maybe we should check this
            self.dtype = dtype
        else:
            # for region in region find moore size
            # pick biggest moore size and pump to gulp_size
            # set up data around region

    def read(self):
        # Figure out what to read and read it
        picks = np.arange(self.step*self.gulp_size, (self.step+1)*self.gulp_size)
        d = self.reader[picks]
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
    def __init__(self, filenames, gulp_size, gulp_nframe, dtype, *args, **kwargs):
        super().__init__(filenames, gulp_nframe, *args, **kwargs)
        self.dtype = dtype
        self.gulp_size = gulp_size

    def create_reader(self, filename):
        # Log line about reading
        # Do a lookup on bifrost datatype to numpy datatype
        dcode = self.dtype.rstrip('0123456789')
        nbits = int(self.dtype[len(dcode):])
        np_dtype = name_nbit2numpy(dcode, nbits)

        return IntfRead(filename, self.gulp_size, np_dtype)

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
            ospans[0].data[...] = indata
            return [1]
        else:
            return [0]
    
    