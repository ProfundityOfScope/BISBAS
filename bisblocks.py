#!/usr/bin/env python3
"""
This file contains various blocks for the Bifrost-ISBAS pipeline
"""

import os
import logging
from copy import deepcopy
from datetime import datetime

import h5py
import cupy as cp
import numpy as np
import bifrost as bf
import bifrost.pipeline as bfp
from bifrost.dtype import numpy2string, string2numpy

__author__ = "Seth Bruzewski"
__credits__ = ["Seth Bruzewski", "Jayce Dowell", "Gregory Taylor"]

__license__ = "MIT"
__version__ = "1.0.2"
__maintainer__ = "Seth Bruzewski"
__email__ = "bruzewskis@unm.edu"
__status__ = "development"

blockslogger = logging.getLogger('__main__')


class H5Reader(object):
    '''
    File read object
    '''
    def __init__(self, filename, dataname, gulp_size):

        # Initialize our reader object
        self.fo = h5py.File(filename, 'a')
        self.data = self.fo[dataname]
        self.dtype = self.data.dtype

        # Double check this gulp-size is acceptable
        self.shape = self.data.shape
        self.size = np.product(self.shape)
        self.imsize = np.product(self.shape[-2:])
        if self.imsize%gulp_size==0:
            self.gulp_size = gulp_size
        else:
            raise ValueError('Gulp must evenly divide image size')

        # log
        blockslogger.debug(f'Reading {dataname} {self.shape} from {filename}')

        # Make a buffer for reading (hdf5 being picky)
        self.linelen = np.size(self, 2)
        bsize = 2*max(self.gulp_size, self.linelen)
        self.buffer = np.zeros((bsize, np.size(self, 0)), dtype=self.dtype)
        blockslogger.debug(f'Created read buffer with shape {self.buffer.shape}')
        self.head = 0
        self.linecount = 0

    def read(self):

        try:
            # This will read via the buffer
            while self.head < self.gulp_size:
                stop = self.head + self.linelen
                self.buffer[self.head:stop] = self.data[:, self.linecount].T

                self.head += self.linelen
                self.linecount += 1

            out = self.buffer[:self.gulp_size]
            self.head -= self.gulp_size
            self.buffer = np.roll(self.buffer, -self.gulp_size, axis=0)

            return out
        except IndexError:
            # Catch the index error if we're past the end
            return np.empty((0, np.size(self, 0)), dtype=self.dtype)

    def __enter__(self):
        return self

    def close(self):
        pass

    def __exit__(self, type, value, tb):
        self.fo.close()

class ReadH5Block(bfp.SourceBlock):
    """ 
    Meant for reading our data, could be generalized, but difficult. Currently assumes
    we want to keep first dimension, the there are two more we want to ravel, basically.
    """

    def __init__(self, filename, dataname, gulp_pixels, *args, **kwargs):
        super().__init__([filename], 1, *args, **kwargs)
        self.filename = filename
        self.dataname = dataname
        self.gulp_pixels = gulp_pixels

    def create_reader(self, filename):

        return H5Reader(filename, self.dataname, self.gulp_pixels)

    def on_sequence(self, ireader, filename):
        dshape = ireader.shape
        dtype_str = numpy2string(ireader.dtype)
        ohdr = {'name':     self.filename,
                'dataname': self.dataname,
                'inshape':  str(dshape),
                '_tensor':  {'dtype':  dtype_str,
                             'shape':  [-1, self.gulp_pixels, dshape[0]],
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

class WriteH5Block(bfp.SinkBlock):

    def __init__(self, iring, filename, dataname, *args, **kwargs):
        super().__init__(iring, *args, **kwargs)
        self.fo = h5py.File(filename, 'a')
        self.filename = filename
        self.dataname = dataname

    def on_sequence(self, iseq):

        # Grab useful things from header
        hdr = iseq.header
        span, self.gulp, depth = hdr['_tensor']['shape']
        dtype_np = string2numpy(hdr['_tensor']['dtype'])

        # Grab useful things from file
        inshape = eval(hdr['inshape'])
        outshape = (depth, inshape[1], inshape[2])

        blockslogger.debug(f'Writing to {self.dataname} {outshape} in {self.filename}')

        # Create dataset
        if self.dataname in self.fo:
            # should probably verify this is good
            self.data = self.fo[self.dataname]
        else:
            self.data = self.fo.create_dataset(self.dataname, 
                                          data=np.empty(outshape, 
                                                        dtype=dtype_np))

        # Record gulp, set up buffer
        self.linelen = outshape[2]
        self.buffer = np.empty((2*max([self.gulp, self.linelen]), depth), 
                               dtype=dtype_np)
        blockslogger.debug(f'Created write buffer with shape {self.buffer.shape}')
        self.head = 0
        self.linecount = 0

    def on_data(self, ispan):

        # Put data into the file
        self.buffer[self.head:self.head+self.gulp] = ispan.data[0]
        self.head += self.gulp

        # Write out as many times as needed
        while self.head > self.linelen:
            self.data[:,self.linecount] = self.buffer[:self.linelen].T
            self.linecount += 1

            self.head -= self.linelen
            self.buffer = np.roll(self.buffer, -self.linelen, axis=0)

class ReferenceBlock(bfp.TransformBlock):
    def __init__(self, iring, ref_stack, *args, **kwargs):
        super().__init__(iring, *args, **kwargs)
        self.ref_stack = cp.asarray(ref_stack)

    def on_sequence(self, iseq):
        ohdr = deepcopy(iseq.header)
        ohdr["name"] += "_referenced"

        blockslogger.debug('Started ReferenceBlock')

        return ohdr

    def on_data(self, ispan, ospan):
        in_nframe  = ispan.nframe
        out_nframe = in_nframe

        stream = bf.device.get_stream()
        with cp.cuda.ExternalStream(stream):
            idata = ispan.data.as_cupy()
            odata = ospan.data.as_cupy()

            odata[...] = idata
            odata -= self.ref_stack
            ospan.data[...] = bf.ndarray(odata)

        return out_nframe

class GenTimeseriesBlock(bfp.TransformBlock):
    ''' (1,npix,nintf) -> (1,npix,ndates) '''

    def __init__(self, iring, dates, G, *args, **kwargs):
        super().__init__(iring, *args, **kwargs)
        self.dates = cp.asarray(dates)
        self.nd = len(dates) #cp
        self.G = cp.asarray(G) #cp

    def on_sequence(self, iseq):

        ohdr = deepcopy(iseq.header)
        ohdr['name'] += '_as_ts'
        ohdr['_tensor']['shape'][-1] = self.nd

        blockslogger.debug('Started GenTimeseriesBlock')

        return ohdr

    def on_data(self, ispan, ospan):
        in_nframe  = ispan.nframe
        out_nframe = in_nframe

        stream = bf.device.get_stream()
        with cp.cuda.ExternalStream(stream):
            idata = ispan.data.as_cupy() 
            odata = ospan.data.as_cupy()

            # Set up matrices to solve
            M = ~cp.isnan(idata[0])
            A = cp.matmul(self.G.T[None, :, :], M[:, :, None] * self.G[None, :, :]).astype(idata.dtype)
            B = cp.nansum(self.G.T[:, :, None] * (M*idata[0]).T[None, :, :], axis=1).T

            # Mask out low-rank values
            lowrank = cp.linalg.matrix_rank(A) != self.nd - 1
            A[lowrank] = cp.eye(self.nd-1)
            B[lowrank] = cp.full(self.nd-1, np.nan)

            # Solve
            model = cp.linalg.solve(A, B)

            # Turn it into a cumulative timeseries
            datediffs = (self.dates - cp.roll(self.dates, 1))[1:]
            changes = datediffs[None,:] * model
            ts = cp.zeros((1,cp.size(idata[0], 0), self.nd))
            ts[:,:,1:] = cp.cumsum(changes, axis=1)

            odata[...] = ts
            ospan.data[...] = bf.ndarray(odata)

        return out_nframe

class ConvertToMillimetersBlock(bfp.TransformBlock):

    def __init__(self, iring, conv, *args, **kwargs):
        super().__init__(iring, *args, **kwargs)
        self.conv = conv

    def on_sequence(self, iseq):
        ohdr = deepcopy(iseq.header)
        ohdr['conversion'] = f'{self.conv}'

        blockslogger.debug('Started ConvertToMillimetersBlock')
        return ohdr

    def on_data(self, ispan, ospan):
        in_nframe  = ispan.nframe
        out_nframe = in_nframe

        stream = bf.device.get_stream()
        with cp.cuda.ExternalStream(stream):
            idata = ispan.data.as_cupy() 
            odata = ospan.data.as_cupy()

            odata[...] = idata
            odata *= self.conv
            ospan.data[...] = bf.ndarray(odata)# may be unneeded?

        return out_nframe

class AccumModelBlock(bfp.SinkBlock):
    '''
    TBD
    '''
    def __init__(self, iring, *args, **kwargs):
        super().__init__(iring, *args, **kwargs)

    def on_sequence(self, iseq):

        # Grab useful things from header
        hdr = iseq.header
        span, self.gulp, depth = hdr['_tensor']['shape']

        # Grab useful things from file
        inshape = eval(hdr['inshape'])
        self.imshape = (inshape[1], inshape[2])

        # Set up some stuff for the accumulation (keeping all terms)
        self.GTG = cp.zeros((depth, 6, 6))
        self.GTd = cp.zeros((depth, 6))
        self.niter = 0

    def on_data(self, ispan):
        in_nframe  = ispan.nframe

        stream = bf.device.get_stream()
        with cp.cuda.ExternalStream(stream):
            idata = ispan.data[0].as_cupy() 

            ### ACCUMULATE FOR DOTS ###
            # Figure out what the G matrix should look like
            inds = self.niter * self.gulp + cp.arange(0, self.gulp)
            yinds, xinds = cp.unravel_index(inds, self.imshape)
            ones = cp.ones_like(xinds)
            G = cp.column_stack([ones, xinds, yinds, xinds**2, yinds**2, xinds*yinds])

            # Accumulate dot-product
            """
            By elevating to a tensor problem, we can perform multiple solves
            and avoid a lot of nasty NaN values

            GTG = (ng,nd)(ng,6)(ng,6)->(nd,6,6)
            GTd = (ng,6)(ng,nd)->(nd,6)
            """
            M = ~cp.isnan(idata)
            self.GTG += cp.einsum('jl,ji,jk->lik', M, G, G)
            self.GTd += cp.nansum(cp.einsum('jk,ji->ijk', G, idata), axis=1)
            self.niter += 1

class ApplyModelBlock(bfp.TransformBlock):

    def __init__(self, iring, models, *args, **kwargs):
        super().__init__(iring, *args, **kwargs)
        self.models = cp.asarray(models)

        self.step = 0
        self.ntrend = models.shape[0]

    def on_sequence(self, iseq):
        ohdr = deepcopy(iseq.header)

        blockslogger.debug('Started ApplyModelBlock')

        return ohdr

    def on_data(self, ispan, ospan):
        in_nframe  = ispan.nframe
        out_nframe = in_nframe

        stream = bf.device.get_stream()
        with cp.cuda.ExternalStream(stream):
            idata = ispan.data.as_cupy() 
            odata = ospan.data.as_cupy()

            # Do some stuff with idata
            gulp_size = cp.size(idata[0],0)
            r_start = self.step * gulp_size
            r_end = (self.step+1) * gulp_size
            yc, xc = cp.unravel_index(cp.arange(r_start, r_end), self.imshape)

            # d(7800) m(3,20) -> c(7800,20)
            ones = cp.full_like(xc, 1)
            raw = cp.column_stack([ones, xc, yc, xc**2, yc**2, xc*yc])
            corr = cp.dot(raw[:,:self.ntrend], self.models)
            corr = cp.expand_dims(corr, axis=0)

            self.step += 1

            odata[...] = idata
            odata -= corr
            ospan.data[...] = bf.ndarray(odata)

        return out_nframe

class CalcRatesBlock(bfp.TransformBlock):

    def __init__(self, iring, taxis, deg=1, *args, **kwargs):
        super().__init__(iring, *args, **kwargs)
        self.taxis = cp.asarray(taxis)
        self.deg = deg

    def on_sequence(self, iseq):
        ohdr = deepcopy(iseq.header)
        shape = ohdr['inshape']
        blockslogger.debug(f'=============={shape}')
        ohdr['_tensor']['shape'] = iseq.header['_tensor']['shape'][:-1]

        blockslogger.debug('Started CalcRatesBlock')

        return ohdr

    def on_data(self, ispan, ospan):
        in_nframe  = ispan.nframe
        out_nframe = in_nframe

        stream = bf.device.get_stream()
        with cp.cuda.ExternalStream(stream):
            idata = ispan.data.as_cupy() 
            odata = ospan.data.as_cupy()

            fits = cp.polyfit(self.taxis, idata[0].T, 1)
            rate = fits[0].reshape((1,-1))

            odata[...] = rate
            ospan.data[...] = bf.ndarray(odata)

        return out_nframe

class WriteRatesBlock(bfp.SinkBlock):

    def __init__(self, iring, outfile, outname, *args, **kwargs):
        super().__init__(iring, *args, **kwargs)

        # Set up accumulation
        self.niter = 0

    def on_sequence(self, iseq):

        # Grab useful things from header
        hdr = iseq.header

        blockslogger.debug('Started AccumRatesBlock')

    def on_data(self, ispan):
        pass
