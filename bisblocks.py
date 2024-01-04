#!/usr/bin/env python3
"""
This file contains various blocks for the Bifrost-ISBAS pipeline
"""

import logging
from copy import deepcopy
from ast import literal_eval

import h5py
import cupy as cp
import numpy as np
import bifrost as bf
import bifrost.pipeline as bfp
from bifrost.dtype import numpy2string, string2numpy

__author__ = "Seth Bruzewski"
__credits__ = ["Seth Bruzewski", "Jayce Dowell", "Gregory Taylor"]

__license__ = "MIT"
__version__ = "1.0.4"
__maintainer__ = "Seth Bruzewski"
__email__ = "bruzewskis@unm.edu"
__status__ = "development"

blockslogger = logging.getLogger('__main__')

# pylint: disable=W1113

class H5Reader(object):
    """File read object."""

    def __init__(self, filename, dataname, gulp_size):

        # Initialize our reader object
        self.fo = h5py.File(filename, 'a')
        self.data = self.fo[dataname]
        self.dtype = self.data.dtype

        # Double check this gulp-size is acceptable
        self.shape = self.data.shape
        self.size = np.product(self.shape)
        self.imsize = np.product(self.shape[-2:])
        if self.imsize % gulp_size == 0:
            self.gulp_size = gulp_size
        else:
            raise ValueError(f'Gulp {gulp_size} must evenly divide imsize {self.imsize}')

        blockslogger.debug('Reading %s %s from %s', dataname, self.shape,
                           filename)

        # Make a buffer for reading (hdf5 being picky)
        self.linelen = np.size(self, 2)
        bsize = 2*max(self.gulp_size, self.linelen)
        self.buffer = np.zeros((bsize, np.size(self, 0)), dtype=self.dtype)
        blockslogger.debug('Created read buffer of shape %s', self.buffer.shape)
        self.head = 0
        self.linecount = 0

    def read(self):
        '''Reads a gulp of data from the file.'''

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
        """Close the file."""

    def __exit__(self, exc_type, value, traceback):
        self.fo.close()


class ReadH5Block(bfp.SourceBlock):
    """
    Block for reading.

    Meant for reading our data, could be generalized, but difficult. Currently
    assumes we want to keep first dimension, the there are two more we want to
    ravel, basically.
    """

    def __init__(self, filename, gulp_pixels, dataname, *args,
                 **kwargs):
        super().__init__([filename], 1, *args, **kwargs)
        self.filename = filename
        self.dataname = dataname
        self.gulp_pixels = gulp_pixels

    def create_reader(self, sourcename):
        return H5Reader(sourcename, self.dataname, self.gulp_pixels)

    def on_sequence(self, reader, sourcename):
        dshape = reader.shape
        dtype_str = numpy2string(reader.dtype)
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
    '''Block for writing to HDF5 files.'''
    # Things we'll set up along the way
    gulp_size = None
    data = None
    linelen = None
    buffer = None
    head = None
    linecount = None

    def __init__(self, iring, filename, dataname, *args, **kwargs):
        super().__init__(iring, *args, **kwargs)
        self.fo = h5py.File(filename, 'a')
        self.filename = filename
        self.dataname = dataname



    def on_sequence(self, iseq):

        # Grab useful things from header
        hdr = iseq.header
        _, self.gulp_size, depth = hdr['_tensor']['shape']
        dtype_np = string2numpy(hdr['_tensor']['dtype'])

        # Grab useful things from file
        inshape = literal_eval(hdr['inshape'])
        outshape = (depth, inshape[1], inshape[2])

        blockslogger.debug('Writing to %s %s in %s', self.dataname, outshape,
                           self.filename)

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
        self.buffer = np.empty((2*max([self.gulp_size, self.linelen]), depth),
                               dtype=dtype_np)
        blockslogger.debug('Created write buffer with shape %s', self.buffer.shape)
        self.head = 0
        self.linecount = 0

    def on_data(self, ispan):

        # Put data into the file
        self.buffer[self.head:self.head+self.gulp_size] = ispan.data[0]
        self.head += self.gulp_size

        # Write out as many times as needed
        while self.head >= self.linelen:
            self.data[:,self.linecount] = self.buffer[:self.linelen].T
            self.linecount += 1

            self.head -= self.linelen
            self.buffer = np.roll(self.buffer, -self.linelen, axis=0)

            blockslogger.debug('Wrote out line %s', self.linecount-1)

    def on_skip(self, *args, **kwargs):
        raise NotImplementedError

class MaskBlock(bfp.MultiTransformBlock):
    '''Block for masking out low coherence values.'''

    def __init__(self, iring1, iring2, min_coherence=0.3, *args, **kwargs):
        super().__init__([iring1, iring2], *args, **kwargs)
        self.cutoff = min_coherence

    def on_sequence(self, iseqs):
        hdrs = [ iseqs[0].header ]
        return hdrs

    def on_data(self, ispans, ospans):
        in_nframe1 = ispans[0].nframe
        out_nframe = in_nframe1

        idata = ispans[0].data
        imask = ispans[1].data
        odata = ospans[0].data

        odata[...] = np.where(imask > self.cutoff, idata, np.nan)
        return [out_nframe]

    def on_skip(self, *args, **kwargs):
        raise NotImplementedError


class ReferenceBlock(bfp.TransformBlock):
    """Reference the data to a particular coordinate."""

    def __init__(self, iring, ref_stack, *args, **kwargs):
        super().__init__(iring, *args, **kwargs)
        self.ref_stack = cp.asarray(ref_stack)

    def on_sequence(self, iseq):
        ohdr = deepcopy(iseq.header)
        ohdr["name"] += "_referenced"

        blockslogger.debug('Started ReferenceBlock')

        return ohdr

    def on_data(self, ispan, ospan):
        in_nframe = ispan.nframe
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
    """Do the math to convert the interferograms to a timeseries."""

    def __init__(self, iring, dates, G, filter_value=10, *args, **kwargs):
        super().__init__(iring, *args, **kwargs)
        self.dates = cp.asarray(dates)
        self.nd = len(dates)
        self.time_mat = cp.asarray(G).astype('int32') # max-min < 2**31
        self.filter = filter_value

        # This will be useful
        self.datediffs = (self.dates - cp.roll(self.dates, 1))[None, 1:]

    def on_sequence(self, iseq):

        ohdr = deepcopy(iseq.header)
        ohdr['name'] += '_as_ts'
        ohdr['_tensor']['shape'][-1] = self.nd

        blockslogger.debug('Started GenTimeseriesBlock')

        return ohdr

    def on_data(self, ispan, ospan):
        in_nframe = ispan.nframe
        out_nframe = in_nframe

        stream = bf.device.get_stream()
        with cp.cuda.ExternalStream(stream):
            idata = ispan.data.as_cupy()
            odata = ospan.data.as_cupy()

            # Set up matrices to solve
            mask = ~cp.isnan(idata[0])
            left = cp.matmul(self.time_mat.T[None, :, :],
                             mask[:, :, None] * self.time_mat[None, :, :])
            right = cp.nansum(self.time_mat.T[:, :, None] * (mask*idata[0]).T[None, :, :], axis=1).T

            # Mask out low-rank values
            # note: These A matrices will always be symmetric, positive, and
            # real. Will generally be sparse. Determinant is fastest to work
            # with, but prone to numerical instability when on the GPU. We
            # still need to figure out a good way to work around this
            sign, _ = cp.linalg.slogdet(left)
            lowrank = cp.less(sign, 1)

            # Mask low rank
            left[lowrank] = cp.eye(self.nd-1)
            left[lowrank] = cp.full(self.nd-1, np.nan)

            # Solve
            model = cp.linalg.solve(left, right)

            # Filter nasty values, warn if we find any
            condition = cp.abs(model) > self.filter
            if cp.any(condition):
                blockslogger.warning('Found %s values bigger than filter(%.3f)',
                                     np.sum(condition), self.filter)
            model = cp.where(condition, np.nan, model)

            # Turn it into a cumulative timeseries
            changes = self.datediffs * model
            ts = cp.zeros((1, cp.size(idata[0], 0), self.nd))
            ts[:, :, 1:] = cp.cumsum(changes, axis=1)

            odata[...] = ts
            ospan.data[...] = bf.ndarray(odata)

        return out_nframe


class ConvertToMillimetersBlock(bfp.TransformBlock):
    """Convert the units from radians to millimeters."""

    def __init__(self, iring, conv, *args, **kwargs):
        super().__init__(iring, *args, **kwargs)
        self.conv = conv

    def on_sequence(self, iseq):
        ohdr = deepcopy(iseq.header)
        ohdr['conversion'] = f'{self.conv}'

        blockslogger.debug('Started ConvertToMillimetersBlock')
        return ohdr

    def on_data(self, ispan, ospan):
        in_nframe = ispan.nframe
        out_nframe = in_nframe

        stream = bf.device.get_stream()
        with cp.cuda.ExternalStream(stream):
            idata = ispan.data.as_cupy()
            odata = ospan.data.as_cupy()

            odata[...] = idata
            odata *= self.conv
            ospan.data[...] = bf.ndarray(odata)

        return out_nframe


class AccumModelBlock(bfp.SinkBlock):
    """Accumate matrix for future dot product."""

    # Initialize things we'll need along the way
    gulp_size = None
    imshape = None
    leftmat = None
    rightmat = None
    niter = None

    def on_sequence(self, iseq):

        # Grab useful things from header
        hdr = iseq.header
        _, self.gulp_size, depth = hdr['_tensor']['shape']

        # Grab useful things from file
        inshape = literal_eval(hdr['inshape'])
        self.imshape = (inshape[1], inshape[2])

        # Set up some stuff for the accumulation (keeping all terms)
        self.leftmat = cp.zeros((depth, 6, 6))
        self.rightmat = cp.zeros((depth, 6))
        self.niter = 0

    def on_data(self, ispan):

        stream = bf.device.get_stream()
        with cp.cuda.ExternalStream(stream):
            idata = ispan.data[0].as_cupy()

            # Figure out what the G matrix should look like
            inds = self.niter * self.gulp_size + cp.arange(0, self.gulp_size)
            yinds, xinds, *_ = cp.unravel_index(inds, self.imshape)
            ones = cp.ones_like(xinds)
            coords = cp.column_stack([ones, xinds, yinds,
                                 xinds**2, yinds**2, xinds*yinds])

            # Accumulate dot-product
            # [left] GTG = (ng,nd)(ng,6)(ng,6)->(nd,6,6)
            # [right] GTd = (ng,6)(ng,nd)->(nd,6)
            mask = ~cp.isnan(idata)
            self.leftmat += cp.einsum('jl,ji,jk->lik', mask, coords, coords)
            self.rightmat += cp.nansum(cp.einsum('jk,ji->ijk', coords, idata),
                                       axis=1)
            self.niter += 1

    def on_skip(self, *args, **kwargs):
        raise NotImplementedError


class ApplyModelBlock(bfp.TransformBlock):
    """Applies the model we built with the previous block."""

    imshape = None

    def __init__(self, iring, models, *args, **kwargs):
        super().__init__(iring, *args, **kwargs)
        self.models = cp.asarray(models.T)

        self.step = 0
        self.ntrend = self.models.shape[0]

    def on_sequence(self, iseq):
        ohdr = deepcopy(iseq.header)
        _, ny, nx = literal_eval(ohdr['inshape'])
        self.imshape = (ny, nx)

        blockslogger.debug('Started ApplyModelBlock')

        return ohdr

    def on_data(self, ispan, ospan):
        out_nframe = ispan.nframe

        stream = bf.device.get_stream()
        with cp.cuda.ExternalStream(stream):
            idata = ispan.data.as_cupy()
            odata = ospan.data.as_cupy()

            # Do some stuff with idata
            gulp_size = cp.size(idata[0], 0)
            r_start = self.step * gulp_size
            r_end = (self.step+1) * gulp_size
            yc, xc, *_ = cp.unravel_index(cp.arange(r_start, r_end), self.imshape)

            # d(7800,3) m(3,20) -> c(7800,20)
            ones = cp.ones(len(xc)).astype(np.float64)
            coords = cp.column_stack([ones, xc, yc, xc**2, yc**2, xc*yc])[:, :self.ntrend]
            corr = cp.dot(coords, self.models)
            corr = cp.expand_dims(corr, axis=0)

            self.step += 1

            odata[...] = idata
            odata -= corr
            ospan.data[...] = bf.ndarray(odata)

        return out_nframe

class CalcRatesBlock(bfp.TransformBlock):
    """Calculate the rates for each pixel."""

    def __init__(self, iring, taxis, deg=1, *args, **kwargs):
        super().__init__(iring, *args, **kwargs)
        self.taxis = cp.asarray(taxis)
        self.deg = deg

    def on_sequence(self, iseq):
        ohdr = deepcopy(iseq.header)

        ohdr['_tensor']['shape'][2] = 1

        blockslogger.debug('Started CalcRatesBlock')

        return ohdr

    def on_data(self, ispan, ospan):
        in_nframe = ispan.nframe
        out_nframe = in_nframe

        stream = bf.device.get_stream()
        with cp.cuda.ExternalStream(stream):
            idata = ispan.data.as_cupy()
            odata = ospan.data.as_cupy()

            fits = cp.polyfit(self.taxis, idata[0].T, 1)

            odata[...] = fits[0].reshape((1, -1, 1))
            ospan.data[...] = bf.ndarray(odata)

        return out_nframe


class WriteTempBlock(bfp.SinkBlock):
    """Write out a temparary file, since we can't parallel write to HDF5."""

    # Initialize things we need later
    gulp_size = None
    dtype = None
    outshape = None
    imshape = None
    mmap = None

    def __init__(self, iring, outfile, *args, **kwargs):
        super().__init__(iring, *args, **kwargs)

        # name?
        self.file = outfile

        # Set up accumulation
        self.niter = 0

    def on_sequence(self, iseq):

        # Grab useful things from header
        hdr = iseq.header
        _, self.gulp_size, depth = hdr['_tensor']['shape']
        self.dtype = string2numpy(hdr['_tensor']['dtype'])

        # Grab useful things from file
        inshape = literal_eval(hdr['inshape'])
        self.outshape = (depth, inshape[1], inshape[2])
        self.imshape = (inshape[1], inshape[2])

        self.mmap = np.memmap(self.file, dtype=self.dtype, mode='w+',
                              shape=self.outshape)

        blockslogger.debug('Started WriteTempBlock to file %s', self.file)
        blockslogger.debug('Writing shape=%s, dtype=%s', self.outshape, self.dtype)

    def on_data(self, ispan):

        r_start = self.niter * self.gulp_size
        r_end = (self.niter+1) * self.gulp_size
        yc, xc, *_ = np.unravel_index(np.arange(r_start, r_end), self.imshape)

        idata = ispan.data[0].T
        self.mmap[:, yc, xc] = idata

        self.niter += 1

    def on_skip(self, *args, **kwargs):
        raise NotImplementedError
