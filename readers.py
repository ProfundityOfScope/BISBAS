#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Temp text up here

Created on Thu Jul 14 12:22:08 2022
@author: bruzewskis
"""

import numpy as np
import os
import sys
import time
import logging
from scipy.io import netcdf

__version__ = 0.1

readerslogger = logging.getLogger('__main__')

class DataStack(object):
    '''
    This class abstracts away file handling. Given a directory of interferograms
    it will read them all the a memory map, then selectively extract and format
    requested data for you
    '''
    
    def __init__(self, file_objs):
        self.file_objs = file_objs

        # Figure out image properties
        self.imshape = self.file_objs[0].variables['z'].shape #unsafe if empty
        self.imsize = np.product(self.imshape)
        readerslogger.debug(f'Found {len(self.file_objs)} files with shape {self.imshape}')

        # Establish grid convention
        dims = self.file_objs[0].dimensions
        if 'lon' in dims and 'lat' in dims:
            readerslogger.debug('Lon/Lat coordinates inferred from data')
            self.xgrd = 'lon'
            self.ygrd = 'lat'
        elif 'x' in dims and 'y' in dims:
            readerslogger.debug('X/Y coordinates inferred from data')
            self.xgrid = 'x'
            self.ygrid = 'y'
        else:
            raise KeyError('File gridding does not follow any known standard')
        
        # Figure out object properties
        self.shape = (len(self.file_objs), self.imsize, 3)
        self.size = np.product(self.shape)
        readerslogger.debug(f'Images will be treated as a {self.shape} object')
        
        # Save some data for later use
        self._xarr = self.file_objs[0].variables[self.xgrd][:].copy()
        self._yarr = self.file_objs[0].variables[self.ygrd][:].copy()
    
    @classmethod
    def read(cls, files):

        file_objs = []
        for file in files:
            file_obj = netcdf.netcdf_file(file) #mmap
            file_objs.append(file_obj)
                    
        return cls(file_objs)
    
    @classmethod
    def empty(cls, directory, num_dates, xcoords, ycoords):
        # Loop through and create num_dates empty files in directory
        # Image shape can be inferred from xcoords/ycoords
        # Then create a DataStack to wrap them
        pass
    
    def __getitem__(self, key):
        '''
        This is a specialized getitem, with the sole purpose of extracting
        strips from the data. It will yell at you if you try to extract
        anything other than strips
        '''
                    
        # Evaluate key to be safe
        if isinstance(key, np.ndarray):
            # All good, we like arrays
            newkey = key
        elif isinstance(key, slice):
            # Slices are a pain with how we index, convert
            newkey = np.arange(self.imsize)[key]
        else:
            # Yell if we have to
            raise IndexError('Give me a slice or an array')
            
        data = np.empty((np.size(self, 0), len(newkey), np.size(self,2)))
        for i in range(len(self.file_objs)):
            file = self.file_objs[i]
            x = file.variables[self.xgrd]
            y = file.variables[self.ygrd]
            z = file.variables['z']
    
            xt = x[newkey%x.shape[0]].copy()
            yt = y[newkey//x.shape[0]].copy()
    
            zind = np.unravel_index(newkey, z.shape)
            zt = z[zind].copy()
    
            data[i,:,0] = xt
            data[i,:,1] = yt
            data[i,:,2] = zt
        return data
        
    def __setitem__(self, key, value):
        # Use this to dump data out to files
        pass
    
    def data_near(self, x0, y0, chunk_size):

        # Extract data using coords method
        coords_rav = self.coords_near(x0, y0, chunk_size)
        subdata = self.__getitem__(coords_rav)

        return subdata

    def coords_near(self, x0, y0, chunk_size):
        
        # This is how we would deal with a non-uniform spacing
        xp = np.interp(x0, self._xarr, np.arange(len(self._xarr)))
        yp = np.interp(y0, self._yarr, np.arange(len(self._yarr)))
        readerslogger.debug(f'Requested reference around pixel x:{xp:.0f}, y:{yp:.0f}')

        outofbounds = any([ xp <= chunk_size,
                            xp >= len(self._xarr)-chunk_size,
                            yp <= chunk_size,
                            yp >= len(self._yarr)-chunk_size ])
        if outofbounds:
            readerslogger.error(f'This position is too close to the edge of image')
            raise ValueError(f'This position is too close to the edge of image')

        # How far to look around
        chunk_rad = chunk_size/2
        
        # Find corners
        xmin = np.ceil( xp - chunk_rad ).astype(int)
        ymin = np.ceil( yp - chunk_rad ).astype(int)
        xmax = xmin + chunk_size
        ymax = ymin + chunk_size
        
        # Extracts data inside those corners
        xind = np.arange(xmin, xmax)
        yind = np.arange(ymin, ymax)
        coords = np.array(np.meshgrid(xind, yind)).reshape(2,-1)
        coords_rav = np.ravel_multi_index( coords, self.imshape )

        return coords_rav

    def find_best_chunk_size(self, x0, y0, num):
        
        min_size = np.ceil(np.sqrt(num)).astype(int)
        for chunk_size in np.arange(min_size,100):
            subdata = self.data_near(x0, y0, chunk_size)
            
            # Check to see if we have enough good pixels
            num_good_per_date = np.sum(~np.isnan(subdata[:,:,2]), axis=1)
            if np.all(num_good_per_date >= num):
                readerslogger.debug(f'Chunk size of {chunk_size} has enough good pixels')
                return chunk_size
        else:
            raise ValueError('Not enough good pixels near this point')

    
    def __del__(self):
        for file in self.file_objs:
            file.close()

def read_baselines(fname):
    '''
    This code taken from original code
    read the baseline table and store in date-sorted order
    '''

    readerslogger.info('Reading some shit')
    strdat=np.genfromtxt(fname,str,usecols=0)
    numdat=np.genfromtxt(fname,usecols=(1,2,4))
    sortorder=np.argsort(numdat[:,1])

    ids=strdat[sortorder]
    jdates=np.array([str(int(np.floor(i))) for i in numdat[sortorder,0]])
    dates=numdat[sortorder,1]
    bperp=numdat[sortorder,2]
    
    return ids,jdates,dates,bperp

def get_igram_ids(sat, fname, ids):
    '''
    This code taken from original code, modified
    read intf.in and convert to a list of tuples of IDs
    '''

    sat_map = {'ALOS': slice(13,18), 'ALOS2': slice(12,17), 'S1': slice(0,18)}
    if not sat in sat_map:
        readerslogger.error(f'Error: satellite {sat} not yet implemented.')
        sys.exit(1)

    igrams=np.genfromtxt(fname,dtype=str)
    igram_ids=[]
    sat_slice = sat_map[sat]
    igram_ids = np.zeros((len(igrams),2), dtype=int)
    for i in range(len(igrams)):
        igram = igrams[i]
        igramsplit=igram.split(':')
        strid0=igramsplit[0][sat_slice]
        strid1=igramsplit[1][sat_slice]
        id0=np.where(ids==strid0)[0][0]
        id1=np.where(ids==strid1)[0][0]
        igram_ids[i] = np.array([id0, id1])
    return igram_ids

if __name__=='__main__':
    tgt = '/Users/bruzewskis/Documents/Projects/BISBAS/testing/intf/'
    files = []
    for r,d,f in os.walk(tgt):
        for file in f:
            if file.endswith('.grd'):
                path = os.path.join(r, file)
                print(path)
                files.append(path)
    
    logFormat = logging.Formatter('%(asctime)s [%(levelname)-8s] %(message)s', 
                                  datefmt='%Y-%m-%d %H:%M:%S')
    logFormat.converter = time.gmtime
    logHandler = logging.StreamHandler(sys.stdout)
    readerslogger.setLevel(logging.DEBUG)
    readerslogger.info('Testing readers file')

    b = DataStack.read(files)
    print(b.shape)

    n = 10
    xr = np.random.normal(255.3, 0.1, n)
    yr = np.random.normal(36.6, 0.1, n)

    rads = np.zeros(n)
    for i in range(n):
        rads[i] = b.find_best_chunk_size(xr[i], yr[i], 10)
    print(rads)
    print(max(rads))      

    test = b.coords_near(xr[i], yr[i], 10)
    print(test.shape)   
            
    