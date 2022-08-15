#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 14 12:22:08 2022

@author: bruzewskis
"""

import numpy as np
import os
from scipy.io import netcdf

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
        
        # Figure out object properties
        self.shape = (len(self.file_objs), self.imsize, 3)
        self.size = np.product(self.shape)
        
        # Save some data for later use
        self._xarr = self.file_objs[0].variables['x'][:].copy()
        self._yarr = self.file_objs[0].variables['y'][:].copy()
    
    @classmethod
    def read(cls, directory):
        files = []
        for r,d,f in os.walk(directory):
            for file in f:
                if file.endswith('.grd'):
                    file_obj = netcdf.netcdf_file(os.path.join(r,file)) #mmap
                    files.append(file_obj)
                    
        return cls(files)
    
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
            x = file.variables['x']
            y = file.variables['y']
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
    
    def data_near(self, x0, y0, num):
        
        # This is how we would deal with a non-uniform spacing
        xp = np.interp(x0, self._xarr, np.arange(len(self._xarr)))
        yp = np.interp(y0, self._yarr, np.arange(len(self._yarr)))
        
        min_size = np.ceil(np.sqrt(num)).astype(int)
        for moore_size in np.arange(min_size,10):
            moore_rad = moore_size/2
            
            # Find corners
            xmin = np.ceil( xp - moore_rad ).astype(int)
            ymin = np.ceil( yp - moore_rad ).astype(int)
            xmax = xmin + moore_size
            ymax = ymin + moore_size
            
            # Extracts data inside those corners
            xind = np.arange(xmin, xmax)
            yind = np.arange(ymin, ymax)
            coords = np.array(np.meshgrid(xind, yind)).reshape(2,-1)
            subdata = self.__getitem__(np.ravel_multi_index( coords, self.imshape ))
            
            # Check to see if we have enough good pixels
            num_good_per_date = np.sum(~np.isnan(subdata[:,:,2]), axis=1)
            if np.all(num_good_per_date >= num):
                return subdata
        else:
            raise ValueError('Not enough good pixels near this point')
    
    def __del__(self):
        for file in self.file_objs:
            file.close()

if __name__=='__main__':
    tgt = '/Users/bruzewskis/Documents/Projects/bifrost_isbas/isbas/test/intf/'
    
    b = DataStack.read(tgt)
    data = b[:]
    print(b.shape)
            
            
    