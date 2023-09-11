import h5py
import numpy as np

foi = h5py.File('timeseries.h5', 'r')
foo = h5py.File('ts_dnsmp.h5', 'w')

for attr in foi.attrs:
	foo.attrs[attr] = foi.attrs[attr]

ind = 100
foo['timeseries_image'] = foi['timeseries'][ind]
foo['timeseries_chunk'] = foi['timeseries'][:, 2000:2100, 2000:2100]
foo['timeseries_dnsmp'] = foi['timeseries'][:, ::10, ::10]
foo['timeseries_ramp_image'] = foi['detrended'][ind]
foo['timeseries_ramp_chunk'] = foi['detrended'][:, 2000:2100, 2000:2100]
foo['timeseries_ramp_dnsmp'] = foi['detrended'][:, ::10, ::10]
foo['velocity'] = foi['rates'][:]

foi.close()
foo.close()