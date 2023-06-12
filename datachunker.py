import h5py
import numpy as np
import os

with h5py.File('ifgramStack.h5', 'r') as fo:

	t0,y0,x0 = fo['coherence'].shape
	attrs = dict(fo.attrs)

	for n in range(1,7):
		for m in range(1,7):

			tn = int(t0/2**n)
			yn = int(y0/2**m)
			xn = int(x0/2**m)
			name = f'ifgs_t{2**n:02d}_p{2**m:02d}.h5'

			print(tn, yn, xn, name)