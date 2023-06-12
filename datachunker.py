import h5py
import numpy as np
import os

with h5py.File('ifgramStack.h5', 'r') as fo:

	t0,y0,x0 = fo['coherence'].shape
	attrs = dict(fo.attrs)

	for n in range(2,6):
		for m in range(2,6):

			tn = int(t0/2**n)
			yn = int(y0/2**m)
			xn = int(x0/2**m)
			name = f'ifgs_t{2**n:02d}_p{2**m:02d}.h5'

			with h5py.File(name, 'w') as fo2:

				for a in attrs:
					fo2[a] = attrs[a]

				fo2['coherence'] = fo['coherence'][:tn, :yn, :xn]
				fo2['date'] = fo['date'][:tn]
			print(tn, yn, xn, name)