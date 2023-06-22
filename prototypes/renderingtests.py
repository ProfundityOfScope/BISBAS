import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import transforms
import h5py
from time import time
from datetime import datetime

def makeg(datepairs):
    
    # Conver to numbers, find differences
    dates = np.sort(np.unique(datepairs))
    t0 = datetime.strptime(dates[0], '%Y%m%d')
    dr = np.array([(datetime.strptime(d, '%Y%m%d')-t0).days for d in dates])
    diffdates = np.roll(dr, -1)[:-1] - dr[:-1]

    # Convert datestrs to indices
    indmap = {dates[i]: i for i in range(len(dates))}
    indpairs = np.vectorize(lambda x: indmap[x])(datepairs)
    idarr = np.arange(1, len(dates))

    # Numpifies the creation of of the boolean arrays
    arr_less = indpairs[:, 0][:, None] < idarr
    arr_greq = indpairs[:, 1][:, None] >= idarr
    bool_arr = np.logical_and(arr_less, arr_greq)

    G = bool_arr * diffdates
    if not np.linalg.matrix_rank(G) == len(dates)-1:
        raise ValueError('G is of incorrect order')
    
    return G, dr

tgt = '/Users/bruzewskis/Documents/Projects/BISBAS/ifgs_t08_p08.h5'
if True:
    with h5py.File(tgt, 'r') as fo:
        G, dates = makeg(fo['date'][:].astype(str))
        
        print(fo['coherence'].shape)
        coherence = fo['coherence'][:, 250:350,250:350]
        phase = fo['unwrapPhase'][:, 250:350, 250:350]
        phase[coherence < 0.2] = np.nan
        
        start = time()
        mx, my = 50,50
        med_ref = np.nanmedian(phase[:, my-5:my+5, mx-5:mx+5], axis=(1, 2))
        phase -= med_ref[:, None, None]
        print('Median Subtract:', time()-start)
        
        # Generate timeseries
        M = ~np.isnan(phase).reshape(len(phase),-1).T
        D = phase.reshape(len(phase),-1).T
        # (1,nd,ni) @ ( (np,ni,1)*(1,ni,nd) )
        # (1,nd,ni) @ (np,ni,nd)
        # (np, nd, nd)
        A = np.matmul(G.T[None, :, :], M[:, :, None] * G[None, :, :])
        B = np.nansum(G.T[:, :, None] * (M*D).T[None, :, :], axis=1).T
        
        # Mask out low-rank values
        start = time()
        nd = len(dates)
        lowrank = np.linalg.matrix_rank(A) !=  nd-1
        ranktime = time()-start
        print('Rank Check:', ranktime)
        
        start = time()
        lowrank = np.linalg.det(A)==0
        dettime = time()-start
        print('Det Mask:', dettime)
        print('Speedup:', ranktime/dettime)
        
        
        
        A[lowrank] = np.eye(nd-1)
        B[lowrank] = np.full(nd-1, np.nan)
        Mo = np.linalg.solve(A, B)

        # Turn it into a cumulative timeseries
        start = time()
        datediffs = (dates - np.roll(dates, 1))[1:]
        changes = datediffs[None, :] * Mo
        ts = np.zeros((changes.shape[0], len(dates)))
        ts[:, 1:] = np.cumsum(changes, axis=1)
        ts = ts.T.reshape((nd, phase.shape[1], phase.shape[2]))
        print('Cumsum to TS:', time()-start)
        
        # plt.subplot2grid((1,2),(0,1))
        # plt.imshow(ts[-1], interpolation='nearest', cmap='Spectral_r')