#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 20 15:12:14 2023

@author: bruzewskis
"""

from astropy.time import Time
import numpy as np
import matplotlib.pyplot as plt

def fittime(outputs):
    
    t0 = Time(outputs[0][:19]).mjd
    time = np.array([ Time(s[:19]).mjd-t0 for s in outputs ])
    line = np.array([ int(s[46:]) for s in outputs ])
    
    pfit = np.polyfit(time, line, 1)
    t_complete = (4231 - pfit[1])/pfit[0]
    pfunc = np.poly1d(pfit)
    xx = np.linspace(0, t_complete)
    yy = pfunc(xx)
    
    time_left = (t0 + t_complete - Time.now().mjd)* 24*60
    print(f'Total runtime likely {t_complete*24:.2f} hours')
    print(f'Likely completion in {time_left:.2f} minutes')
    
    plt.scatter(time*86400, line)
    plt.plot(xx*86400, yy, ls='--')
    plt.axhline(4231, c='k', ls=':')
    
if __name__=='__main__':
    
    outputs = ['2023-07-24 01:16:36 [DEBUG   ] Wrote out line 0',
               '2023-07-24 01:17:26 [DEBUG   ] Wrote out line 226',
               '2023-07-24 01:21:34 [DEBUG   ] Wrote out line 1248',
               '2023-07-24 01:24:38 [DEBUG   ] Wrote out line 1925',
               '2023-07-24 01:30:18 [DEBUG   ] Wrote out line 3311',
               '2023-07-24 01:32:32 [DEBUG   ] Wrote out line 3879']
    fittime(outputs)
    