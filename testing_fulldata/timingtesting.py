#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 10 12:14:51 2023

@author: bruzewskis
"""

import os
import time
import numpy as np
import matplotlib.pyplot as plt

def find_factors(n):
    out = []
    for i in range(1,int(np.sqrt(n))):
        if n%i==0:
            out.append(i)
            out.append(n//i)
    return sorted(out)


if __name__=='__main__':
    
    # Figure out dimensions and factors
    lx, ly, lt = 3250, 3900, 20
    total_size = int(lx*ly)
    print('Total Pixels:', total_size)
    factors = find_factors(total_size)
    
    steps = int((len(factors)-10)/20)
    rfactors = factors[7:-5:steps]

    with open('timings.csv', 'w') as fp:
        fp.write('size, time\n')
        fp.flush()
        for factor in rfactors:
            size = factor * lt
            
            start = time.time()
            os.system(f'python ../bisbas.py -g {factor}')
            diff = time.time() - start
            
            fp.write(f'{size:d}, {diff:.5e}\n')
            fp.flush()
            