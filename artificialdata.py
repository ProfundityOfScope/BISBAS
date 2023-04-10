#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  6 11:23:57 2023

@author: bruzewskis
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage


def generate_perlin_noise_2d(shape, res):
    def f(t):
        return 6*t**5 - 15*t**4 + 10*t**3

    delta = (res[0] / shape[0], res[1] / shape[1])
    d = (shape[0] // res[0], shape[1] // res[1])
    grid = np.mgrid[0:res[0]:delta[0], 0:res[1]:delta[1]].transpose(1, 2, 0) % 1
    # Gradients
    angles = 2*np.pi*np.random.rand(res[0]+1, res[1]+1)
    gradients = np.dstack((np.cos(angles), np.sin(angles)))
    g00 = gradients[0:-1, 0:-1].repeat(d[0], 0).repeat(d[1], 1)
    g10 = gradients[1:, 0:-1].repeat(d[0], 0).repeat(d[1], 1)
    g01 = gradients[0:-1, 1:].repeat(d[0], 0).repeat(d[1], 1)
    g11 = gradients[1:, 1:].repeat(d[0], 0).repeat(d[1], 1)
    # Ramps
    n00 = np.sum(grid * g00, 2)
    n10 = np.sum(np.dstack((grid[:, :, 0]-1, grid[:, :, 1])) * g10, 2)
    n01 = np.sum(np.dstack((grid[:, :, 0], grid[:, :, 1]-1)) * g01, 2)
    n11 = np.sum(np.dstack((grid[:, :, 0]-1, grid[:, :, 1]-1)) * g11, 2)
    # Interpolation
    t = f(grid)
    n0 = n00*(1-t[:, :, 0]) + t[:, :, 0]*n10
    n1 = n01*(1-t[:, :, 0]) + t[:, :, 0]*n11
    return np.sqrt(2)*((1-t[:, :, 1])*n0 + t[:, :, 1]*n1)


def generate_fractal_noise_2d(shape, res, octaves=1, persistence=0.5):
    noise = np.zeros(shape)
    frequency = 1
    amplitude = 1
    for _ in range(octaves):
        noise += amplitude * generate_perlin_noise_2d(shape, (frequency*res[0],
                                                              frequency*res[1]))
        frequency *= 2
        amplitude *= persistence
    return noise

def generate_real_fake(shape, cutoff=0.25, angle=15, bad_frac=0.05,
                       edge_noise=False, seed=1337):

    np.random.seed(seed)

    # Make the ground
    result = 1 + generate_fractal_noise_2d(shape, (5, 5), 4)

    # rotate
    ground = ndimage.rotate(result, -angle, reshape=True, cval=np.nan)
    ground[ground < cutoff] = cutoff
    x = np.linspace(15.1, 15.2, ground.shape[1])
    y = np.linspace(32.5, 32.6, ground.shape[0])

    # Random bad data
    px = np.product(ground.shape)
    yind, xind = np.unravel_index(np.random.randint(0, px, int(bad_frac*px)),
                                  ground.shape)
    ground[yind, xind] = np.nan

    # mess up the edges a bit
    if edge_noise:
        pixels = np.empty((2, 0))
        badrows = 100
        scale = (1*shape[0])**(1/badrows)
        inds = np.arange(0, shape[0])
        for i in range(badrows):
            num = 1*shape[0] / scale**i
            left = np.random.choice(inds, int(num), replace=False)
            right = np.random.choice(inds, int(num), replace=False)
            left_pairs = np.array([left, np.full_like(left, i)])
            right_pairs = np.array([right, np.full_like(right, shape[0]-(i+1))])
            pixels = np.column_stack([pixels, left_pairs, right_pairs])

        pixels -= np.array(shape)[:, None]//2
        rpx = np.zeros_like(pixels)
        rpx[0] = np.sin(np.deg2rad(angle)) * pixels[1] + np.cos(np.deg2rad(angle)) * pixels[0]
        rpx[1] = np.cos(np.deg2rad(angle)) * pixels[1] - np.sin(np.deg2rad(angle)) * pixels[0]
        rpx += np.array(ground.shape)[:, None]//2
        rpx = rpx.astype(int)
        ground[rpx[0],rpx[1]] = np.nan

    plt.figure(figsize=(6, 5))
    plt.pcolormesh(x, y, ground)
    plt.colorbar()
    plt.show()

if __name__=='__main__':

    generate_real_fake((1000,1000),
                       cutoff=0.25,
                       angle=15,
                       bad_frac=0.05,
                       edge_noise=1)
