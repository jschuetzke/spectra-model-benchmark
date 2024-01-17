#!/usr/bin/env python
# coding: utf-8

import json
import numpy as np
from tqdm import tqdm
from scipy.ndimage import gaussian_filter1d

# GLOBAL PARAMETERS

signal_length =  5000
shift_range = 100 # for now, we shift all peaks independently
variation_range = 0.1 # +/- of absolute height for each peak
kernel_range = (2.5, 30) # min and max for guassian kernel sizes

# Target Fingerprint

positions_phase = [4290, 4700]
heights_phase = [0.744, 1.0]

# Alternative Fingerprints

min_peaks = 2
max_peaks = 10

n_train = 5000 # times 2, positive + negative

rng = np.random.default_rng(2023)

def vary_peaks(position_list, height_list):
    # since we set a boundary parameter, positions should never exceed range
    # still applying clip to be sure
    new_positions = np.clip(np.array([
        rng.integers(f-shift_range, f+shift_range) for f in position_list
        ]), 0, 4999)
    new_heights = np.clip(np.array([
        rng.uniform(f-variation_range, f+variation_range) for f in height_list
        ]), 0, 2)
    return new_positions, new_heights



def main():   
    # initialize arrays to fill
    x = np.zeros([(n_train*2), signal_length])
    y = np.zeros(n_train*2)

    # generate "positive" samples
    for i in tqdm(range(n_train)):
        scan = np.zeros(signal_length)
        # apply shifts
        new_pos, new_hi = vary_peaks(positions_phase, heights_phase)
        scan[new_pos] = new_hi
        # convolve with gaussian kernel
        scan = gaussian_filter1d(scan, rng.uniform(*kernel_range), 
                                    mode='constant')
        x[i] = scan

    # generate "negative" samples
    # three cases - pattern not contained, pattern as impurity, pattern as major

    cases = rng.choice(3, size=n_train, p=[0.2,0.4,0.4])

    for i in tqdm(range(n_train)):
        scan = np.zeros(signal_length)
        new_pos = rng.integers(
            100, 
            signal_length-100,
            rng.integers(min_peaks, max_peaks, endpoint=True)
            )
        new_his = rng.uniform(0.01, 1., new_pos.size, endpoint=True)
        new_his = np.round(new_his / np.max(new_his), 3)
        
        scan[new_pos] = new_his

        if cases[i] == 0:
            pass
        elif cases[i] == 1:
            new_pos, new_hi = vary_peaks(positions_phase, heights_phase)
            factor = rng.uniform(0., 0.7)
            new_hi *= factor
            scan[new_pos] = new_hi
        else:
            new_pos, new_hi = vary_peaks(positions_phase, heights_phase)
            factor = rng.uniform(0.03, 0.2)
            scan *= factor
            scan[new_pos] = new_hi

        # convolve with gaussian kernel
        scan = gaussian_filter1d(scan, rng.uniform(*kernel_range), 
                                    mode='constant')
        x[i+n_train] = scan
    y[:n_train] = 1.
    y[n_train:] = 0.

    indices = np.arange(n_train*2)
    rng.shuffle(indices)
    train_val_split = int(n_train*0.8*2)
    x_train = x[indices[:train_val_split]]
    x_val = x[indices[train_val_split:]]
    y_train = y[indices[:train_val_split]]
    y_val = y[indices[train_val_split:]]

    np.save('./x_train.npy', x_train)
    np.save('./y_train.npy', y_train)
    np.save('./x_val.npy', x_val)
    np.save('./y_val.npy', y_val)

if __name__ == '__main__':
    main()
