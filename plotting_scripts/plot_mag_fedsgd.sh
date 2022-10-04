#!/bin/bash

# plot for dpsgd
python3 plot_magnitude.py results/magnitude/mag_exp_256_0.txt_1_256_0.0001_0.001_DPSGD.results results/magnitude/mag_exp_256_0.txt_1_256_0.001_0.001_DPSGD.results results/magnitude/mag_exp_256_0.txt_1_256_0.01_0.001_DPSGD.results results/magnitude/mag_exp_256_0.txt_1_256_0.1_0.001_DPSGD.results -t "nk = 256, B = 256 (DPSGD)" --group-by "noise"

# plot for single noise
python3 plot_magnitude.py results/magnitude/mag_exp_256_0.txt_1_256_0.0001_0.001_SINGLENOISE.results results/magnitude/mag_exp_256_0.txt_1_256_0.001_0.001_SINGLENOISE.results results/magnitude/mag_exp_256_0.txt_1_256_0.01_0.001_SINGLENOISE.results results/magnitude/mag_exp_256_0.txt_1_256_0.1_0.001_SINGLENOISE.results -t "nk = 256, B = 256 (SINGLENOISE)" --group-by "noise"
