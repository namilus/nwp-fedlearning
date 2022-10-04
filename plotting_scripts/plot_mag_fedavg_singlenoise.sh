#!/bin/bash

# plot for sigma = 0.0001
python3 plot_magnitude.py results/magnitude/mag_exp_256_0.txt_1000_32_0.0001_0.001_SINGLENOISE.results results/magnitude/mag_exp_256_0.txt_100_32_0.0001_0.001_SINGLENOISE.results results/magnitude/mag_exp_256_0.txt_10_32_0.0001_0.001_SINGLENOISE.results results/magnitude/mag_exp_256_0.txt_1_32_0.0001_0.001_SINGLENOISE.results -t "nk = 256, B = 32, sigma = 0.0001 (SINGLENOISE)" --group-by "epochs"


# plot for sigma = 0.001
python3 plot_magnitude.py results/magnitude/mag_exp_256_0.txt_1000_32_0.001_0.001_SINGLENOISE.results results/magnitude/mag_exp_256_0.txt_100_32_0.001_0.001_SINGLENOISE.results results/magnitude/mag_exp_256_0.txt_10_32_0.001_0.001_SINGLENOISE.results results/magnitude/mag_exp_256_0.txt_1_32_0.001_0.001_SINGLENOISE.results -t "nk = 256, B = 32, sigma = 0.001 (SINGLENOISE)" --group-by "epochs"

# plot for sigma = 0.01
python3 plot_magnitude.py results/magnitude/mag_exp_256_0.txt_1000_32_0.01_0.001_SINGLENOISE.results results/magnitude/mag_exp_256_0.txt_100_32_0.01_0.001_SINGLENOISE.results results/magnitude/mag_exp_256_0.txt_10_32_0.01_0.001_SINGLENOISE.results results/magnitude/mag_exp_256_0.txt_1_32_0.01_0.001_SINGLENOISE.results -t "nk = 256, B = 32, sigma = 0.01 (SINGLENOISE)" --group-by "epochs"

# plot for sigma = 0.1
python3 plot_magnitude.py results/magnitude/mag_exp_256_0.txt_1000_32_0.1_0.001_SINGLENOISE.results results/magnitude/mag_exp_256_0.txt_100_32_0.1_0.001_SINGLENOISE.results results/magnitude/mag_exp_256_0.txt_10_32_0.1_0.001_SINGLENOISE.results results/magnitude/mag_exp_256_0.txt_1_32_0.1_0.001_SINGLENOISE.results -t "nk = 256, B = 32, sigma = 0.1 (SINGLENOISE)" --group-by "epochs"
