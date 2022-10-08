#!/bin/bash
# varying nk, B = 32, no noise
python3 measure_f1.py -f sample_datasets/16_0.txt -e 1000 -bs 16 -s 0 -o f1_nonoise/
python3 measure_f1.py -f sample_datasets/32_0.txt -e 1000 -bs 32 -s 0 -o f1_nonoise/
python3 measure_f1.py -f sample_datasets/64_0.txt -e 1000 -bs 32 -s 0 -o f1_nonoise/
python3 measure_f1.py -f sample_datasets/128_0.txt -e 1000 -bs 32 -s 0 -o f1_nonoise/
python3 measure_f1.py -f sample_datasets/256_0.txt -e 1000 -bs 32 -s 0 -o f1_nonoise/

# varying nk, B = nk, no noise
python3 measure_f1.py -f sample_datasets/16_0.txt -e 1000 -bs 16 -s 0 -o f1_nonoise/
python3 measure_f1.py -f sample_datasets/32_0.txt -e 1000 -bs 32 -s 0 -o f1_nonoise/
python3 measure_f1.py -f sample_datasets/64_0.txt -e 1000 -bs 64 -s 0 -o f1_nonoise/
python3 measure_f1.py -f sample_datasets/128_0.txt -e 1000 -bs 128 -s 0 -o f1_nonoise/
python3 measure_f1.py -f sample_datasets/256_0.txt -e 1000 -bs 256 -s 0 -o f1_nonoise/
