#!/bin/bash
# noise fedsgd updates with dpsgd
python3 measure_f1.py -f sample_datasets/32_0.txt -e 1 -bs 32 -s 0.0 -o fedsgd/ --cutoff
python3 measure_f1.py -f sample_datasets/32_0.txt -e 1 -bs 32 -s 0.1 -dp 1 -o fedsgd/ --cutoff
python3 measure_f1.py -f sample_datasets/32_0.txt -e 1 -bs 32 -s 0.01 -dp 1 -o fedsgd/ --cutoff
python3 measure_f1.py -f sample_datasets/32_0.txt -e 1 -bs 32 -s 0.001 -dp 1 -o fedsgd/ --cutoff
python3 measure_f1.py -f sample_datasets/32_0.txt -e 1 -bs 32 -s 0.0001 -dp 1 -o fedsgd/ --cutoff

python3 measure_f1.py -f sample_datasets/64_0.txt -e 1 -bs 64 -s 0.0 -o fedsgd/ --cutoff
python3 measure_f1.py -f sample_datasets/64_0.txt -e 1 -bs 64 -s 0.1 -dp 1 -o fedsgd/ --cutoff
python3 measure_f1.py -f sample_datasets/64_0.txt -e 1 -bs 64 -s 0.01 -dp 1 -o fedsgd/ --cutoff
python3 measure_f1.py -f sample_datasets/64_0.txt -e 1 -bs 64 -s 0.001 -dp 1 -o fedsgd/ --cutoff
python3 measure_f1.py -f sample_datasets/64_0.txt -e 1 -bs 64 -s 0.0001 -dp 1 -o fedsgd/ --cutoff

python3 measure_f1.py -f sample_datasets/128_0.txt -e 1 -bs 128 -s 0.0 -o fedsgd/ --cutoff
python3 measure_f1.py -f sample_datasets/128_0.txt -e 1 -bs 128 -s 0.1 -dp 1 -o fedsgd/ --cutoff
python3 measure_f1.py -f sample_datasets/128_0.txt -e 1 -bs 128 -s 0.01 -dp 1 -o fedsgd/ --cutoff
python3 measure_f1.py -f sample_datasets/128_0.txt -e 1 -bs 128 -s 0.001 -dp 1 -o fedsgd/ --cutoff
python3 measure_f1.py -f sample_datasets/128_0.txt -e 1 -bs 128 -s 0.0001 -dp 1 -o fedsgd/ --cutoff

python3 measure_f1.py -f sample_datasets/256_0.txt -e 1 -bs 256 -s 0.0 -o fedsgd/ --cutoff
python3 measure_f1.py -f sample_datasets/256_0.txt -e 1 -bs 256 -s 0.1 -dp 1 -o fedsgd/ --cutoff
python3 measure_f1.py -f sample_datasets/256_0.txt -e 1 -bs 256 -s 0.01 -dp 1 -o fedsgd/ --cutoff
python3 measure_f1.py -f sample_datasets/256_0.txt -e 1 -bs 256 -s 0.001 -dp 1 -o fedsgd/ --cutoff
python3 measure_f1.py -f sample_datasets/256_0.txt -e 1 -bs 256 -s 0.0001 -dp 1 -o fedsgd/ --cutoff

# noise fedsgd the final parameters
python3 measure_f1.py -f sample_datasets/32_0.txt -e 1 -bs 32 -s 0.0 -o fedsgd/ --cutoff
python3 measure_f1.py -f sample_datasets/32_0.txt -e 1 -bs 32 -s 0.1 -dp 2 -o fedsgd/ --cutoff
python3 measure_f1.py -f sample_datasets/32_0.txt -e 1 -bs 32 -s 0.01 -dp 2 -o fedsgd/ --cutoff
python3 measure_f1.py -f sample_datasets/32_0.txt -e 1 -bs 32 -s 0.001 -dp 2 -o fedsgd/ --cutoff
python3 measure_f1.py -f sample_datasets/32_0.txt -e 1 -bs 32 -s 0.0001 -dp 2 -o fedsgd/ --cutoff

python3 measure_f1.py -f sample_datasets/64_0.txt -e 1 -bs 64 -s 0.0 -o fedsgd/ --cutoff
python3 measure_f1.py -f sample_datasets/64_0.txt -e 1 -bs 64 -s 0.1 -dp 2 -o fedsgd/ --cutoff
python3 measure_f1.py -f sample_datasets/64_0.txt -e 1 -bs 64 -s 0.01 -dp 2 -o fedsgd/ --cutoff
python3 measure_f1.py -f sample_datasets/64_0.txt -e 1 -bs 64 -s 0.001 -dp 2 -o fedsgd/ --cutoff
python3 measure_f1.py -f sample_datasets/64_0.txt -e 1 -bs 64 -s 0.0001 -dp 2 -o fedsgd/ --cutoff

python3 measure_f1.py -f sample_datasets/128_0.txt -e 1 -bs 128 -s 0.0 -o fedsgd/ --cutoff
python3 measure_f1.py -f sample_datasets/128_0.txt -e 1 -bs 128 -s 0.1 -dp 2 -o fedsgd/ --cutoff
python3 measure_f1.py -f sample_datasets/128_0.txt -e 1 -bs 128 -s 0.01 -dp 2 -o fedsgd/ --cutoff
python3 measure_f1.py -f sample_datasets/128_0.txt -e 1 -bs 128 -s 0.001 -dp 2 -o fedsgd/ --cutoff
python3 measure_f1.py -f sample_datasets/128_0.txt -e 1 -bs 128 -s 0.0001 -dp 2 -o fedsgd/ --cutoff

python3 measure_f1.py -f sample_datasets/256_0.txt -e 1 -bs 256 -s 0.0 -o fedsgd/
python3 measure_f1.py -f sample_datasets/256_0.txt -e 1 -bs 256 -s 0.1 -dp 2 -o fedsgd/
python3 measure_f1.py -f sample_datasets/256_0.txt -e 1 -bs 256 -s 0.01 -dp 2 -o fedsgd/
python3 measure_f1.py -f sample_datasets/256_0.txt -e 1 -bs 256 -s 0.001 -dp 2 -o fedsgd/
python3 measure_f1.py -f sample_datasets/256_0.txt -e 1 -bs 256 -s 0.0001 -dp 2 -o fedsgd/
