#!/bin/bash

# Fed AVG  f1 with epochs (DPSGD) 
python3 measure_f1.py -f sample_datasets/256_0.txt -e 1000 -bs 32 -s 0.0 -o fedavg/ --cutoff
python3 measure_f1.py -f sample_datasets/256_0.txt -e 1000 -bs 32 -s 0.1 -dp 1 -o fedavg/ --cutoff
python3 measure_f1.py -f sample_datasets/256_0.txt -e 1000 -bs 32 -s 0.01 -dp 1 -o fedavg/ --cutoff
python3 measure_f1.py -f sample_datasets/256_0.txt -e 1000 -bs 32 -s 0.001 -dp 1 -o fedavg/ --cutoff 
python3 measure_f1.py -f sample_datasets/256_0.txt -e 1000 -bs 32 -s 0.0001 -dp 1 -o fedavg/ --cutoff

# Fed AVG f1 after single noise addition at the end
# epochs = 1
python3 measure_f1.py -f sample_datasets/256_0.txt -e 1 -bs 32 -s 0.0 -o fedavg/ --cutoff 
python3 measure_f1.py -f sample_datasets/256_0.txt -e 1 -bs 32 -s 0.1 -dp 2 --measure-final-only -o fedavg/ --cutoff
python3 measure_f1.py -f sample_datasets/256_0.txt -e 1 -bs 32 -s 0.01 -dp 2 --measure-final-only -o fedavg/ --cutoff
python3 measure_f1.py -f sample_datasets/256_0.txt -e 1 -bs 32 -s 0.001 -dp 2 --measure-final-only -o fedavg/ --cutoff
python3 measure_f1.py -f sample_datasets/256_0.txt -e 1 -bs 32 -s 0.0001 -dp 2 --measure-final-only -o fedavg/ --cutoff

# epochs = 10
python3 measure_f1.py -f sample_datasets/256_0.txt -e 10 -bs 32 -s 0.0 -o fedavg/ --cutoff 
python3 measure_f1.py -f sample_datasets/256_0.txt -e 10 -bs 32 -s 0.1 -dp 2 --measure-final-only -o fedavg/ --cutoff
python3 measure_f1.py -f sample_datasets/256_0.txt -e 10 -bs 32 -s 0.01 -dp 2 --measure-final-only -o fedavg/ --cutoff
python3 measure_f1.py -f sample_datasets/256_0.txt -e 10 -bs 32 -s 0.001 -dp 2 --measure-final-only -o fedavg/ --cutoff
python3 measure_f1.py -f sample_datasets/256_0.txt -e 10 -bs 32 -s 0.0001 -dp 2 --measure-final-only -o fedavg/ --cutoff
# epochs = 100
python3 measure_f1.py -f sample_datasets/256_0.txt -e 100 -bs 32 -s 0.0 -o fedavg/ --cutoff
python3 measure_f1.py -f sample_datasets/256_0.txt -e 100 -bs 32 -s 0.1 -dp 2 --measure-final-only -o fedavg/ --cutoff
python3 measure_f1.py -f sample_datasets/256_0.txt -e 100 -bs 32 -s 0.01 -dp 2 --measure-final-only -o fedavg/ --cutoff
python3 measure_f1.py -f sample_datasets/256_0.txt -e 100 -bs 32 -s 0.001 -dp 2 --measure-final-only -o fedavg/ --cutoff
python3 measure_f1.py -f sample_datasets/256_0.txt -e 100 -bs 32 -s 0.0001 -dp 2 --measure-final-only -o fedavg/ --cutoff
# epochs = 1000
python3 measure_f1.py -f sample_datasets/256_0.txt -e 1000 -bs 32 -s 0.0 -o fedavg/ --cutoff
python3 measure_f1.py -f sample_datasets/256_0.txt -e 1000 -bs 32 -s 0.1 -dp 2 --measure-final-only -o fedavg/ --cutoff
python3 measure_f1.py -f sample_datasets/256_0.txt -e 1000 -bs 32 -s 0.01 -dp 2 --measure-final-only -o fedavg/ --cutoff 
python3 measure_f1.py -f sample_datasets/256_0.txt -e 1000 -bs 32 -s 0.001 -dp 2 --measure-final-only -o fedavg/ --cutoff
python3 measure_f1.py -f sample_datasets/256_0.txt -e 1000 -bs 32 -s 0.0001 -dp 2 --measure-final-only -o fedavg/ --cutoff
