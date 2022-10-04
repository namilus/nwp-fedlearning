# measure magnitude of tokens for fedavg for different noise levels (DPSGD)
python3 measure_magnitude.py -f sample_datasets/256_0.txt -e 1 -bs 32 -s 0.1 -dp 1 -o magnitude_results/
python3 measure_magnitude.py -f sample_datasets/256_0.txt -e 10 -bs 32 -s 0.1 -dp 1 -o magnitude_results/
python3 measure_magnitude.py -f sample_datasets/256_0.txt -e 100 -bs 32 -s 0.1 -dp 1 -o magnitude_results/
python3 measure_magnitude.py -f sample_datasets/256_0.txt -e 1000 -bs 32 -s 0.1 -dp 1 -o magnitude_results/

python3 measure_magnitude.py -f sample_datasets/256_0.txt -e 1 -bs 32 -s 0.01 -dp 1 -o magnitude_results/
python3 measure_magnitude.py -f sample_datasets/256_0.txt -e 10 -bs 32 -s 0.01 -dp 1 -o magnitude_results/
python3 measure_magnitude.py -f sample_datasets/256_0.txt -e 100 -bs 32 -s 0.01 -dp 1 -o magnitude_results/
python3 measure_magnitude.py -f sample_datasets/256_0.txt -e 1000 -bs 32 -s 0.01 -dp 1 -o magnitude_results/

python3 measure_magnitude.py -f sample_datasets/256_0.txt -e 1 -bs 32 -s 0.001 -dp 1 -o magnitude_results/
python3 measure_magnitude.py -f sample_datasets/256_0.txt -e 10 -bs 32 -s 0.001 -dp 1 -o magnitude_results/
python3 measure_magnitude.py -f sample_datasets/256_0.txt -e 100 -bs 32 -s 0.001 -dp 1 -o magnitude_results/
python3 measure_magnitude.py -f sample_datasets/256_0.txt -e 1000 -bs 32 -s 0.001 -dp 1 -o magnitude_results/

python3 measure_magnitude.py -f sample_datasets/256_0.txt -e 1 -bs 32 -s 0.0001 -dp 1 -o magnitude_results/
python3 measure_magnitude.py -f sample_datasets/256_0.txt -e 10 -bs 32 -s 0.0001 -dp 1 -o magnitude_results/
python3 measure_magnitude.py -f sample_datasets/256_0.txt -e 100 -bs 32 -s 0.0001 -dp 1 -o magnitude_results/
python3 measure_magnitude.py -f sample_datasets/256_0.txt -e 1000 -bs 32 -s 0.0001 -dp 1 -o magnitude_results/


# measure magnitude of tokens for fedavg for different noise levels (SINGLENOISE)
python3 measure_magnitude.py -f sample_datasets/256_0.txt -e 1 -bs 32 -s 0.1 -dp 2 -o magnitude_results/
python3 measure_magnitude.py -f sample_datasets/256_0.txt -e 10 -bs 32 -s 0.1 -dp 2 -o magnitude_results/
python3 measure_magnitude.py -f sample_datasets/256_0.txt -e 100 -bs 32 -s 0.1 -dp 2 -o magnitude_results/
python3 measure_magnitude.py -f sample_datasets/256_0.txt -e 1000 -bs 32 -s 0.1 -dp 2 -o magnitude_results/

python3 measure_magnitude.py -f sample_datasets/256_0.txt -e 1 -bs 32 -s 0.01 -dp 2 -o magnitude_results/
python3 measure_magnitude.py -f sample_datasets/256_0.txt -e 10 -bs 32 -s 0.01 -dp 2 -o magnitude_results/
python3 measure_magnitude.py -f sample_datasets/256_0.txt -e 100 -bs 32 -s 0.01 -dp 2 -o magnitude_results/
python3 measure_magnitude.py -f sample_datasets/256_0.txt -e 1000 -bs 32 -s 0.01 -dp 2 -o magnitude_results/

python3 measure_magnitude.py -f sample_datasets/256_0.txt -e 1 -bs 32 -s 0.001 -dp 2 -o magnitude_results/
python3 measure_magnitude.py -f sample_datasets/256_0.txt -e 10 -bs 32 -s 0.001 -dp 2 -o magnitude_results/
python3 measure_magnitude.py -f sample_datasets/256_0.txt -e 100 -bs 32 -s 0.001 -dp 2 -o magnitude_results/
python3 measure_magnitude.py -f sample_datasets/256_0.txt -e 1000 -bs 32 -s 0.001 -dp 2 -o magnitude_results/

python3 measure_magnitude.py -f sample_datasets/256_0.txt -e 1 -bs 32 -s 0.0001 -dp 2 -o magnitude_results/
python3 measure_magnitude.py -f sample_datasets/256_0.txt -e 10 -bs 32 -s 0.0001 -dp 2 -o magnitude_results/
python3 measure_magnitude.py -f sample_datasets/256_0.txt -e 100 -bs 32 -s 0.0001 -dp 2 -o magnitude_results/
python3 measure_magnitude.py -f sample_datasets/256_0.txt -e 1000 -bs 32 -s 0.0001 -dp 2 -o magnitude_results/


# measure magnitude of tokens for fedsgd for different noise levels  (DPSGD)
python3 measure_magnitude.py -f sample_datasets/256_0.txt -e 1 -bs 256 -s 0.1 -dp 1 -o magnitude_results/
python3 measure_magnitude.py -f sample_datasets/256_0.txt -e 1 -bs 256 -s 0.01 -dp 1 -o magnitude_results/
python3 measure_magnitude.py -f sample_datasets/256_0.txt -e 1 -bs 256 -s 0.001 -dp 1 -o magnitude_results/
python3 measure_magnitude.py -f sample_datasets/256_0.txt -e 1 -bs 256 -s 0.0001 -dp 1 -o magnitude_results/

# measure magnitude of tokens for fedsgd for different noise levels  (SINGLENOISE)
python3 measure_magnitude.py -f sample_datasets/256_0.txt -e 1 -bs 256 -s 0.1 -dp 2 -o magnitude_results/
python3 measure_magnitude.py -f sample_datasets/256_0.txt -e 1 -bs 256 -s 0.01 -dp 2 -o magnitude_results/
python3 measure_magnitude.py -f sample_datasets/256_0.txt -e 1 -bs 256 -s 0.001 -dp 2 -o magnitude_results/
python3 measure_magnitude.py -f sample_datasets/256_0.txt -e 1 -bs 256 -s 0.0001 -dp 2 -o magnitude_results/
