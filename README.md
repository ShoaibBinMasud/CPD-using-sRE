# CPD-using-sRE
## Package Dependencies 
- python=3.6.5
- numpy=1.14.0
- scipy=1.0.0
- pytorch=0.4.1
- pandas=0.23.4
## How to run the code
# Reproducing results for real-world dataset
$ python main.py --dataset beedance -- window 20  --epsilon 1 --DELTA 10 --xi 10
# Reproducing the results on synthetic dataset
Step 1: First generate the synthetic data via running dataset/Synthetic/generate_synthetic_data.py 
Step 2: $ python toy.py --dataset synthetic
