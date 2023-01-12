# CPD-using-sRE
## Package Dependencies to sRMMMD-based knockoff filter
- python=3.6.5
- numpy=1.14.0
- scipy=1.0.0
- pytorch=0.4.1
- cvxpy=1.0.10
- cvxopt=1.2.0
- pandas=0.23.4
## How to run the code
# For real-world dataset
$ python main.py --dataset beedance -- window 20  --epsilon 1 -DELTA 10 --xi 10
# To reproduce the results on synthetic dataset
$ python toy.py --dataset synthetic
