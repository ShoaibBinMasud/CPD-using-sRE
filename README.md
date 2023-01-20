# CPD-using-sRE
## Package Dependencies 
- python=3.6.5
- numpy=1.14.0
- scipy=1.0.0
- pytorch=0.4.1
- pandas=0.23.4
## How to run the code
# For real-world dataset
$ python main.py --dataset beedance -- window 20  --epsilon 1 -DELTA 10 --xi 10
# To reproduce the results on synthetic dataset
$ python toy.py --dataset synthetic
