import numpy as np
import em
import common

X = np.loadtxt("data/test_incomplete.txt")
X_gold = np.loadtxt("data/test_complete.txt")

K = 4
n, d = X.shape
seed = 0

# TODO: Your code here
