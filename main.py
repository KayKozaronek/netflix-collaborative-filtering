import numpy as np
import kmeans
import common
import naive_em
import em

X = np.loadtxt("data/toy_data.txt")

# TODO: Your code here
for k in range(1,5):
    for seed in range(0,5):
        m, p = common.init(X, k, seed)
        m, p, c = kmeans.run(X, m, p)
        common.plot(X, m, p, f"Plot + {k}")
        print(c)