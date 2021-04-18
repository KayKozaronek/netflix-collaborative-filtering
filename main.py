import numpy as np
import kmeans
import common
import naive_em
import em

# X = np.loadtxt("data/toy_data.txt")

# for k in range(1,5):
#     print(f"Clusters: {k}")
#     for seed in range(0,5):
#         print(f"Seed:  {seed}")
#         # KMeans
#         for alg in [kmeans, naive_em]:
#             if alg == kmeans:
#                 name = "Kmeans"
#             else:
#                 name = "Naive_EM"
#             m, p = common.init(X, k, seed)
#             m, p, c = alg.run(X, m, p)
#             common.plot(X, m, p, f"{name} Plot + {k}")
#             print(f"{name} result: {c}")

# # Naive EM
# for k in range(1,5):
#     print(f"Clusters: {k}")
#     for seed in range(0,5):
#         m, p = common.init(X, k, seed)
#         m, p, c = naive_em.run(X, m, p)
#         common.plot(X, m, p, f"Plot + {k}")
#         print(c)


# for K in range(1,5):
#     m, p = common.init(X, K, seed = 0)
#     m, p, c = naive_em.run(X, m, p)
#     common.plot(X,m,p,f"Plot {K}")
#     bic = common.bic(X, m, c)
#     print(f"K: {K}, C: {c}, BIC: {bic}")

X = np.loadtxt("data/netflix_incomplete.txt")

# for K in [1,12]:
#     for seed in range(5):
#         m, p = common.init(X, K, seed = seed)
#         m, p, c = em.run(X, m, p)
#         common.plot(X,m,p,f"Plot {K}")
#         bic = common.bic(X, m, c)
#         print(f"K: {K}, C: {c}, BIC: {bic}")

X_gold = np.loadtxt('data/netflix_complete.txt')

K = 12
seed = 1

m, p = common.init(X, K, seed = seed)
m, p, c = em.run(X, m, p)

X_pred = em.fill_matrix(X, m)

rmse = common.rmse(X_gold, X_pred)

print(rmse)