import numpy as np


# TODO: change variable names and comment
def get_V_jl(x, l, N, K):
    x = x.squeeze()
    temp = np.zeros((N, K))
    index_cluster = l[x]
    temp[(x, index_cluster)] = 1
    temp = temp.sum(0)
    return temp


# TODO: change variable names and comment
def get_fair_accuracy(u_V, V_list, l, N, K):
    V_j_list = np.array([get_V_jl(x, l, N, K) for x in V_list])
    balance = np.zeros(K)
    J = len(V_list)
    for k in range(K):
        V_j_list_k = V_j_list[:, k].copy()
        balance_temp = np.tile(V_j_list_k, [J, 1])
        balance_temp = balance_temp.T / np.maximum(balance_temp, 1e-20)
        mask = np.ones(balance_temp.shape, dtype=bool)
        np.fill_diagonal(mask, 0)
        balance[k] = balance_temp[mask].min()

    return balance.min(), balance.mean()


# TODO: change variable names and comment
def get_fair_accuracy_proportional(u_V, V_list, l, N, K):
    V_j_list = np.array([get_V_jl(x, l, N, K) for x in V_list])
    clustered_uV = V_j_list / sum(V_j_list)
    fairness_error = np.zeros(K)
    u_V = np.array(u_V)

    for k in range(K):
        fairness_error[k] = (-u_V * np.log(np.maximum(clustered_uV[:, k], 1e-20)) + u_V * np.log(u_V)).sum()
    return fairness_error.sum()
