import math

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances
import numexpr as ne
from numba import jit  # ToDo: check optimization
from utils.metrics import get_fair_accuracy_proportional


def normalize(S_in):
    maxcol = S_in.max(1)[:, np.newaxis]
    S_in = ne.evaluate('S_in - maxcol')
    S_out = np.exp(S_in)
    S_out_sum = S_out.sum(1)[:, np.newaxis]
    S_out = ne.evaluate('S_out/S_out_sum')
    return S_out


def normalize_2(S_in):
    S_in_sum = S_in.sum(1)[:, np.newaxis]
    # S_in = np.divide(S_in,S_in_sum)
    S_in = ne.evaluate('S_in/S_in_sum')
    return S_in


def bound_energy(S, S_in, a_term, b_term):
    return np.nansum((S*np.log(np.maximum(S, 1e-15)) - S*np.log(np.maximum(S_in, 1e-15)) + a_term*S + b_term*S))


# @jit(parallel=True)
def compute_b_j_parallel(J, S, V_list, u_V):
    result = [compute_b_j(V_list[j], u_V[j], S) for j in range(J)]
    return result


def compute_b_j(V_j,u_j,S_):
    N,K = S_.shape
    V_j = V_j.astype('float')
    S_sum = S_.sum(0)
    R_j = ne.evaluate('u_j*(1/S_sum)')
    F_j_a = np.tile((ne.evaluate('u_j*V_j')), [K, 1]).T
    F_j_b = np.maximum(np.tile(np.dot(V_j, S_), [N, 1]), 1e-15)
    F_j = ne.evaluate('R_j - (F_j_a/F_j_b)')
    return F_j


# @jit
def get_S_discrete(l,N,K):
    x = range(N)
    temp = np.zeros((N,K),dtype=float)
    temp[(x, l)] = 1
    return temp


def fairness_term_V_j(u_j, S, V_j):
    V_j = V_j.astype('float')
    S_term = np.maximum(np.dot(V_j, S), 1e-20)
    S_sum = np.maximum(S.sum(axis=0), 1e-20)
    S_term = ne.evaluate('u_j*(log(S_sum) - log(S_term))')
    return S_term


class FairKmeans:
    def __init__(self, fair_lambda, n_clusters, n_init=25, lipchitz_value=2.0, max_iter=100, bound_iterations=10000):
        self.fair_lambda = fair_lambda
        self.labels_ = None
        self.dataset_balance = None
        self.proportion_bias_variable = None  # U_v
        self.cluster_centers_ = None
        self.V_list = None
        self.max_iter = max_iter
        self.bound_iterations = bound_iterations
        self.lipchitz_value = lipchitz_value
        self.fairness_errors = []
        self.n_clusters = n_clusters
        self.n_init = n_init

        # TODO: check meaning of next variables
        self.S = []
        self.E_org = []
        self.E_cluster = []
        self.E_fair = []
        self.E_cluster_discrete = []
        # TODO END

    def fit(self, X, bias_vector):
        rows_amount, _ = X.shape
        self.V_list = [np.array(bias_vector == j) for j in np.unique(bias_vector)]
        V_sum = [x.sum() for x in self.V_list]

        self.dataset_balance = min(V_sum) / max(V_sum)
        self.proportion_bias_variable = [x / rows_amount for x in V_sum]
        # initial values
        # ==============
        self.cluster_centers_ = KMeans(n_clusters=self.n_clusters)._init_centroids(X=X,
                                                                                   x_squared_norms=None,
                                                                                   random_state=np.random.RandomState(),
                                                                                   n_centroids=self.n_clusters,
                                                                                   init='k-means++')
        self.labels_ = euclidean_distances(X, self.cluster_centers_).argmin(axis=1)
        # ==============
        self.fair_clustering_train(X, rows_amount)

    def fit_transform(self, X, bias_vector):
        self.fit(X, bias_vector)
        return self.labels_

    def fit_predict(self, X, bias_vector):
        return self.fit_transform(X, bias_vector)

    def fair_clustering_train(self, X, rows_dimensions):
        old_fair_clustering_energy = None
        for _ in range(self.max_iters):
            tmp_list = [np.where(self.labels_ == k)[0] for k in range(self.clusters_amount)]
            self.cluster_centers_ = [X[tmp, :].mean(axis=0) for tmp in tmp_list]  # updating cluster centers
            centers_stacked = np.asarray(np.vstack(self.cluster_centers_))
            square_distances = euclidean_distances(X, centers_stacked, squared=True)
            a_p = square_distances.copy()

            labels, S, bound_E = self.bound_update(a_p)

            self.fairness_errors.append(
                get_fair_accuracy_proportional(self.proportion_bias_variable, self.V_list, labels, rows_dimensions, self.clusters_amount)
            )

            current_clustering_energy, clusterE, fairE, clusterE_discrete = self.compute_energy_fair_clustering(X, centers_stacked, labels, S)
            self.E_org.append(current_clustering_energy)
            self.E_cluster.append(clusterE)
            self.E_fair.append(fairE)
            self.E_cluster_discrete.append(clusterE_discrete)

            if old_fair_clustering_energy is not None:
                if abs(current_clustering_energy - old_fair_clustering_energy) <= 1e-4 * abs(old_fair_clustering_energy):
                    print('......Job  done......')
                    break
            old_fair_clustering_energy = current_clustering_energy

    def bound_update(self, a_p):
        old_bound_energy = float('inf')
        J = len(self.proportion_bias_variable)
        S = normalize_2(np.exp((-a_p)))
        report_energy = None

        for i in range(self.bound_iterations):
            S_in = S.copy()
            # Get a and b
            terms = -a_p.copy()

            b_j_list = compute_b_j_parallel(J, S, self.V_list, self.proportion_bias_variable)
            b_j_list = sum(b_j_list)
            fair_lambda = self.fair_lambda
            lipchitz_value = self.lipchitz_value

            b_term = ne.evaluate('fair_lambda * b_j_list')
            terms = ne.evaluate('(terms - b_term)/lipchitz_value')
            S_in_2 = normalize(terms)
            S = ne.evaluate('S_in * S_in_2')
            S = normalize_2(S)

            energy_bound = bound_energy(S, S_in, a_p, b_term)
            report_energy = energy_bound

            if i > 1 and (abs(energy_bound - old_bound_energy) <= 1e-5 * abs(old_bound_energy)):
                print('Converged')
                break
            else:
                old_bound_energy = energy_bound

        labels = np.argmax(S, axis=1)
        return labels, S, report_energy

    def compute_energy_fair_clustering(self, X, C, labels, S):
        print('compute energy')
        J = len(self.proportion_bias_variable)
        N, K = S.shape

        e_dist = euclidean_distances(X, C, squared=True)
        clustering_E = ne.evaluate('S*e_dist').sum()
        # the following calculation is the sum of the discrete energy of every label
        clustering_E_discrete = sum([np.sum(e_dist[np.asarray(np.where(labels == k)).squeeze(), k]) for k in range(K)])

        # Fairness term
        fairness_E = [fairness_term_V_j(self.proportion_bias_variable[j], S, self.V_list[j]) for j in range(J)]
        fairness_E = (self.fair_lambda * sum(fairness_E)).sum()

        energy_fair_clustering = clustering_E + fairness_E

        return energy_fair_clustering, clustering_E, fairness_E, clustering_E_discrete
