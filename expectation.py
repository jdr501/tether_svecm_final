import numpy as np
import matrix_operations as mo
from statsmodels.tsa.regime_switching import markov_switching as ms


def hamilton(eta, p_hat, initial_value):
    """
    :param eta: marginal density t= 1,...,T
    :param p_hat: transition probability matrix
    :param initial_value: initial eta_0_0
    :return:
    optimal inference
    optimal forecast
    """
    # creating empty array for to save values
    regimes = eta.shape[0]
    t_dimension = eta.shape[1] + 1  # note that eta is from 1...T but epsilons runs from 0...T
    epsilon_t_t_1 = np.zeros([regimes, t_dimension])
    epsilon_t_t = np.zeros([regimes, t_dimension])
    epsilon_t_t[:, [0]] = initial_value
    # iteration from t = 1,..., T
    for t in range(1, t_dimension):
        # print(epsilon_t_t[:, [t - 1]])
        eps = p_hat @ epsilon_t_t[:, [t - 1]]
        # print(f'this is eps: {eps}for period:t={t}')
        epsilon_t_t_1[:, [t]] = eps
        # print(f'this is eta: {eta[:, [t-1]]}for period:t={t}')
        tmp = np.multiply(eta[:, [t - 1]], eps)
        # print(f'this is tmp:{tmp}for period:t={t}')
        epsilon_t_t[:, [t]] = tmp / tmp.sum(axis=0)
    return epsilon_t_t_1, epsilon_t_t


def kim(p_hat, epsilon_t_t):
    """
    :param p_hat: transition probabilities
    :param epsilon_t_t: optimal inference from hamilton filter
    :return:
    epsilon_t_T smoothed inference
    epsilon_t_T(2)
    """
    vec_p_trans = mo.mat_vec(p_hat.T)

    # creating empty array to save the values
    regimes = epsilon_t_t.shape[0]
    t_dimension = epsilon_t_t.shape[1]
    epsilon_t_t_upper = np.zeros([regimes, t_dimension])
    epsilon_t_t_upper_2 = np.zeros([regimes * regimes, t_dimension])

    epsilon_t_t_upper[:, [-1]] = epsilon_t_t[:, [-1]]

    # iteration for smoothed inference
    for t in range(epsilon_t_t.shape[1] - 2, -1, -1):
        inner = np.divide(epsilon_t_t_upper[:, [t + 1]], p_hat @ epsilon_t_t[:, [t]])
        outer = (p_hat.T @ inner)
        epsilon_t_t_upper[:, [t]] = outer * epsilon_t_t[:, [t]]

    # iteration for epsilon_t_T(2)
    for t in range(0, epsilon_t_t.shape[1] - 2):  # 0,...,T-1
        inner = np.divide(epsilon_t_t_upper[:, [t + 1]], p_hat @ epsilon_t_t[:, [t]])
        outer = np.kron(inner, epsilon_t_t[:, [t]])
        epsilon_t_t_upper_2[:, [t]] = vec_p_trans * outer

    return epsilon_t_t_upper, epsilon_t_t_upper_2


class Expectation:
    def __init__(self, p, theta, sigmas, e_0, regimes):
        self.epsilon_t_T2 = None
        self.epsilon_t_T = None
        self.eta_t = None
        self.regimes = regimes
        self.p = p
        self.theta = theta
        self.sigmas = sigmas
        self.e_o = e_0
        self.result = None
        self.likelihood = None

    def conditional_likelihoods(self, z_t_1, delta_y_t):
        t_len = delta_y_t.shape[-1]
        ut = mo.resid(delta_y_t, z_t_1, self.theta)
        eta_t = np.zeros([self.regimes, t_len])
        k = delta_y_t.shape[0]
        for t in range(t_len):
            u = ut[:, [t]]
            for regime in range(self.regimes):
                eta_t[regime, t] = (2 * np.pi) ** (-k / 2) * \
                                   np.sqrt(np.linalg.det(self.sigmas[regime])) * \
                                   np.exp(-0.5 * u.T @ np.linalg.pinv(self.sigmas[regime]) @ u)
        self.eta_t = eta_t

    def convergence(self, epsilon_t_t_1):
        lt_sum = 0
        for t in range(1, epsilon_t_t_1.shape[1]):
            lt_sum = lt_sum + np.log(epsilon_t_t_1[:, [t]].T @ self.eta_t[:, [t-1]])

        self.likelihood = lt_sum

    def run_expectation(self, z_t_1, delta_y_t):
        self.conditional_likelihoods(z_t_1, delta_y_t)
        epsilon_t_t_1, epsilon_t_t = hamilton(self.eta_t, self.p, self.e_o)
        result2 = kim(self.p, epsilon_t_t)
        self.epsilon_t_T = result2[0]
        self.epsilon_t_T2 = result2[1]
        self.result = result2
        self.convergence(epsilon_t_t_1)
