import numpy as np
import matrix_operations as mo
from statsmodels.tsa.regime_switching import markov_switching as ms


class Expectation:
    def __init__(self, p, theta, sigmas, e_0, regimes):
        self.eta_t_T2 = None
        self.eta_t_T = None
        self.eta_t = None
        self.regimes = regimes
        self.p = p
        self.theta = theta
        self.sigmas = sigmas
        self.e_o = e_0

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

    def run_expectation(self, z_t_1, delta_y_t):
        self.conditional_likelihoods(z_t_1, delta_y_t)
        result = ms.cy_hamilton_filter_log(self.e_o, self.p, self.eta_t, 0)
        predicted_joint_probabilities = result[1]
        filtered_joint_probabilities = result[3]
        result2 = ms.cy_kim_smoother_log(self.p, predicted_joint_probabilities, filtered_joint_probabilities)
        self.eta_t_T = result2[0]
        self.eta_t_T2 = result2[1]
        print(filtered_joint_probabilities[:,1:3] )
