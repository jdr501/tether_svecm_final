import numpy as np
import scipy as sp
from statsmodels.tsa.regime_switching import markov_switching as ms
from statsmodels.tools.sm_exceptions import EstimationWarning
import matrix_operations as mo
import warnings
from scipy.optimize import Bounds
from scipy.optimize import minimize as mn


class Optimization:
    def __init__(self, epsilon_t_t_upper, epsilon_t_t_upper2, regimes, theta, u_hat):
        self.theta_hat = None
        self.b_hat_matrix = None
        self.result = None
        self.vec_sum = None
        self.t_m = None
        self.vec_p_hat_ = None
        self.p = None
        self.epsilon_t_t_upper = epsilon_t_t_upper
        self.epsilon_t_t_upper2 = epsilon_t_t_upper2
        self.regimes = regimes
        self.theta = theta
        self.u_hat = u_hat
        self.k = len(u_hat)
        self.t_dimension = u_hat.shape[1]

        sigma_hat_list = []
        for m in range(self.regimes):
            sigma_hat_list.append(np.zeros([self.k, self.k]))
        self.sigma_hat = sigma_hat_list

        lam_hat_list = []
        for m in range(self.regimes - 1):
            lam_hat_list.append(np.zeros([self.k, self.k]))
        self.lam_m_hat = lam_hat_list

    def em_regime_transition(self, result):
        """
        EM step for regime transition probabilities
        """
        numo = np.sum(self.epsilon_t_t_upper2[:, 0:-1], axis=1).reshape((-1, 1))
        denom = np.kron(np.ones([self.regimes, 1]), np.sum(self.epsilon_t_t_upper[:, 1:], axis=1).reshape(-1, 1))
        vecp = np.divide(numo, denom)
        regime_transition = mo.vec_matrix(vecp, r=self.regimes, c=self.regimes)
        # Marginalize the smoothed joint probabilities to just S_t, S_{t-1} | T
        tmp = result[0]
        for i in range(self.regimes):

            # Transition parameters (recall we're not yet supporting TVTP here)

            # It may be the case that due to rounding error this estimates
            # transition probabilities that sum to greater than one. If so,
            # re-scale the probabilities and warn the user that something
            # is not quite right
            delta = np.sum(regime_transition[i]) - 1
            if delta > 0:
                warnings.warn('Invalid regime transition probabilities'
                              ' estimated in EM iteration; probabilities have'
                              ' been re-scaled to continue estimation.', EstimationWarning)
                regime_transition[i] /= 1 + delta + 1e-6

        return regime_transition

    def vec_p_hat(self, result):
        self.p = self.em_regime_transition(result)
        self.vec_p_hat_ = mo.mat_vec(self.p)

    def likelihood_constant_param(self):
        vec_sum = []
        t_m = []
        for regime in range(self.regimes):
            t_m.append((self.epsilon_t_t_upper[regime, :]).sum())
            vec_sum.append(mo.vec_summation(self.epsilon_t_t_upper[regime, 1:], self.u_hat))

        self.t_m = t_m
        self.vec_sum = vec_sum

    def likelihood(self, x):
        # The following code assigns values of x array to b matrix and the lambdas
        shape_b = self.k * self.k
        b_matrix = np.array([x[i] for i in range(shape_b)]
                            ).reshape(self.k, self.k).T
        lam_m = []
        for regime in range(self.regimes - 1):
            lam_m.append(np.zeros([self.k, self.k]))

        x_lambda_values = shape_b
        for regime in range(self.regimes - 1):
            for k in range(self.k):
                lam_m[regime][k, k] = x[x_lambda_values + k]
            x_lambda_values = x_lambda_values + self.k

        sum_likelihoods = 0
        for regime in range(self.regimes):
            if regime == 0:
                sum_likelihoods = self.t_dimension * np.log(
                    np.absolute(np.linalg.det(b_matrix))) + \
                                  0.5 * np.trace(
                    (np.linalg.pinv(b_matrix.T) @
                     np.linalg.pinv(b_matrix)) @ self.vec_sum[regime]

                )
            else:
                #print(f'this is determinant {np.linalg.det(lam_m[regime - 1])}')
                #print(f'this is log determinant{np.log(np.linalg.det(lam_m[regime - 1]))}')
                sum_likelihoods = sum_likelihoods + \
                                  self.t_m[regime] / 2 * \
                                  np.log(np.linalg.det(
                                      lam_m[regime - 1])) + 0.5 * np.trace((np.linalg.pinv(b_matrix.T) @
                                                                            np.linalg.pinv(lam_m[regime - 1]) @
                                                                            np.linalg.pinv(b_matrix)) @
                                                                           self.vec_sum[regime])

        return sum_likelihoods

    def optimization(self, x0):
        print('tabulating initial params')
        self.likelihood_constant_param()
        print('constant params pass')

        # Bounds
        length = (self.k + self.regimes - 1) * self.k
        lower_bound = []
        upper_bound = []
        for i in range(length):
            if i > self.k * self.k - 1:
                lower_bound.append(0.01)  # lower bound for lambda
            else:
                lower_bound.append(-10000)  # lower bound for B
            upper_bound.append(10000)  # upper bound
        bounds = Bounds(lower_bound, upper_bound)

        # Numerical Optimization
        self.result = mn(self.likelihood, x0, bounds=bounds, method='COBYLA',
                         options={'maxiter': 15000, 'disp': False})  #
        print(self.result['x'])

        # B^hat matrix estimate
        self.b_hat_matrix = self.result['x'][0:self.k * self.k].reshape(self.k, self.k).T

        # Lambda m hat
        start = self.k * self.k
        for m in range(self.regimes - 1):
            end = start + self.k
            self.lam_m_hat[m] = mo.replace_diagonal(self.result['x'][start:end])
            start = end
        # Sigma hat
        for m in range(self.regimes):
            if m == 0:
                self.sigma_hat[m] = self.b_hat_matrix @ self.b_hat_matrix.T
            else:
                self.sigma_hat[m] = self.b_hat_matrix @ self.lam_m_hat[m - 1] @ self.b_hat_matrix.T

    """
    def estimate_theta_hat(self, zt_1, delta_yt):
        denominator = 0
        for m in range(self.regimes):
            denominator = denominator + np.kron(
                                  mo.vec_summation(self.epsilon_t_t_upper[m, 1:], zt_1),
                                  np.linalg.pinv(self.sigma_hat[m]))
        denominator = np.linalg.pinv(denominator)

        numerator = 0
        regime_sum = 0
        for t in range(1, self.t_dimension):
            for m in range(self.regimes):
                regime_sum = regime_sum + np.kron(
                    (self.epsilon_t_t_upper[m, t] * zt_1[:, [t]]),
                    np.linalg.pinv(
                        self.sigma_hat[m]))

            numerator = numerator + regime_sum @ delta_yt[:, t]
        theta_hat = denominator @ numerator
        self.theta_hat = theta_hat.reshape([-1, 1])
   """

    def estimate_theta_hat(self, zt_1, delta_yt):
        # Denominator for the estimate of theta hat
        msum = 0
        for m in range(self.regimes):
            tsum = 0
            for t in range(1, self.t_dimension):
                zt_new = zt_1[:, [t]]
                tsum = tsum + self.epsilon_t_t_upper[m, [t]] * zt_new @ zt_new.T
            msum = msum + np.kron(tsum, np.linalg.pinv(self.sigma_hat[m]))
        denom = np.linalg.pinv(msum)
        # print(denom.shape)

        # numerator for the estimate of theta hat
        tsum = 0
        for t in range(1, self.t_dimension):
            d_yt_new = delta_yt[:, [t]]
            zt_new = zt_1[:, [t]]
            msum = 0
            for m in range(self.regimes):
                msum = msum + np.kron((self.epsilon_t_t_upper[m, [t]] * zt_new), np.linalg.pinv(self.sigma_hat[m]))
            tsum = tsum + msum @ d_yt_new
            # print(tsum.shape)
        self.theta_hat = denom @ tsum
        self.theta_hat = self.theta_hat.reshape([-1, 1])
