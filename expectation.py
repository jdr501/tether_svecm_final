import numpy as np
import matrix_operations as mo
from statsmodels.tsa.regime_switching import markov_switching as ms


def py_kim_smoother(regime_transition, predicted_joint_probabilities, filtered_joint_probabilities):
    """
    Kim smoother using pure Python

    Parameters
    ----------
    regime_transition : array
        Matrix of regime transition probabilities, shaped either
        (k_regimes, k_regimes, 1) or if there are time-varying transition
        probabilities (k_regimes, k_regimes, nobs).
    predicted_joint_probabilities : array
        Array containing Pr[S_t=s_t, ..., S_{t-order}=s_{t-order} | Y_{t-1}] -
        the joint probability of the current and previous `order` periods
        being in each combination of regimes conditional on time t-1
        information. Shaped (k_regimes,) * (order + 1) + (nobs,).
    filtered_joint_probabilities : array
        Array containing Pr[S_t=s_t, ..., S_{t-order}=s_{t-order} | Y_{t}] -
        the joint probability of the current and previous `order` periods
        being in each combination of regimes conditional on time t
        information. Shaped (k_regimes,) * (order + 1) + (nobs,).

    Returns
    -------
    smoothed_joint_probabilities : array
        Array containing Pr[S_t=s_t, ..., S_{t-order}=s_{t-order} | Y_T] -
        the joint probability of the current and previous `order` periods
        being in each combination of regimes conditional on all information.
        Shaped (k_regimes,) * (order + 1) + (nobs,).
    smoothed_marginal_probabilities : array
        Array containing Pr[S_t=s_t | Y_T] - the probability of being in each
        regime conditional on all information. Shaped (k_regimes, nobs).
    """

    # Dimensions
    k_regimes = filtered_joint_probabilities.shape[0]
    nobs = filtered_joint_probabilities.shape[-1]
    order = filtered_joint_probabilities.ndim - 2
    dtype = filtered_joint_probabilities.dtype

    # Storage
    smoothed_joint_probabilities = np.zeros(
        (k_regimes,) * (order + 1) + (nobs,), dtype=dtype)
    smoothed_marginal_probabilities = np.zeros((k_regimes, nobs), dtype=dtype)

    # S_T, S_{T-1}, ..., S_{T-r} | T
    smoothed_joint_probabilities[..., -1] = (
        filtered_joint_probabilities[..., -1])

    # Reshape transition so we can use broadcasting
    shape = (k_regimes, k_regimes)
    shape += (1,) * (order)
    shape += (regime_transition.shape[-1],)
    regime_transition = np.reshape(regime_transition, shape)

    # Get appropriate subset of transition matrix
    if regime_transition.shape[-1] == nobs + order:
        regime_transition = regime_transition[..., order:]

    # Kim smoother iterations
    transition_t = 0
    for t in range(nobs - 2, -1, -1):
        if regime_transition.shape[-1] > 1:
            transition_t = t + 1

        # S_{t+1}, S_t, ..., S_{t-r+1} | t
        # x = predicted_joint_probabilities[..., t]
        x = (filtered_joint_probabilities[..., t] *
             regime_transition[..., transition_t])
        # S_{t+1}, S_t, ..., S_{t-r+2} | T / S_{t+1}, S_t, ..., S_{t-r+2} | t
        y = (smoothed_joint_probabilities[..., t + 1] /
             predicted_joint_probabilities[..., t + 1])
        # S_t, S_{t-1}, ..., S_{t-r+1} | T
        smoothed_joint_probabilities[..., t] = (x * y[..., None]).sum(axis=0)

    # Get smoothed marginal probabilities S_t | T by integrating out
    # S_{t-k+1}, S_{t-k+2}, ..., S_{t-1}
    smoothed_marginal_probabilities = smoothed_joint_probabilities
    for i in range(1, smoothed_marginal_probabilities.ndim - 1):
        smoothed_marginal_probabilities = np.sum(
            smoothed_marginal_probabilities, axis=-2)

    return smoothed_joint_probabilities, smoothed_marginal_probabilities


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
        result2 = py_kim_smoother(self.p, predicted_joint_probabilities, filtered_joint_probabilities)
        self.eta_t_T = result2[0]
        self.eta_t_T2 = result2[1]
        print(self.eta_t_T[:, 1:5])
