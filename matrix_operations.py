import numpy as np


def mat_vec(matrix):
    """
    This function vectorized any matrix
    :param matrix: should be an ndarray
    :return: returns a vector of the matrix as numpy (row*Col, ) array
    """
    return matrix.T.flatten()


def vec_matrix(vec, r=None, c=None):
    """
    :param vec: numpy array
    :param r: rows of the matrix
    :param c: columns of the matrix
    :return: numpy ndarray
    """
    if r is None and c is None:
        r = c = int(np.sqrt(len(vec)))
    if r is None and type(c) == int:
        r = len(vec) - c
    if c is None and type(r) == int:
        c = len(vec) - r
    return vec.reshape(-1, 1).reshape(c, r).T


def resid(dy_t, z_t, theta, ik):
    u_hat = np.zeros(dy_t.shape)
    for i in range(dy_t.shape[1]):
        z = z_t[:, [i]]
        dy = dy_t[:, [i]]
        u_hat[:, [i]] = dy - np.kron(z.T, ik) @ theta
        return u_hat
