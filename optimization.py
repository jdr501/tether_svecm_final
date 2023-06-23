import numpy as np


class Optimization:
    def __int__(self, eta_t_t_upper, eta_t_t_upper2, regimes):
        self.eta_t_t_upper = eta_t_t_upper
        self.eta_t_t_upper2 = eta_t_t_upper2
        self.regimes = regimes

    def vec_p_hat(self):
        np.kron(np.ones([self.regimes, 1]), eta_t_t_upper)
