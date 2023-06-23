import initialization
import expectation
import data
import numpy as np

lag_ = 1
regimes_ = 2
# initialization
df = data.df
b0 = np.random.normal(0, 0.1e-4, size=(4, 4))
b0 = np.zeros([4, 4])  # this ensures there's no random initial values comment this off when running the code
beta_ = np.array([0, 0, 0, 1]).reshape(-1, 1)
initial = initialization.Initializes(df, lag_, regimes_, b0, beta=beta_)
initial.run_initialization()

# expectation
expect = expectation.Expectation(initial.p, initial.initial_theta_hat, initial.sigmas, initial.e_0_0, regimes_)
expect.run_expectation(initial.z_t_1, initial.delta_y_t)

# optimization


