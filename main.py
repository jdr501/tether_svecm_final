import initialization
import expectation
import optimization
import matrix_operations as mo
import data
import numpy as np

lag_ = 1
regimes_ = 2
# initialization
df = data.df
b0 = np.random.normal(0, 0.1, size=(4, 4))
#b0 = np.zeros([4, 4])  # this ensures there's no random initial values comment this off when running the code
beta_ = np.array([0, 0, 0, 1]).reshape(-1, 1)
initial = initialization.Initializes(df, lag_, regimes_, b0, beta=beta_)
initial.run_initialization()
print(initial.initial_b)
# expectation
expect = expectation.Expectation(initial.p, initial.initial_theta_hat, initial.sigmas, initial.e_0_0, regimes_)
expect.run_expectation(initial.z_t_1, initial.delta_y_t)
print(expect.epsilon_t_T)
print(expect.epsilon_t_T2)
# optimization

opt = optimization.Optimization(expect.epsilon_t_T, expect.epsilon_t_T2, regimes_, initial.initial_theta_hat,
                                initial.u_hat)
print(opt.em_regime_transition(expect.result))
opt.vec_p_hat(expect.result)
opt.likelihood_constant_param()
print(opt.vec_p_hat_)
x0 = [5.96082486e+01, 5.74334765e-01, 2.83277325e-01, 3.66479528e+00,
      -2.08881529e-01, 6.32170541e-04, -1.09137417e-01, -3.80763529e-01,
      4.24379418e+00, 1.83658083e-01, 2.16692718e-03, 1.29590368e+00,
      2.20826553e+00, -2.98484217e-01, -5.38269363e-03, 1.19668239e-03,
      0.012, 0.102, 0.843, 16.52]
opt.likelihood(x0)
opt.optimization(x0)
opt.estimate_theta_hat(initial.z_t_1, initial.delta_y_t)
print(opt.theta_hat)
print(f'this is sigmahat: {opt.sigma_hat}')

for i in range(3):
    print(f'THIS IS ITERATION NUMBER:{i}')
    # expectation
    print(expect.epsilon_t_T[:, [0]])
    p_hat = mo.vec_matrix(opt.p, r=opt.regimes, c=opt.regimes).T
    print(f'phat:{p_hat}')
    expect = expectation.Expectation(p_hat, opt.theta_hat, opt.sigma_hat, expect.epsilon_t_T[:, [0]], regimes_)
    expect.run_expectation(initial.z_t_1, initial.delta_y_t)

    # optimization

    opt = optimization.Optimization(expect.epsilon_t_T, expect.epsilon_t_T2, regimes_, initial.initial_theta_hat,
                                    initial.u_hat)
    opt.em_regime_transition(expect.result)
    opt.vec_p_hat(expect.result)
    opt.likelihood_constant_param()
    print(opt.vec_p_hat_)
    x0 = [5.96082486e+01, 5.74334765e-01, 2.83277325e-01, 3.66479528e+00,
          -2.08881529e-01, 6.32170541e-04, -1.09137417e-01, -3.80763529e-01,
          4.24379418e+00, 1.83658083e-01, 2.16692718e-03, 1.29590368e+00,
          2.20826553e+00, -2.98484217e-01, -5.38269363e-03, 1.19668239e-03,
          0.012, 0.102, 0.843, 16.52]
    opt.likelihood(x0)
    opt.optimization(x0)
    opt.estimate_theta_hat(initial.z_t_1, initial.delta_y_t)
    print(opt.theta_hat)
