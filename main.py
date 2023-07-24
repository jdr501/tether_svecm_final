"hello "
import initialization
import expectation
import optimization
import matrix_operations as mo
import data
import numpy as np
import matplotlib.pyplot as plt
print('hello')
lag_ = 1
regimes_ = 2
# initialization
df = data.df
b0 = np.random.normal(0, 0.1, size=(4, 4))
beta_ = np.array([0, 0, 0, 1]).reshape(-1, 1)

liklihood_values = []
liklihood_values.append(-10)
liklihood_values.append(-8)
i = 0
while np.abs(liklihood_values[i - 1] - liklihood_values[i - 2]) > 1e-4:
    print(f'THIS IS THE ITERATION NUMBER :{i}')

    if i == 0:
        initial = initialization.Initializes(df, lag_, regimes_, b0, beta=beta_)
        initial.run_initialization()
        print(initial.initial_b)

        # expectation
        expect = expectation.Expectation(initial.p, initial.initial_theta_hat, initial.sigmas, initial.e_0_0, regimes_)
        expect.run_expectation(initial.z_t_1, initial.delta_y_t)
        print(expect.epsilon_t_T)
        print(expect.epsilon_t_T2)
        print(f'this is likelihood values:{expect.likelihood}')
        liklihood_values.append(expect.likelihood[0][0])
        # optimization

        opt = optimization.Optimization(expect.epsilon_t_T, expect.epsilon_t_T2, regimes_, initial.initial_theta_hat,
                                        initial.u_hat)
        print(opt.em_regime_transition(expect.result))
        opt.vec_p_hat(expect.result)
        opt.likelihood_constant_param()
        print(opt.vec_p_hat_)
        print(f'this is the shape of initial theta{initial.initial_theta_hat.shape}')

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


    else:
        x0 = opt.result['x']
        # expectation
        print(expect.epsilon_t_T[:, [0]])
        p_hat = mo.vec_matrix(opt.p, r=opt.regimes, c=opt.regimes).T
        print(f'phat:{p_hat}')
        expect = expectation.Expectation(p_hat, opt.theta_hat, opt.sigma_hat, expect.epsilon_t_T[:, [0]], regimes_)
        expect.run_expectation(initial.z_t_1, initial.delta_y_t)
        liklihood_values.append(expect.likelihood[0][0])

        # optimization

        opt = optimization.Optimization(expect.epsilon_t_T, expect.epsilon_t_T2, regimes_, initial.initial_theta_hat,
                                        initial.u_hat)
        opt.em_regime_transition(expect.result)
        opt.vec_p_hat(expect.result)
        opt.likelihood_constant_param()
        print(opt.vec_p_hat_)
        opt.likelihood(x0)
        opt.optimization(x0)
        opt.estimate_theta_hat(initial.z_t_1, initial.delta_y_t)
        print(opt.theta_hat)
    i = i + 1

print(liklihood_values[2:])
plt.figure()
plt.plot(np.array(liklihood_values[2:]))
plt.ylabel('Liklihood Values')
plt.show()
plt.savefig('convergence_oil2.png')

plt.figure(figsize=(20, 2))
plt.plot(expect.epsilon_t_T[0, :])
plt.ylabel('state 1 Smooth Probability')
plt.show()
plt.savefig('original_state1prob2.png')

plt.figure(figsize=(20, 2))
plt.plot(expect.epsilon_t_T[1, :])
plt.ylabel('state 2 Smooth Probability')
plt.show()
plt.savefig('original_ state2prob2.png')
