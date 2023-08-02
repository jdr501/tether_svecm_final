import multiprocessing

import initialization
import expectation
import optimization
import matrix_operations as mo
import data
import numpy as np
import matplotlib.pyplot as plt
import json
import codecs

lag_ = 1
regimes_ = 2
# initialization
df = data.df
draws = 10
b0_list = []

for i in range(draws):
    b0 = np.random.normal(0, 0.1, size=(4, 4))
    b0_list.append(b0)

beta_ = np.array([0, 0, 0, 1]).reshape(-1, 1)

results = {}


def draw(b0):
    likelihood_values = [-10, -8]
    i = 0
    while np.abs(likelihood_values[i - 1] - likelihood_values[i - 2]) > 1e-4:
        print(f'THIS IS THE ITERATION NUMBER :{i}')

        if i == 0:
            initial = initialization.Initializes(df, lag_, regimes_, b0, beta=beta_)
            initial.run_initialization()
            print(initial.initial_b)

            # expectation
            expect = expectation.Expectation(initial.p, initial.initial_theta_hat, initial.sigmas, initial.e_0_0,
                                             regimes_)
            expect.run_expectation(initial.z_t_1, initial.delta_y_t)
            print(expect.epsilon_t_T)
            print(expect.epsilon_t_T2)
            print(f'this is likelihood values:{expect.likelihood}')
            likelihood_values.append(expect.likelihood[0][0])
            # optimization

            opt = optimization.Optimization(expect.epsilon_t_T, expect.epsilon_t_T2, regimes_,
                                            initial.initial_theta_hat,
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
            likelihood_values.append(expect.likelihood[0][0])

            # optimization

            opt = optimization.Optimization(expect.epsilon_t_T, expect.epsilon_t_T2, regimes_,
                                            initial.initial_theta_hat,
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

    temp = {'likelihood_values': likelihood_values[2:], 'epsilon_t_T': expect.epsilon_t_T.tolist(),
            'likelihood': likelihood_values[-1]}
    return temp


d = 0
with multiprocessing.Pool() as pool:
    # call the function for each item in parallel
    for result in pool.map(draw, b0_list):
        results.update({f'{d}': result})
        d += 1

file_path = "/results.json"

json.dump(results, codecs.open(file_path, 'w', encoding='utf-8'),
          separators=(',', ':'),
          sort_keys=True,
          indent=4)

#with open("results.json", "w") as outfile:
    #json.dump(results, outfile, sort_keys=True, indent=4)
""" 
print(likelihood_values[2:])
plt.figure()
plt.plot(np.array(likelihood_values[2:]))
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

"""
