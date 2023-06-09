import initialization
import data
import numpy as np

df = data.df
b0 = np.random.normal(0, 0.1e-4, size=(4, 4))
b0 = np.zeros([4, 4])  # this ensures there's no random initial values comment this off when running the code
beta_ = np.array([0, 0, 0, 1]).reshape(-1, 1)
initial = initialization.Initializes(df, 1, 2, b0, beta=beta_)
initial.data_matrix()
initial.initial_params()
initial.initial_b_matrix()
print('ok')
