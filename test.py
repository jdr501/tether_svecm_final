import matrix_operations as mo
import numpy as np

matrix = np.arange(1,17).reshape([4,4])
print(matrix)
print(mo.mat_vec(matrix))
print(mo.vec_matrix(mo.mat_vec(matrix)))



det =    b13*(b24*(b31*b42 - b32*b41) - b22*(b31*b44 - b34*b41) + b21*(b32*b44 - b34*b42))
 - b14*(b23*(b31*b42 - b32*b41) - b22*(b31*b43 - b33*b41) + b21*(b32*b43 - b33*b42))
 - b12*(b24*(b31*b43 - b33*b41) - b23*(b31*b44 - b34*b41) + b21*(b33*b44 - b34*b43))
 + b11*(b24*(b32*b43 - b33*b42) - b23*(b32*b41 - b34*b42) + b22*(b33*b44 - b34*b43))


