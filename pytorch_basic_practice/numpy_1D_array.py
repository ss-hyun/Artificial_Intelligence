import numpy as np

t = np.array([0., 1., 2., 3., 4., 5., 6.])
print(t)

print('Rank of t: ', t.ndim)
print('Shape of t: ', t.shape)

print('t[0] t[1] t[-1] =', t[0], t[1], t[-1])
print('t[2:5] t[4:-1] = ', t[2:5], t[4:-1])
print('t[:2] t[3:] = ', t[:2], t[3:])
