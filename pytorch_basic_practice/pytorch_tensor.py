import numpy as np
import torch

# 1D Array with PyTorch
t = torch.Tensor([0., 1., 2., 3., 4., 5., 6.])
print(t)

print(t.dim())  # rank
print(t.shape)  # shape
print(t.size()) # shape
print('t[0] t[1] t[-1] =', t[0], t[1], t[-1])   # element
print('t[2:5] t[4:-1] = ', t[2:5], t[4:-1])     # slicing
print('t[:2] t[3:] = ', t[:2], t[3:])           # slicing

# 2D Array with PyTorch
t = torch.FloatTensor([ [1., 2., 3.],
                        [4., 5., 6.],
                        [7., 8., 9.],
                        [10., 11., 12.]
                    ])
print(t)

print(t.dim())  # rank
print(t.size()) # shape
print(t[:, 1])
print(t[:, 1].size())
print(t[:, :-1])

# Broadcasting
## same shape
m1 = torch.FloatTensor([[3, 3]])
m2 = torch.FloatTensor([[2, 2]])
print(m1 + m2)

## vector + scalar
m1 = torch.FloatTensor([[1, 2]])
m2 = torch.FloatTensor([3]) # 3 -> [[3, 3]]
print(m1 + m2)

## 2x1 vector + 1x2 vector
m1 = torch.FloatTensor([[1, 2]]) # [[1, 2]] -> [[1, 2], [1, 2]]
m2 = torch.FloatTensor([[3], [4]]) # [[3],[4]] -> [[3, 3], [4, 4]]
print(m1 + m2)
