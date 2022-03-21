import numpy as np
import torch

# Multiplication
print('---------------')
print(' Mul vs Matmul ')
print('---------------')
m1 = torch.FloatTensor([[1, 2], [3, 4]])
m2 = torch.FloatTensor([[1], [2]])
print('Shape of Matrix 1: ', m1.shape) # 2 x 2
print('Shape of Matrix 2: ', m2.shape) # 2 x 1

print('\nMultiplication')
print('mul function :\n', m1.mul(m2)) # 2 x 2
print('operation * :\n', m1 * m2)

print('\nMatrix Multiplication')
print('matmul function :\n', m1.matmul(m2)) # 2 x 1

# Mean
print('\n------Mean------')
t = torch.FloatTensor([1, 2])
print('tensor: ', t)
print('mean(): ', t.mean())

print("\nCan't use mean() ont integers")
t = torch.LongTensor([1, 2])
print('tensor: ', t)
try:
    print(t.mean())
except Exception as exc:
    print(exc)

t = torch.FloatTensor([[1, 2], [3, 4]])
print('\ntensor:')
print(t)
print('mean(): ', t.mean())
print('mean(dim=0): ', t.mean(dim=0)) # remove dimension 0
print('mean(dim=1): ', t.mean(dim=1)) # remove dimension 1
print('mean(dim=-1): ', t.mean(dim=-1)) # dim=-1 -> remove last dimension

# Mean
print('\n------Sum------')
t = torch.FloatTensor([[1, 2], [3, 4]])
print('tensor:')
print(t)
print('sum(): ', t.sum())
print('sum(dim=0): ', t.sum(dim=0))
print('sum(dim=1): ', t.sum(dim=1))
print('sum(dim=-1): ', t.sum(dim=-1))

# Max & ArgMax
print('\n--------------')
print(' Max & Argmax ')
print('--------------')
t = torch.FloatTensor([[1, 2], [3, 4]])
print('tensor:')
print(t)
print('max(): ', t.max()) # returns one value: max

print('\nmax(dim=0):\n', t.max(dim=0)) # returns two values: max & argmax
print('\nMax: ', t.max(dim=0)[0], '- max(dim=0)[0]') # maximum value
print('Argmax: ', t.max(dim=0)[1], '- max(dim=0)[1]') # index of the element with maximum value

print('\nmax(dim=1):\n', t.max(dim=1))
print('\nmax(dim=-1):\n', t.max(dim=-1))