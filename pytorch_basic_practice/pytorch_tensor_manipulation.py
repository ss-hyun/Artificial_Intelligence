import numpy as np
import torch

# View (Reshape)
# - resizing the tensor while maintaining the number of elements
print('------View------')
t = np.array(
    [
        [
            [0, 1, 2],
            [3, 4, 5]
        ],
        [
            [6, 7, 8],
            [9, 10, 11]
        ]
    ]
)
ft = torch.FloatTensor(t)
print(ft.shape)

print(ft.view([-1, 3]))
print(ft.view([-1, 3]).shape)

print(ft.view([-1, 1, 3]))
print(ft.view([-1, 1, 3]).shape)


# Squeeze
# - remove dimension of (size 1)
print('\n------Squeeze------')
ft = torch.FloatTensor([[0], [1], [2]])
print(ft)
print(ft.shape)

print(ft.squeeze())
print(ft.squeeze().shape)

ft = torch.FloatTensor([[[0]], [[1]], [[2]]])
print(ft)
print(ft.shape)

print(ft.squeeze())
print(ft.squeeze().shape)

# Unsqueeze
# - add (size 1) dimensions to a specific location
print('\n------Unsqueeze------')
ft = torch.Tensor([0, 1, 2])
print(ft)
print(ft.shape)

print(ft.unsqueeze(0))
print(ft.unsqueeze(0).shape)

print(ft.view(1, -1))
print(ft.view(1, -1).shape)

print(ft.unsqueeze(1))
print(ft.unsqueeze(1).shape)

print(ft.unsqueeze(-1))
print(ft.unsqueeze(-1).shape)

# Type Casting
print('\n------Type Casting------')
lt = torch.LongTensor([1, 2, 3, 4])
print(lt)

print(lt.float())

bt = torch.ByteTensor([True, False, False, True])
print(bt)

print(bt.long())
print(bt.float())

# Concatenate
print('\n------Concatenate------')
x = torch.FloatTensor([[1, 2], [3, 4]])
y = torch.FloatTensor([[5, 6], [7, 8]])

print(torch.cat([x, y], dim=0))
print(torch.cat([x, y], dim=1))

# Stacking
print('\n------Stacking------')
x = torch.FloatTensor([1, 4])
y = torch.FloatTensor([2, 5])
z = torch.FloatTensor([3, 6])

print(torch.stack([x, y, z]))
print(torch.stack([x, y, z], dim=1))

print(torch.cat([x.unsqueeze(0), y.unsqueeze(0), z.unsqueeze(0)], dim=0))

# Ones and Zeros
# - tensor filled with 0/1
print('\n------Ones and Zeros------')
x = torch.FloatTensor([[0, 1, 2], [2, 1, 0]])
print(x)

print(torch.ones_like(x))
print(torch.zeros_like(x))

# In-place Operation
print('\n------In-place Operation------')
x = torch.FloatTensor([[1, 2], [3, 4]])

print(x.mul(2.))
print(x)
print(x.mul_(2.))
print(x)
