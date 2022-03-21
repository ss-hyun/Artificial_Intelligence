import numpy as np
from numpy.random import randn

# Define the network
N, D_in, H, D_out = 64, 1000, 100, 10
x, y = randn(N, D_in), randn(N, D_out) # x : 64*1000 , y : 64*10
w1, w2 = randn(D_in, H), randn(H, D_out) # w1 : 1000*100 , w2 : 100*10

for t in range(10000):
    # Forward pass
    h = 1 / (1 + np.exp(-x.dot(w1))) # h : 64*100
    y_pred = h.dot(w2) # y_pred : 64*10
    loss = np.square(y_pred - y).sum()
    if t % 100 == 0:
        print(t, loss)

    # Calculate the analytical gradients
    grad_y_pred = 2.0 * (y_pred - y) # grad_y_pred : 64*10
    grad_w2 = h.T.dot(grad_y_pred) # grad_w2 : 100*10, dot operation - h transpose & grad_y_pred
    grad_h = grad_y_pred.dot(w2.T) # grad_h : 64*100, dot operation with w2 transpose array
    grad_w1 = x.T.dot(grad_h * h * (1 - h)) # grad_w1 : 1000*100

    # Gradient descent
    w1 -= 1e-4 * grad_w1
    w2 -= 1e-4 * grad_w2
