import numpy as np
import matplotlib.pyplot as plt

dims = [4096] * 7
hs = []
x = np.random.randn(16, dims[0])
for Din, Dout in zip(dims[:-1], dims[1:]):
    # small random numbers
    W = 0.01 * np.random.randn(Din, Dout)
    print(W.shape)
    x = np.tanh(x.dot(W))
    print(x.shape)
    hs.append(x)

fig, ax = plt.subplots()
print(fig, ax)
for i in range(0, len(hs)):
    H, bins = np.histogram(hs[i], bins=20)
    print(H, bins)
    ax.plot(bins, np.insert(H, 0, H[0]))
ax.legend("0123456")
plt.show()
