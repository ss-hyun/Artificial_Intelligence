import numpy as np
import matplotlib.pyplot as plt


def small(Din, Dout):
    return 0.01 * np.random.randn(Din, Dout)


def large(Din, Dout):
    return 0.05 * np.random.randn(Din, Dout)


def Xavier(Din, Dout):
    return np.random.randn(Din, Dout) * np.sqrt(2/Din)


def ReLU(x):
    return np.maximum(0, x)


active = {
    small: np.tanh,
    large: np.tanh,
    Xavier: ReLU
}

fig = plt.figure(figsize=(13, 9))
size = (2, 2)

for idx, label in enumerate([small, large, Xavier]):
    dims = [4096] * 7
    hs = []
    x = np.random.randn(16, dims[0])
    for Din, Dout in zip(dims[:-1], dims[1:]):
        W = label(Din, Dout)
        x = active[label](x.dot(W))
        hs.append(x)

    ax = fig.add_subplot(size[0], size[1], idx + 1)
    for i in range(0, len(hs)):
        H, bins = np.histogram(hs[i], bins=20)
        # print(H, bins)
        ax.plot(bins, np.insert(H, 0, H[0]))
    ax.legend("0123456")

fig.tight_layout()
plt.show()
