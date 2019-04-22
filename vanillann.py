import torch
import numpy as np

if torch.cuda.is_available:
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

X = torch.linspace(-2*np.pi, 2 * np.pi, 1000)
Y = torch.sin(X)


def sigmoid(x, derivative=False):
    sigma = 1.0 / (1.0 + torch.exp(-x))
    if derivative:
        return sigma * (1.0 - sigma)

    return sigma


def relu(x, derivative=False):
    if derivative:
        x[x > 0] = 1
        x[x <= 0] = 0
        return x

    x[x < 0] = 0
    return x


def mae(y, y_hat, derivative=False):
    if derivative:
        if y_hat > y:
            return 1.0
        elif y_hat == y:
            return 0.0
        return -1.0

    return torch.abs(y_hat - y)


w1 = torch.rand(100, 1)
b1 = torch.zeros(100, 1)
w2 = torch.rand(100, 100)
b2 = torch.zeros(100, 1)
w3 = torch.rand(1, 100)
b3 = torch.zeros(1, 1)

epochs = 100
lr = 0.001

for _ in range(epochs):

    cost = 0

    for i in range(X.shape[0]):
        x = X[i].unsqueeze(-1).unsqueeze(-1)
        y = Y[i].unsqueeze(-1).unsqueeze(-1)

        z1 = w1 @ x + b1
        a1 = relu(z1)
        z2 = w2 @ a1 + b2
        a2 = relu(z2)
        y_hat = w3 @ a2 + b3

        loss = mae(y, y_hat)
        cost += loss

        loss_derivative = torch.Tensor([mae(y, y_hat, derivative=True)]).unsqueeze(-1)
        dw3 = torch.matmul(loss_derivative, torch.t(a2))
        db3 = loss_derivative
        da2 = torch.t(torch.matmul(loss_derivative, w3))
        dz2 = relu(da2, derivative=True)
        dw2 = torch.matmul(a1, torch.t(dz2))
        db2 = dz2
        da1 = torch.matmul(w2, dz2)
        dz1 = relu(da1, derivative=True)
        dw1 = torch.matmul(dz1, x)
        db1 = dz1

        w3 -= lr * dw3
        b3 -= lr * db3
        w2 -= lr * dw2
        b2 -= lr * db2
        w1 -= lr * dw1
        b1 -= lr * db1

    print('Epoch {}, Cost: {}'.format(_, cost / X.shape[0]))
