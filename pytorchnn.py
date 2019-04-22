import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

class Model(nn.Module):

    def __init__(self):
        super().__init__()

        self.l1 = nn.Linear(1, 100)
        self.l2 = nn.Linear(100, 100)
        self.out = nn.Linear(100, 1)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = self.out(x)
        return x

pi = 3.1415926

X = torch.linspace(-2 * pi, 2 * pi, 1000).unsqueeze(-1)
Y = torch.sin(X)

lr = 0.00025
epochs = 100

model = Model()

optimizer = torch.optim.Adam(model.parameters(), lr=0.0025)
loss_func = nn.MSELoss()

losses = []

for i in range(epochs):
    optimizer.zero_grad()
    y_hat = model(X)
    loss = loss_func(y_hat, Y)
    losses.append(loss.data)
    loss.backward()
    optimizer.step()

plt.plot(losses)
plt.show()