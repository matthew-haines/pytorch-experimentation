import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')


class Model(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()

        self.l1 = nn.Linear(input_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.l3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = self.l3(x)
        return x


pi = 3.1415926

X = torch.linspace(0, 10, 100).unsqueeze(-1)
Y = torch.sin(X)

batch_size = 32
alpha = 0.01
epochs = 1000


def main():

    x_batches = torch.split(X, batch_size)
    y_batches = torch.split(Y, batch_size)

    model = Model(X.shape[1], 100, Y.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=alpha)
    criterion = nn.MSELoss()

    losses = []

    for _ in range(epochs):
        running_loss = 0.0
        for x, y in zip(x_batches, y_batches):
            optimizer.zero_grad()
            y_hat = model(x)
            loss = criterion(y_hat, y)
            running_loss += loss
            loss.backward()
            optimizer.step()

        running_loss /= len(x_batches)
        print('Epoch: {}, Loss: {}'.format(_, running_loss))
        losses.append(running_loss)

    plt.plot(losses)
    plt.show()

    plt.plot(model(X).data.cpu().numpy(), label='predicted')
    plt.plot(Y.data.cpu().numpy(), label='actual')
    plt.legend()
    plt.show()

main()