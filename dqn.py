import gym
import torch
from torch import nn
import torch.nn.functional as F
from collections import deque
import random
import numpy as np
import matplotlib.pyplot as plt

if torch.cuda.is_available:
    torch.set_default_tensor_type('torch.cuda.FloatTensor')


def mean(x):
    return sum(x) / float(len(x))


class Network(nn.Module):

    def __init__(self, input_dim, output_dim, optimizer, criterion):
        super().__init__()

        self.l1 = nn.Linear(input_dim, 256)
        self.l2 = nn.Linear(256, 2)

        self.optimizer = optimizer(self.parameters(), lr=0.001)
        self.criterion = criterion()

    def forward(self, x):
        if type(x) == np.ndarray:
            x = torch.from_numpy(x).type('torch.cuda.FloatTensor')
        else:
            x = x.type('torch.cuda.FloatTensor')
        x = torch.relu(self.l1(x))
        x = self.l2(x)

        return x

    def train(self, x, y):
        x = x.to('cuda')
        y = y.to('cuda')
        self.optimizer.zero_grad()
        y_hat = self.forward(x)
        loss = self.criterion(y_hat, y)
        loss.backward()
        self.optimizer.step()


class DQN:

    def __init__(self, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01, n_episodes=20000, gamma=0.99, batch_size=128):
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.n_episodes = n_episodes
        self.gamma = gamma
        self.batch_size = batch_size

        self.replay_memory = deque(maxlen=10000)
        self.network = Network(4, 2, torch.optim.Adam, nn.MSELoss)

        self.scores = []
        self.running_means = []

        self.env = gym.make('CartPole-v0')

    def remember(self, state, action, reward, next_state, done):
        self.replay_memory.append((state, action, reward, next_state, done))

    def replay(self):
        x_batch = []
        y_batch = []
        minibatch = random.sample(self.replay_memory, min(
            self.batch_size, len(self.replay_memory)))
        for state, action, reward, next_state, done in minibatch:
            y_hat = self.network.forward(state)
            y_hat[action] = reward if done else reward + self.gamma * \
                torch.max(self.network.forward(next_state))
            y_batch.append(y_hat)
            x_batch.append(torch.from_numpy(state))

        self.network.train(torch.stack(x_batch), torch.stack(y_batch))
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def run(self):
        for i in range(self.n_episodes):
            state = self.env.reset()
            done = False
            score = 0
            while not done:
                if random.random() < self.epsilon:
                    action = random.randint(0, 1)
                else:
                    action = torch.argmax(self.network.forward(state)).item()

                next_state, reward, done, _ = self.env.step(action)
                self.remember(state, action, reward, next_state, done)
                state = next_state
                score += 1

            self.scores.append(score)

            if len(self.scores) > 100:
                running_mean = mean(self.scores[-100:-1])
                self.running_means.append(running_mean)
                if i % 100 == 0:
                    print("Episode: {}, Score: {}, Epsilon: {}".format(
                        i, running_mean, self.epsilon))

            self.replay()


if __name__ == '__main__':
    agent = DQN()
    agent.run()
    plt.plot(agent.scores)
    plt.plot(agent.running_means)
    plt.show()
