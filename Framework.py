import torch

if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')


class DenseLayer:
    """
    Standard densely connected layer
    out = activation(X matmul W + b)
    """

    def __init__(self, units, input_dim, activation=None):
        """
        Parameters:
            units: An integer that specifies the width of the output matrix
            input_dim: An integer that specifies the width of the input matrix
            activation: A function with two parameters: (input, derivative=False) that is used as activation for this layer
        """
        self.units = units
        self.input_dim = input_dim
        self.activation = activation

        self.weight = torch.randn(self.input_dim, self.units)
        self.bias = torch.zeros(1, self.units)

    def forward(self, x):
        """
        Parameters:
            x: A matrix of shape (batch_size, input_dim)

        Returns:
            A matrix of shape (batch_size, units)
        """
        self.x = x
        self.z = self.x @ self.weight + self.bias
        self.a = self.activation(self.z, derivative=False)
        return self.a

    def backward(self, da):
        """
        Parameters:
            da: The derivative of the loss w.r.t to the activated layer, shape (batch_size, units)
        Returns:
            The derivative of the loss w.r.t to the input to the layer, shape (batch_size, input_dim)
        """
        self.dz = da * self.activation(self.z, derivative=True)
        self.dw = self.x @ self.dz
        self.db = torch.sum(self.dz, dim=0)
        self.dx = self.weight @ torch.t(self.dz)

        return self.dx

    def update(self, alpha):
        """
        Parameters:
            Alpha: The learning rate of the network, each parameter p is updated as: p -= alpha * dp
        """
        self.weight -= alpha * self.dw
        self.bias -= alpha * self.db


class Model:
    """
    A neural network
    """

    def __init__(self):
        """
        Parameters:
            None
        """
        self.layers = []

    def add(self, layer):
        """
        Parameters:
            layer: A layer object that has methods forward and backward
        """
        assert self.layers[-1].units == layer.input_dim, "Previous layer output dimension ({}) does not match layer input dimension ({})".format(
            self.layers[-1].units, layer.input_dim)

        self.layers.append(layer)

    def predict(self, x):
        """
        Parameters:
            x: An input matrix of shape (batch_size, layer 0 input_dim)
        Returns:
            An output matrix of shape (batch_size, layer n units)
        """
        assert self.layers[0].input_dim == x.shape[1], "Input dimensions do not match"
        for layer in self.layers:
            x = layer.forward(x)

        return x

    def __backward__(self, dloss):
        temp = dloss
        for layer in reversed(self.layers):
            temp = layer.backward(temp)

    def train(self, X, Y, epochs, batch_size, alpha, loss_func):
        """
        Parameters:
            x: An input matrix of shape (batch_size, layer 0 input_dim)
            y: The target matrix of shape (batch_size, layer n units)
            epochs: The number of times that the network trains on the data
            batch_size: The size of each minibatch passed to the network
            alpha: The learning rate of each parameter in the network
            loss_func: A loss function with three parameters (y_hat, y, derivative=False) used as criterion in the training
        """
        assert self.layers[0].input_dim == X.shape[1], "Input dimensions do not match"
        assert X.shape[0] == Y.shape[0], "Different sample count in x and y"
        assert self.layers[-1].units == Y.shape[1], "Output dimensions do not match"
        x_batches = X.split(batch_size)
        y_batches = Y.split(batch_size)
        batch_count = len(x_batches)

        for _ in epochs:
            loss = 0.0
            for x, y in zip(x_batches, y_batches):
                y_hat = self.predict(x)
                batch_loss = loss_func(y_hat, y, derivative=False)
                loss += batch_loss
                dloss = loss_func(y_hat, y, derivative=True)
                self.__backward__(dloss)
                for layer in self.layers:
                    layer.update(alpha)

            loss /= batch_count
            print("Epoch: {}, Loss: {}".format(_, loss))


def sigmoid(x, derivative=False):
    """
    Elementwise Sigmoid activation function, σ(x) = 1 / (1 + e^(-x))
    Parameters:
        x: The input to the function
        derivative: Whether to compute to the derivative
    Returns:
        σ(x)
    """
    sigma = 1.0 / (1.0 + torch.exp(-x))
    if derivative:
        return sigma * (1.0 - sigma)

    return sigma


def relu(x, derivative=False):
    """
    Elementwise ReLU activation function. ReLU(x) = max(x, 0)
    Parameters:
        x: The input to the function
        derivative: Whether to compute to the derivative
    Returns:
        ReLU(x)
    """
    if derivative:
        x[x >= 0] = 1.0
        x[x != 1] = 0.0
        return x

    x[x < 0] = 0.0
    return x


def leakyrelu_gen(coefficient):
    """
    Creates a leakyReLU activation function using the coefficient provided
    leakyReLU(x) = max(0, x) - coefficient * max(0, -x)
    """
    def func(x, derivative=False):
        if derivative:
            x[x >= 0] = 1.0
            x[x != 1] = coefficient
            return x
        x[x < 0] *= coefficient
        return x

    return func


def mean_absolute_error(y_hat, y, derivative=False):
    """
    Loss Function
    MAE(y_hat, y) = ∑|y_hat - y| / n
    Parameters:
        y_hat: The predicted values from the neural network
        y: The actual values
        derivative: Whether to compute to the derivative
    Returns:
        A scalar loss value
    """
    if derivative:
        temp = torch.empty(y_hat.shape)
        temp[y_hat > y] = 1.0
        temp[y_hat == y] = 0.0
        temp[y_hat < y] = -1.0
        return temp

    return torch.mean(torch.abs(y_hat - y))

def mean_squared_error(y_hat, y, derivative=False):
    """
    Loss Function
    MSE(y_hat, y) = ∑(y_hat - y)² / n
    Parameters:
        y_hat: The predicted values from the neural network
        y: The actual values
        derivative: Whether to compute to the derivative
    Returns:
        A scalar loss value
    """
    if derivative:
        temp = torch.empty(y_hat.shape)
        temp[y_hat > y] = 2.0 * y_hat[y_hat > y]
        temp[y_hat == y] = 0.0
        temp[y_hat < y] = 2.0 * y_hat[y_hat < y]
        return temp

    return torch.mean((y_hat - y) ** 2)
