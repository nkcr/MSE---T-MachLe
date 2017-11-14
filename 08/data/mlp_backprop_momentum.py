import numpy as np

class MLP:
    """
    This code was adapted from:
    https://rolisz.ro/2013/04/18/neural-networks-in-python/
    """
    def __tanh(self, x):
        return np.tanh(x)

    def __tanh_deriv(self, a):
        return 1.0 - a**2

    def __logistic(self, x):
        return 1.0 / (1.0 + np.exp(-x))

    def __logistic_derivative(self, a):
        return a * ( 1 - a )
    
    def __init__(self, layers, activation='tanh'):
        """
        :param layers: A list containing the number of units in each layer.
        Should be at least two values
        :param activation: The activation function to be used. Can be
        "logistic" or "tanh"
        """
        if activation == 'logistic':
            self.activation = self.__logistic
            self.activation_deriv = self.__logistic_derivative
        elif activation == 'tanh':
            self.activation = self.__tanh
            self.activation_deriv = self.__tanh_deriv

        self.weights = []
        for i in range(1, len(layers) - 1):
            self.weights.append((2 * np.random.random((layers[i - 1] + 1, layers[i] + 1)) - 1) * 0.25)
        self.weights.append((2 * np.random.random((layers[i] + 1, layers[i + 1])) - 1) * 0.25)
    
        self.layers = layers

    def fit(self, X, y, learning_rate=0.1, momentum=0.5, epochs=100):
        """
        Online learning.
        :param X: Input data or features
        :param y: Input targets
        :param learning_rate: parameters defining the speed of learning
        :param epochs: number of times the dataset is presented to the network for learning
        """
        X = np.atleast_2d(X)
        temp = np.ones([X.shape[0], X.shape[1]+1])
        temp[:, 0:-1] = X  # adding the bias unit to the input layer
        X = temp
        y = np.array(y)
        to_return = np.zeros(epochs)

        for k in range(epochs):
            temp = np.zeros(X.shape[0])
            old_deltas = [0, 0]
            for it in range(X.shape[0]):
                i = np.random.randint(X.shape[0])
                a = [X[i]]

                for l in range(len(self.weights)):
                    a.append(self.activation(np.dot(a[l], self.weights[l])))
                error = y[i] - a[-1]
                temp[it] = error ** 2
                deltas = [error * self.activation_deriv(a[-1])]

                for l in range(len(a) - 2, 0, -1): # we need to begin at the second to last layer
                    deltas.append(deltas[-1].dot(self.weights[l].T) * self.activation_deriv(a[l]))
                deltas.reverse()

                for i in range(len(self.weights)):
                    layer = np.atleast_2d(a[i])
                    delta = np.atleast_2d(deltas[i])
                    old_delta = np.atleast_2d(old_deltas[i])
                    self.weights[i] += learning_rate * layer.T.dot(delta) + momentum * layer.T.dot(old_delta)

                old_deltas = deltas
            to_return[k] = np.mean(temp)
        return to_return

    def predict(self, x):
        x = np.array(x)
        temp = np.ones(x.shape[0]+1)
        temp[0:-1] = x
        a = temp
        for l in range(0, len(self.weights)):
            a = self.activation(np.dot(a, self.weights[l]))
        return a

    def init_weights(self):
        self.weights = []
        for i in range(1, len(self.layers) - 1):
            self.weights.append((2 * np.random.random((self.layers[i - 1] + 1, self.layers[i] + 1)) - 1) * 0.25)
        self.weights.append((2 * np.random.random((self.layers[i] + 1, self.layers[i + 1])) - 1) * 0.25)
