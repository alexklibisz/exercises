import numpy as np

class SimpleNN:

    def __init__(self, layer_dims=np.array([3,3,1])):
        # Take layer dims as given.
        self.layer_dims = layer_dims
        return

    def _sigmoid_activation(self, z):
        return 1 / (1 + np.exp(-z))

    def _sigmoid_activation_gradietn(self, z):
        g = self._sigmoid_activation
        return np.multiply(g(z), (1 - g(z)))

    def _forward_prop(self, X, Theta):
        return

    def _compute_cost(self, X, y, Theta, learn_rate = 0.001):
        return

    def _back_prop(self, X, y, Theta, learn_rate = 0.001):
        return

    def train(self, X, y):

        # Initialize random theta based on layer_dims.
        self.Theta = np.array([None] * len(self.layer_dims))
        for i in range(len(self.layer_dims) - 1):
            n = self.layer_dims[i]
            m = self.layer_dims[i+1]
            print(n,m)
            self.Theta[i] = i

        print(self.Theta)

        return

    def evaluate(self, X):

        return
