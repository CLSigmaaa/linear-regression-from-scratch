import numpy as np
import matplotlib.pyplot as plt

class LinearRegression:
    def __init__(self, x, y, theta):
        self.cost_history = []
        if isinstance(x, np.ndarray) and isinstance(y, np.ndarray) and isinstance(theta, np.ndarray):
            self.theta = theta
            if x.shape[0] == y.shape[0]:
                self.x = x
                self.y = y
            else:
                raise ValueError("x and y must have the same number of rows")
        else:
            raise ValueError("x, y, and theta must be numpy arrays")

    def model(self, x, theta):
        return np.dot(x, theta)

    def cost(self, x, y, theta):
        m = len(x)
        return 1/(2*m) * np.sum((self.model(x, theta)-y)**2)
    
    def gradient(self, x, y, theta):
        m = len(x)
        return 1/m * x.T.dot(self.model(x, theta) - y)
    
    def gradient_descent(self, theta, x, y, learning_rate, n_iterations):
        cost_history = np.zeros(n_iterations)
        for i in range(0, n_iterations):
            theta = theta - learning_rate * self.gradient(x, y, theta)
            cost_history[i] = self.cost(x, y, theta)
        self.theta = theta
        self.cost_history = cost_history
        return theta, cost_history
    
    def predict(self, x):
        return self.model(x, self.theta)
    
    def show_learning_curve(self):
        plt.plot(range(len(self.cost_history)), self.cost_history)
        plt.xlabel("Number of iterations")
        plt.ylabel("Cost")
        plt.show()

    def show_model(self, feature_column):
        plt.scatter(self.x[:, feature_column], self.y)
        plt.plot(self.x[:, feature_column], self.predict(self.x), color='r')
        plt.show()